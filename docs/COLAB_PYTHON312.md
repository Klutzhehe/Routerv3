# Running the PNS-bridge stack on Colab (Python 3.12)

The cached build in `MyDrive/routerv3_cache/kicad-src-9.0.8.tar.gz` is a KiCad
source **and build** tree, and the bridge inside it is

    kicad-src/build/pcbworld_bridge/pcbworld_pns_bridge.cpython-312-x86_64-linux-gnu.so

`cpython-312` is a hard version lock. Colab's notebook kernel is now **Python
3.13**, which never even considers that filename when resolving the import --
`ModuleNotFoundError`, with the file sitting right there on disk.

Rebuilding against 3.13 is one answer. This is the cheaper one: **side-install
3.12 and run the bridge under it.** Measured end to end at ~100 s, against a
full KiCad compile.

## Setup

```bash
# 1. the cache (source + prebuilt bridge)
!tar -xzf /content/drive/MyDrive/routerv3_cache/kicad-src-9.0.8.tar.gz -C /content

# 2. python 3.12
!add-apt-repository --yes ppa:deadsnakes/ppa
!apt-get -qq update
!DEBIAN_FRONTEND=noninteractive apt-get -qq install -y python3.12 python3.12-dev

# 3. pip -- deadsnakes omits ensurepip, so bootstrap it
!curl -sS https://bootstrap.pypa.io/get-pip.py -o /content/get-pip.py
!python3.12 /content/get-pip.py -q

# 4. deps. CPU torch on purpose: the line policy is ~64k params and
#    CPU-resident by design (docs/UNIFIED_RL_DESIGN.md).
!python3.12 -m pip install -q numpy gymnasium
!python3.12 -m pip install -q torch --index-url https://download.pytorch.org/whl/cpu
```

## Running

Everything goes through `python3.12`, never the notebook kernel:

```bash
%env PYTHONPATH=/content/Routerv3:/content/kicad-src/build/pcbworld_bridge
!python3.12 -u /content/Routerv3/scripts/smoke_line_env.py /content/board.kicad_pcb
```

**The notebook kernel itself can never import the bridge.** It is 3.13; the
`.so` is 3.12. Anything touching the router has to be a `python3.12`
subprocess. That is fine for training and eval, which are scripts anyway, and
awkward only for interactive poking.

## Verified on this setup

    python3.12 pcbworld/data/generate_board.py b24.kicad_pcb --num-nets 24 --seed 0
      -> wrote a real 24-net board (drives the bridge)

    python3.12 scripts/smoke_line_env.py b24.kicad_pcb
      -> greedy (a=0) routed 9/24, matching the independently measured
         straight-line baseline of 9/24  ->  verdict HEALTHY

## Reading smoke_line_env's "no gradient" warning

It compares greedy (a=0) against random on **episode reward** and warns when
random does no worse. Measured across densities, 3 seeds each:

| nets | greedy | random |
|---|---|---|
| 1 | 3/3 | 2/3 |
| 4 | 7/12 | 6/12 |
| 8 | 10/24 | 11/24 |
| 24 | 22/72 | 23/72 |

The prior separates from random at 1-4 nets and loses its edge from 8 up,
where both fail ~60%. That is the documented baseline (`UNIFIED_RL_DESIGN.md`:
33% straight-line at 8 nets), not a broken reward -- on a dense board, walking
straight at the pad genuinely is no better than wandering, which is the entire
reason the geodesic potential exists. Treat the warning as "the prior has no
headroom here", not "there is no signal".

Note also that `a = 0` is straight at the **pad**, not down the geodesic
gradient. The field reaches the policy through the reward and through the
`geo_dir` / `clearance` observation features, so this comparison measures the
prior's headroom, not the field's value.
