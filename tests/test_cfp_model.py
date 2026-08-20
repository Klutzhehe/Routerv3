"""Smoke tests for the CFP policy network.

These check plumbing only -- shapes, masking, gradient flow, and that
act()/evaluate_actions() agree on the same action. They say nothing about
routing quality, because no router is involved: the whole point of
pcbworld/agents/cfp/ is that it depends on neither pcbworld_pns_bridge nor
pcbnew, so unlike tests/test_*_env.py these run locally with no Colab build.

A deliberately small CFPConfig is used throughout -- the default (~14M
params, 256x256 canvas) is a slow unit test for no extra coverage.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pcbworld.agents.cfp import (
    NUM_CANVAS_CHANNELS,
    NUM_NET_FEATURES,
    CFPConfig,
    CFPPolicy,
    EDGE_TYPE,
    build_edge_type_matrix,
    make_dummy_observation,
)
from pcbworld.agents.cfp.spec import ACTION_RIPUP, ACTION_ROUTE, parse_net_name

TINY = CFPConfig(
    dim=32,
    num_heads=4,
    net_layers=2,
    fusion_rounds=1,
    canvas_base_channels=8,
    canvas_blocks_per_stage=1,
    field_size=8,
)
CANVAS = 64  # smallest sane canvas: 64 / 16 = a 4x4 token grid


@pytest.fixture
def policy() -> CFPPolicy:
    torch.manual_seed(0)
    return CFPPolicy(TINY)


@pytest.fixture
def obs():
    return make_dummy_observation(batch_size=3, num_nets=9, canvas_size=CANVAS, seed=1)


# -- spec ------------------------------------------------------------------


def test_parse_net_name_matches_generate_board_convention():
    assert parse_net_name("net_3") == ("plain", None, None)
    assert parse_net_name("diffpair_2_P") == ("diff_pair", "diffpair_2", "P")
    assert parse_net_name("lengthgrp_1_4") == ("length_group", "lengthgrp_1", "4")


def test_edge_matrix_encodes_constraints_not_just_proximity():
    names = ["diffpair_0_P", "diffpair_0_N", "lengthgrp_0_0", "lengthgrp_0_1", "net_0"]
    # Put the two diff-pair legs far apart and the unrelated net right next
    # to one of them, so proximity alone would give the wrong answer.
    centroids = np.array([[0.0, 0.0], [9.0, 9.0], [1.0, 1.0], [5.0, 5.0], [0.1, 0.0]])
    bboxes = np.concatenate([centroids - 0.01, centroids + 0.01], axis=1)
    edges = build_edge_type_matrix(names, centroids, bboxes, k_spatial=2)

    assert edges[0, 1] == EDGE_TYPE.DIFF_PAIR and edges[1, 0] == EDGE_TYPE.DIFF_PAIR
    assert edges[2, 3] == EDGE_TYPE.LENGTH_GROUP and edges[3, 2] == EDGE_TYPE.LENGTH_GROUP
    assert np.all(np.diag(edges) == EDGE_TYPE.SELF)
    # The constraint survives distance; the near-neighbor gets a weaker edge.
    assert edges[0, 4] in (EDGE_TYPE.SPATIAL_KNN, EDGE_TYPE.BBOX_CONFLICT)


def test_overlapping_bboxes_become_conflict_edges():
    names = ["net_0", "net_1"]
    centroids = np.array([[0.0, 0.0], [0.5, 0.0]])
    bboxes = np.array([[-1.0, -1.0, 1.0, 1.0], [0.0, -1.0, 2.0, 1.0]])
    edges = build_edge_type_matrix(names, centroids, bboxes, k_spatial=1)
    assert edges[0, 1] == EDGE_TYPE.BBOX_CONFLICT


def test_dummy_observation_is_self_consistent(obs):
    obs.validate()
    assert obs.canvas.shape == (3, NUM_CANVAS_CHANNELS, CANVAS, CANVAS)
    assert obs.net_feats.shape[-1] == NUM_NET_FEATURES


# -- forward pass ----------------------------------------------------------


def test_encode_shapes(policy, obs):
    encoded = policy.net.encode(obs)
    b, n = obs.batch_size, obs.num_nets
    grid = CANVAS // 16
    assert encoded.net_h.shape == (b, n, TINY.dim)
    assert encoded.canvas_map.shape == (b, TINY.dim, grid, grid)
    assert encoded.pointer_logits.shape == (b, 2 * n)
    assert encoded.value.shape == (b,)
    assert torch.isfinite(encoded.value).all()


def test_field_is_conditioned_on_the_selected_net(policy, obs):
    """FiLM must actually make the field net-specific -- if it didn't, the
    architecture would degenerate to one global map per step."""
    encoded = policy.net.encode(obs)
    # Untrained FiLM is initialized to the identity, so train it a step
    # first; otherwise this passes vacuously in both directions.
    torch.nn.init.normal_(policy.net.field_film.to_scale_shift.weight, std=0.5)

    mean_a, _ = policy.net.field_params(encoded, torch.zeros(obs.batch_size, dtype=torch.long))
    mean_b, _ = policy.net.field_params(encoded, torch.ones(obs.batch_size, dtype=torch.long))
    assert mean_a.shape == (obs.batch_size, TINY.num_field_planes, TINY.field_size, TINY.field_size)
    assert not torch.allclose(mean_a, mean_b)


def test_field_has_a_reserve_plane_per_layer_config():
    assert CFPConfig(num_copper_layers=2).num_field_planes == 3
    assert CFPConfig(num_copper_layers=4).num_field_planes == 5


# -- masking ---------------------------------------------------------------


def test_illegal_actions_get_zero_probability(policy, obs):
    encoded = policy.net.encode(obs)
    probs = torch.softmax(encoded.pointer_logits, dim=-1)
    illegal = ~obs.action_mask.flatten(1)
    assert torch.allclose(probs[illegal], torch.zeros_like(probs[illegal]), atol=1e-12)
    assert torch.allclose(probs.sum(-1), torch.ones(obs.batch_size), atol=1e-5)


def test_sampling_never_returns_a_padded_or_illegal_slot(policy, obs):
    valid = obs.action_mask.flatten(1)
    for _ in range(20):
        action = policy.act(obs)
        assert valid.gather(1, action.action_index[:, None]).all()
        kind, slot = action.split(obs.num_nets)
        assert bool(obs.net_mask.gather(1, slot[:, None]).all())
        assert set(kind.tolist()) <= {ACTION_ROUTE, ACTION_RIPUP}


def test_padded_net_slots_do_not_leak_into_the_value_head(policy, obs):
    """Junk in a padding slot must not change any output."""
    baseline = policy.net.encode(obs)
    perturbed = make_dummy_observation(
        batch_size=obs.batch_size, num_nets=obs.num_nets, canvas_size=CANVAS, seed=1
    )
    perturbed.net_feats[:, -1] = 1e3  # last slot is masked out by construction
    perturbed.net_xy[:, -1] = 0.99
    after = policy.net.encode(perturbed)
    torch.testing.assert_close(baseline.value, after.value, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(baseline.pointer_logits, after.pointer_logits, rtol=1e-4, atol=1e-5)


def test_entropy_and_log_prob_are_finite_under_heavy_masking(policy):
    obs = make_dummy_observation(batch_size=2, num_nets=16, canvas_size=CANVAS, seed=7)
    obs.action_mask[:] = False
    obs.action_mask[:, ACTION_ROUTE, 0] = True  # exactly one legal action
    action = policy.act(obs)
    assert torch.isfinite(action.log_prob).all()
    assert torch.isfinite(action.score.entropy).all()
    # One legal action -> the categorical is deterministic -> zero entropy.
    torch.testing.assert_close(
        action.score.cat_entropy, torch.zeros_like(action.score.cat_entropy), atol=1e-5, rtol=0
    )
    assert (action.action_index == 0).all()


# -- PPO-side consistency ---------------------------------------------------


def test_evaluate_actions_reproduces_act_log_prob(policy, obs):
    action = policy.act(obs)
    score = policy.evaluate_actions(obs, action.action_index, action.field)
    torch.testing.assert_close(score.log_prob, action.log_prob, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(score.cat_entropy, action.score.cat_entropy, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(score.field_entropy, action.score.field_entropy, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(score.value, action.value, rtol=1e-4, atol=1e-5)


def test_entropy_terms_are_separable_and_differently_scaled(policy, obs):
    """The whole reason CFPScore splits them: one PPO entropy coefficient
    cannot serve a ~2-nat categorical and a ~700-nat Gaussian at once."""
    action = policy.act(obs)
    cat, field = action.score.cat_entropy, action.score.field_entropy
    assert (field.abs() > 10 * cat.abs()).all()
    torch.testing.assert_close(action.score.entropy, cat + field)


def test_deterministic_act_is_reproducible(policy, obs):
    a = policy.act(obs, deterministic=True)
    b = policy.act(obs, deterministic=True)
    assert torch.equal(a.action_index, b.action_index)
    torch.testing.assert_close(a.field, b.field)


def test_planner_field_is_clamped_but_scored_action_is_not(policy, obs):
    action = policy.act(obs)
    action.field[0, 0, 0, 0] = 1e6
    assert action.planner_field().abs().max() <= 4.0 + 1e-6
    assert action.field.abs().max() > 4.0


def test_gradients_reach_both_towers_and_all_heads(policy, obs):
    action = policy.act(obs)
    score = policy.evaluate_actions(obs, action.action_index, action.field)
    loss = (
        -score.log_prob.mean()
        + score.value.pow(2).mean()
        - 0.01 * score.cat_entropy.mean()
        - 1e-4 * score.field_entropy.mean()
    )
    loss.backward()

    grads = {name: p.grad for name, p in policy.named_parameters()}
    missing = [name for name, g in grads.items() if g is None or not torch.isfinite(g).all()]
    assert not missing, f"no/invalid gradient for: {missing}"

    # Spot-check that each structurally distinct part actually moved.
    for key in [
        "net.canvas_encoder",
        "net.net_embed",
        "net.net_blocks.0.attn.edge_bias",
        "net.net_from_canvas.0",
        "net.canvas_from_net.0",
        "net.pointer_head",
        "net.field_out",
        "net.field_log_std",
        "net.value_head",
    ]:
        touched = [g.abs().sum().item() for name, g in grads.items() if name.startswith(key)]
        assert touched and max(touched) > 0.0, f"{key} received no gradient signal"


def test_edge_bias_gradient_is_nonzero_for_present_edge_types(policy, obs):
    """The relational bias is the mechanism that carries diff-pair and
    length-group structure; if its gradient is dead the graph tower is
    decorative."""
    encoded = policy.net.encode(obs)
    encoded.value.sum().backward()
    bias_grad = policy.net.net_blocks[0].attn.edge_bias.weight.grad
    present = torch.unique(obs.edge_type)
    assert bias_grad is not None
    assert bias_grad[present].abs().sum() > 0.0


# -- sizing -----------------------------------------------------------------


def test_default_config_stays_in_the_intended_size_band():
    """docs/AI_ARCHITECTURE.md budgets ~10-25M params: big enough to be
    genuinely relational, small enough that GPU time stays well under the
    CPU-bound router's env time."""
    from pcbworld.agents.cfp.model import CFPNet

    total = CFPNet(CFPConfig()).num_parameters()
    assert 5e6 < total < 40e6, f"{total / 1e6:.1f}M params is outside the intended band"


def test_describe_runs(policy):
    assert "CFPPolicy" in policy.describe()
