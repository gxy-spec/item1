from __future__ import annotations

import numpy as np

from rl.resources import KKTResourceAllocator


def build_allocator() -> KKTResourceAllocator:
    return KKTResourceAllocator(
        total_bandwidth=10.0,
        total_power=1.0,
        noise=1e-3,
    )


def test_kkt_sum_constraints_hold() -> None:
    allocator = build_allocator()
    result = allocator.allocate(
        active_user_indices=[0, 2],
        num_users=4,
        channel_gains=np.array([2.0, 0.5, 1.0, 0.2], dtype=float),
    )
    assert float(np.sum(result.power)) <= 1.0 + 1e-9
    assert float(np.sum(result.bandwidth)) <= 10.0 + 1e-9


def test_kkt_inactive_users_receive_zero() -> None:
    allocator = build_allocator()
    result = allocator.allocate(
        active_user_indices=[1, 3],
        num_users=4,
        channel_gains=np.array([3.0, 2.0, 1.0, 0.5], dtype=float),
    )
    assert result.power[0] == 0.0
    assert result.power[2] == 0.0
    assert result.bandwidth[0] == 0.0
    assert result.bandwidth[2] == 0.0


def test_kkt_higher_gain_gets_more_resource() -> None:
    allocator = build_allocator()
    result = allocator.allocate(
        active_user_indices=[0, 1],
        num_users=2,
        channel_gains=np.array([5.0, 0.5], dtype=float),
        weights=np.array([1.0, 1.0], dtype=float),
    )
    assert result.power[0] > result.power[1]
    assert result.bandwidth[0] > result.bandwidth[1]


def test_kkt_higher_weight_gets_more_resource() -> None:
    allocator = build_allocator()
    result = allocator.allocate(
        active_user_indices=[0, 1],
        num_users=2,
        channel_gains=np.array([1.0, 1.0], dtype=float),
        weights=np.array([2.0, 0.5], dtype=float),
    )
    assert result.power[0] > result.power[1]
    assert result.bandwidth[0] > result.bandwidth[1]


def test_kkt_empty_active_users_returns_zero() -> None:
    allocator = build_allocator()
    result = allocator.allocate(
        active_user_indices=[],
        num_users=3,
        channel_gains=np.array([1.0, 2.0, 3.0], dtype=float),
    )
    assert np.allclose(result.power, 0.0)
    assert np.allclose(result.bandwidth, 0.0)
