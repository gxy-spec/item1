from __future__ import annotations

import unittest

import numpy as np

from rl.resources import UniformResourceAllocator


class UniformResourceAllocatorTests(unittest.TestCase):
    def test_only_active_users_receive_resources(self) -> None:
        allocator = UniformResourceAllocator(total_bandwidth=10.0, total_power=2.0)
        result = allocator.allocate(active_user_indices=[0, 2], num_users=4)
        self.assertTrue(np.allclose(result.bandwidth, np.array([5.0, 0.0, 5.0, 0.0])))
        self.assertTrue(np.allclose(result.power, np.array([1.0, 0.0, 1.0, 0.0])))

    def test_active_users_share_resources_equally(self) -> None:
        allocator = UniformResourceAllocator(total_bandwidth=12.0, total_power=3.0)
        result = allocator.allocate(active_user_indices=[1, 2, 3], num_users=5)
        self.assertAlmostEqual(result.bandwidth[1], 4.0)
        self.assertAlmostEqual(result.bandwidth[2], 4.0)
        self.assertAlmostEqual(result.bandwidth[3], 4.0)
        self.assertAlmostEqual(result.power[1], 1.0)
        self.assertAlmostEqual(result.power[2], 1.0)
        self.assertAlmostEqual(result.power[3], 1.0)

    def test_empty_active_set_returns_all_zero(self) -> None:
        allocator = UniformResourceAllocator(total_bandwidth=8.0, total_power=1.0)
        result = allocator.allocate(active_user_indices=[], num_users=3)
        self.assertTrue(np.allclose(result.bandwidth, np.zeros(3)))
        self.assertTrue(np.allclose(result.power, np.zeros(3)))

    def test_bandwidth_sum_never_exceeds_total(self) -> None:
        allocator = UniformResourceAllocator(total_bandwidth=10.0, total_power=2.0)
        result = allocator.allocate(active_user_indices=[0, 1, 2], num_users=3)
        self.assertLessEqual(float(np.sum(result.bandwidth)), 10.0 + 1e-12)

    def test_power_sum_never_exceeds_total(self) -> None:
        allocator = UniformResourceAllocator(total_bandwidth=10.0, total_power=2.0)
        result = allocator.allocate(active_user_indices=[0, 1, 2], num_users=3)
        self.assertLessEqual(float(np.sum(result.power)), 2.0 + 1e-12)


if __name__ == "__main__":
    unittest.main()
