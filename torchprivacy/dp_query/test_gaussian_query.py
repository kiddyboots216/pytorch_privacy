"""Tests for GaussianSumQuery."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest, parameterized
import numpy as np
from numpy.testing import assert_allclose

from six.moves import xrange
import torch

from torchprivacy.dp_query import gaussian_query
from torchprivacy.dp_query.test_utils import run_query, assert_near

class GaussianQueryTest(parameterized.TestCase):

    def test_gaussian_sum_no_clip_no_noise(self):
        record1 = torch.tensor([2.0, 0.0])
        record2 = torch.tensor([-1.0, 1.0])
        query = gaussian_query.GaussianSumQuery(
		l2_norm_clip=10.0, stddev=0.0)
        query_result, _ = run_query(query, [record1, record2])
        expected = np.asarray([1.0, 1.0])
        assert_allclose(query_result, expected)

    def test_gaussian_sum_with_clip_no_noise(self):
        record1 = torch.tensor([-6.0, 8.0])  # Clipped to [-3.0, 4.0].
        record2 = torch.tensor([4.0, -3.0])  # Not clipped.

        query = gaussian_query.GaussianSumQuery(
		l2_norm_clip=5.0, stddev=0.0)
        query_result, _ = run_query(query, [record1, record2])
        expected = np.asarray([1.0, 1.0])
        assert_allclose(query_result, expected)

    def test_gaussian_sum_with_changing_clip_no_noise(self):
        record1 = torch.tensor([-6.0, 8.0])  # Clipped to [-3.0, 4.0].
        record2 = torch.tensor([4.0, -3.0])  # Not clipped.

        l2_norm_clip = 5.0
        query = gaussian_query.GaussianSumQuery(
                l2_norm_clip=l2_norm_clip, stddev=0.0)
        query_result, _ = run_query(query, [record1, record2])

        expected = np.asarray([1.0, 1.0])
        assert_allclose(query_result, expected)

        expected = np.asarray([0.0, 0.0])
        assert_allclose(query_result, expected)

    def test_gaussian_sum_with_noise(self):
        record1, record2 = 2.71828, 3.14159
        stddev = 1.0

        query = gaussian_query.GaussianSumQuery(
                l2_norm_clip=5.0, stddev=stddev)
        query_result, _ = run_query(query, torch.tensor([record1, record2]))

        noised_sums = []
        for _ in xrange(1000):
            noised_sums.append(query_result)

        result_stddev = np.std(noised_sums)
        assert_near(result_stddev, stddev, 0.1)

    def test_gaussian_sum_merge(self):
        records1 = [torch.tensor([2.0, 0.0]), torch.tensor([-1.0, 1.0])]
        records2 = [torch.tensor([3.0, 5.0]), torch.tensor([-1.0, 4.0])]

        def get_sample_state(records):
            query = gaussian_query.GaussianSumQuery(l2_norm_clip=10.0, stddev=1.0)
            global_state = query.initial_global_state()
            params = query.derive_sample_params(global_state)
            sample_state = query.initial_sample_state(records[0])
            for record in records:
                sample_state = query.accumulate_record(params, sample_state, record)
            return sample_state

        sample_state_1 = get_sample_state(records1)
        sample_state_2 = get_sample_state(records2)

        merged = gaussian_query.GaussianSumQuery(10.0, 1.0).merge_sample_states(
                sample_state_1,
                sample_state_2)

        expected = np.asarray([3.0, 10.0])
        assert_allclose(merged, expected)

    def test_gaussian_average_no_noise(self):
        record1 = torch.tensor([5.0, 0.0])   # Clipped to [3.0, 0.0].
        record2 = torch.tensor([-1.0, 2.0])  # Not clipped.

        query = gaussian_query.GaussianAverageQuery(
                l2_norm_clip=3.0, sum_stddev=0.0, denominator=2.0)
        query_result, _ = run_query(query, [record1, record2])
        expected_average = np.asarray([1.0, 1.0])
        assert_allclose(query_result, expected_average)

    def test_gaussian_average_with_noise(self):
        record1, record2 = 2.71828, 3.14159
        sum_stddev = 1.0
        denominator = 2.0

        query = gaussian_query.GaussianAverageQuery(
                l2_norm_clip=5.0, sum_stddev=sum_stddev, denominator=denominator)
        query_result, _ = run_query(query, torch.tensor([record1, record2]))

        noised_averages = []
        for _ in range(1000):
            noised_averages.append(query_result)

        result_stddev = np.std(noised_averages)
        avg_stddev = sum_stddev / denominator
        assert_near(result_stddev, avg_stddev, 0.1)

    @parameterized.named_parameters(
            ('type_mismatch', [1.0], (1.0,), TypeError),
            ('too_few_on_left', [1.0], [1.0, 1.0], ValueError),
            ('too_few_on_right', [1.0, 1.0], [1.0], ValueError))

    def test_incompatible_records(self, record1, record2, error_type):
        query = gaussian_query.GaussianSumQuery(1.0, 0.0)
        with self.assertRaises(error_type):
            run_query(query, torch.tensor([record1, record2]))

if __name__ == '__main__':
    absltest.main()
