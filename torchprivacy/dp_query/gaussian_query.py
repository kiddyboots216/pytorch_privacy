"""Implements DPQuery interface for Gaussian average queries.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import torch

from torchprivacy.privacy.dp_query import dp_query
from torchprivacy.privacy.dp_query import normalized_query


class GaussianSumQuery(dp_query.SumAggregationDPQuery):
    """Implements DPQuery interface for Gaussian sum queries.

    Accumulates clipped vectors, then adds Gaussian noise to the sum.
    """

    # pylint: disable=invalid-name
    _GlobalState = collections.namedtuple(
            '_GlobalState', ['l2_norm_clip', 'stddev'])

    def __init__(self, l2_norm_clip, stddev):
        """Initializes the GaussianSumQuery.

        Args:
            l2_norm_clip: The clipping norm to apply to the global norm of each
                record.
            stddev: The stddev of the noise added to the sum.
        """
        self._l2_norm_clip = l2_norm_clip
        self._stddev = stddev
        self._ledger = None

    def set_ledger(self, ledger):
        self._ledger = ledger

    def make_global_state(self, l2_norm_clip, stddev):
        """Creates a global state from the given parameters."""
        # TODO: Do I need to cast this to a torch float? Probably not
        return self._GlobalState(l2_norm_clip, stddev)

    def initial_global_state(self):
        return self.make_global_state(self._l2_norm_clip, self._stddev)

    def derive_sample_params(self, global_state):
        return global_state.l2_norm_clip

    def initial_sample_state(self, template):
        return torch.zeros_like(template)

    def preprocess_record_impl(self, params, record):
        """Clips the l2 norm, returning the clipped record and the l2 norm.

        Args:
            params: The parameters for the sample.
            record: The record to be processed.

        Returns:
            A tuple (preprocessed_records, l2_norm) where `preprocessed_records` is
                the structure of preprocessed tensors, and l2_norm is the total l2 norm
                before clipping.
        """
        l2_norm_clip = params
        try:
            l2_norm = torch.norm(record)
        except:
            l2_norm = record.l2estimate()
        if l2_norm < l2_norm_clip:
            return (record, l2_norm)
        else:
            return (record / torch.abs(l2_norm / l2_norm_clip), l2_norm)

    def preprocess_record(self, params, record):
        preprocessed_record, _ = self.preprocess_record_impl(params, record)
        return preprocessed_record

    def get_noised_result(self, sample_state, global_state):
        """See base class."""
        def add_noise(v):
            return v + torch.normal(0, std=global_state.stddev, v.size())
        if self._ledger:
            self._ledger.record_sum_query(global_state.l2_norm_clip, global_state.stddev)
        return add_noise(sample_state), global_state


class GaussianAverageQuery(normalized_query.NormalizedQuery):
    """Implements DPQuery interface for Gaussian average queries.

    Accumulates clipped vectors, adds Gaussian noise, and normalizes.

    Note that we use "fixed-denominator" estimation: the denominator should be
    specified as the expected number of records per sample. Accumulating the
    denominator separately would also be possible but would be produce a higher
    variance estimator.
    """

    def __init__(self,
                l2_norm_clip,
                sum_stddev,
                denominator):
        """Initializes the GaussianAverageQuery.

        Args:
            l2_norm_clip: The clipping norm to apply to the global norm of each
                record.
            sum_stddev: The stddev of the noise added to the sum (before
                normalization).
            denominator: The normalization constant (applied after noise is added to
                the sum).
        """
        super(GaussianAverageQuery, self).__init__(
                numerator_query=GaussianSumQuery(l2_norm_clip, sum_stddev),
                denominator=denominator)
