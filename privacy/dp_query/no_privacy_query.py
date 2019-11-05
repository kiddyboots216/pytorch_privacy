"""Implements DPQuery interface for no privacy average queries."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

from torchprivacy.privacy.dp_query import dp_query


class NoPrivacySumQuery(dp_query.SumAggregationDPQuery):
    """Implements DPQuery interface for a sum query with no privacy.

    Accumulates vectors without clipping or adding noise.
    """

    def get_noised_result(self, sample_state, global_state):
        """See base class."""
        return sample_state, global_state


class NoPrivacyAverageQuery(dp_query.SumAggregationDPQuery):
    """Implements DPQuery interface for an average query with no privacy.

    Accumulates vectors and normalizes by the total number of accumulated vectors.
    """

    def initial_sample_state(self, template):
        """See base class."""
        return (super(NoPrivacyAverageQuery, self).initial_sample_state(template), (0.0)

    def preprocess_record(self, params, record, weight=1):
        """Multiplies record by weight."""
        return weight * record, weight.float()

    def accumulate_record(self, params, sample_state, record, weight=1):
        """Accumulates record, multiplying by weight."""
        weighted_record = weight * record
        return self.accumulate_preprocessed_record(
                sample_state, (weighted_record, weight.float())) 

    def get_noised_result(self, sample_state, global_state):
        """See base class."""
        sum_state, denominator = sample_state
        return (sum_state / denominator, global_state)
