"""Implements DPQuery interface for normalized queries.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import torch

from torchprivacy.privacy.dp_query import dp_query


class NormalizedQuery(dp_query.DPQuery):
    """DPQuery for queries with a DPQuery numerator and fixed denominator."""

    # pylint: disable=invalid-name
    _GlobalState = collections.namedtuple(
            '_GlobalState', ['numerator_state', 'denominator'])

    def __init__(self, numerator_query, denominator):
        """Initializer for NormalizedQuery.

        Args:
            numerator_query: A DPQuery for the numerator.
            denominator: A value for the denominator. May be None if it will be
                supplied via the set_denominator function before get_noised_result is
                called.
        """
        self._numerator = numerator_query
        self._denominator = denominator

    def set_ledger(self, ledger):
        """See base class."""
        self._numerator.set_ledger(ledger)

    def initial_global_state(self):
        """See base class."""
        if self._denominator is not None:
            denominator = self._denominator
        else:
            denominator = None
        return self._GlobalState(
                self._numerator.initial_global_state(), denominator)

    def derive_sample_params(self, global_state):
        """See base class."""
        return self._numerator.derive_sample_params(global_state.numerator_state)

    def initial_sample_state(self, template):
        """See base class."""
        # NormalizedQuery has no sample state beyond the numerator state.
        return self._numerator.initial_sample_state(template)

    def preprocess_record(self, params, record):
        return self._numerator.preprocess_record(params, record)

    def accumulate_preprocessed_record(
            self, sample_state, preprocessed_record):
        """See base class."""
        return self._numerator.accumulate_preprocessed_record(
                sample_state, preprocessed_record)

    def get_noised_result(self, sample_state, global_state):
        """See base class."""
        noised_sum, new_sum_global_state = self._numerator.get_noised_result(
                sample_state, global_state.numerator_state)
        def normalize(v):
            return torch.div(v, global_state.denominator)

        return (normalize(noised_sum), self._GlobalState(new_sum_global_state,
                        global_state.denominator))

    def merge_sample_states(self, sample_state_1, sample_state_2):
        """See base class."""
        return self._numerator.merge_sample_states(sample_state_1, sample_state_2)
