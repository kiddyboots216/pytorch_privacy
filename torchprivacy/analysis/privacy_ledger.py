"""PrivacyLedger class for keeping a record of private queries."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np
import torch

from torchprivacy.dp_query import dp_query
from torchprivacy.analysis.tensor_buffer import TensorBuffer

SampleEntry = collections.namedtuple(  # pylint: disable=invalid-name
        'SampleEntry', ['population_size', 'selection_probability', 'queries'])

GaussianSumQueryEntry = collections.namedtuple(  # pylint: disable=invalid-name
        'GaussianSumQueryEntry', ['l2_norm_bound', 'noise_stddev'])


def format_ledger(sample_array, query_array):
    """Converts array representation into a list of SampleEntries."""
    samples = []
    query_pos = 0
    sample_pos = 0
    for sample in sample_array:
        population_size, selection_probability, num_queries = sample
        queries = []
        for _ in range(int(num_queries)):
            query = query_array[query_pos]
            assert int(query[0]) == sample_pos
            queries.append(GaussianSumQueryEntry(*query[1:]))
            query_pos += 1
        samples.append(SampleEntry(population_size, selection_probability, queries))
        sample_pos += 1
    return samples


class PrivacyLedger(object):
    """Class for keeping a record of private queries.

    The PrivacyLedger keeps a record of all queries executed over a given dataset
    for the purpose of computing privacy guarantees.
    """

    def __init__(self,
                             population_size,
                             selection_probability):
        """Initialize the PrivacyLedger.

        Args:
            population_size: An integer (may be variable) specifying the size of the
                population, i.e. size of the training data used in each epoch.
            selection_probability: A float (may be variable) specifying the
                probability each record is included in a sample.

        Raises:
            ValueError: If selection_probability is 0.
        """
        self._population_size = population_size
        self._selection_probability = selection_probability

        if selection_probability <= 0:
            raise ValueError('Selection probability must be greater than 0.')
        init_capacity = np.int(np.ceil(1 / selection_probability))

        # The query buffer stores rows corresponding to GaussianSumQueryEntries.
        self._query_buffer = TensorBuffer(init_capacity, [3])
        self._query_count = 0

        # The sample buffer stores rows corresponding to SampleEntries.
        self._sample_buffer = TensorBuffer(init_capacity, [3])
        self._sample_count = 0

    def record_sum_query(self, l2_norm_bound, noise_stddev):
        """Records that a query was issued.

        Args:
            l2_norm_bound: The maximum l2 norm of the tensor group in the query.
            noise_stddev: The standard deviation of the noise applied to the sum.

        Returns:
            An operation recording the sum query to the ledger.
        """
        self._query_count = self._query_count + 1
        query_var = torch.Tensor([self._sample_count, l2_norm_bound, noise_stddev])
        self._query_buffer.append(query_var)

    def finalize_sample(self):
        """Finalizes sample and records sample ledger entry."""
        sample_var = torch.Tensor([self._population_size, self._selection_probability, self._query_count])
        self._sample_buffer.append(sample_var)
        self._sample_count = self._sample_count + 1
        self._query_count = 0

    def get_unformatted_ledger(self):
        return self._sample_buffer.values, self._query_buffer.values

    def get_formatted_ledger(self):
        """Gets the formatted query ledger.

        Returns:
            The query ledger as a list of SampleEntries.
        """
        sample_array = self._sample_buffer.values.numpy()
        query_array = self._query_buffer.values.numpy()

        return format_ledger(sample_array, query_array)


class QueryWithLedger(dp_query.DPQuery):
    """A class for DP queries that record events to a PrivacyLedger.

    QueryWithLedger should be the top-level query in a structure of queries that
    may include sum queries, nested queries, etc. It should simply wrap another
    query and contain a reference to the ledger. Any contained queries (including
    those contained in the leaves of a nested query) should also contain a
    reference to the same ledger object.

    For example usage, see privacy_ledger_test.py.
    """

    def __init__(self, query,
                             population_size=None, selection_probability=None,
                             ledger=None):
        """Initializes the QueryWithLedger.

        Args:
            query: The query whose events should be recorded to the ledger. Any
                subqueries (including those in the leaves of a nested query) should also
                contain a reference to the same ledger given here.
            population_size: An integer (may be variable) specifying the size of the
                population, i.e. size of the training data used in each epoch. May be
                None if `ledger` is specified.
            selection_probability: A float (may be variable) specifying the
                probability each record is included in a sample. May be None if `ledger`
                is specified.
            ledger: A PrivacyLedger to use. Must be specified if either of
                `population_size` or `selection_probability` is None.
        """
        self._query = query
        if population_size is not None and selection_probability is not None:
            self.set_ledger(PrivacyLedger(population_size, selection_probability))
        elif ledger is not None:
            self.set_ledger(ledger)
        else:
            raise ValueError('One of (population_size, selection_probability) or '
                                             'ledger must be specified.')

    @property
    def ledger(self):
        return self._ledger

    def set_ledger(self, ledger):
        self._ledger = ledger
        self._query.set_ledger(ledger)

    def initial_global_state(self):
        """See base class."""
        return self._query.initial_global_state()

    def derive_sample_params(self, global_state):
        """See base class."""
        return self._query.derive_sample_params(global_state)

    def initial_sample_state(self, template):
        """See base class."""
        return self._query.initial_sample_state(template)

    def preprocess_record(self, params, record):
        """See base class."""
        return self._query.preprocess_record(params, record)

    def accumulate_preprocessed_record(self, sample_state, preprocessed_record):
        """See base class."""
        return self._query.accumulate_preprocessed_record(
                sample_state, preprocessed_record)

    def merge_sample_states(self, sample_state_1, sample_state_2):
        """See base class."""
        return self._query.merge_sample_states(sample_state_1, sample_state_2)

    def get_noised_result(self, sample_state, global_state):
        """Ensures sample is recorded to the ledger and returns noised result."""
        # Ensure sample_state is fully aggregated before calling get_noised_result.
        result, new_global_state = self._query.get_noised_result(
                    sample_state, global_state)
        # Ensure inner queries have recorded before finalizing.
        self._ledger.finalize_sample()
        op = lambda tensor: tensor.clone().detach()
        return op(result), new_global_state

    def get_record_derived_data(self):
        return self._query.get_record_derived_data()

class QueryWithPerClientLedger(dp_query.DPQuery):
    """ A class for DPQueries that stores the queries in a privacy ledger.

    Simple wrapper for a DQQuery-PrivacyLedger pair to ensure correct running.
    """

    def __init__(self,
                 query,
                 num_clients,
                 selection_probability):

        # note that this query ledger is not actually relevant here!!
        self.M = num_clients
        self._query = query
        self.set_ledgers([PrivacyLedger(num_clients, selection_probability) for _ in range(num_clients)])

    @property
    def query(self):
        return self._query

    @property
    def ledgers(self):
        return self._ledgers

    def set_ledgers(self, ledgers):
        self._ledgers = ledgers

    def initial_global_state(self):
        """See base class."""
        return self._query.initial_global_state()

    def derive_sample_params(self, global_state):
        """See base class."""
        return self._query.derive_sample_params(global_state)

    def initial_sample_state(self, template):
        """See base class."""
        return self._query.initial_sample_state(template)

    def preprocess_record(self, params, record):
        """See base class."""
        return self._query.preprocess_record(params, record)

    def accumulate_preprocessed_record(self, sample_state, preprocessed_record):
        """See base class."""
        return self._query.accumulate_preprocessed_record(
            sample_state, preprocessed_record)

    def merge_sample_states(self, sample_state_1, sample_state_2):
        """See base class."""
        return self._query.merge_sample_states(sample_state_1, sample_state_2)

    def get_noised_result(self, sample_state, global_state, selected_indices):
        """Ensures sample is recorded to the ledger and returns noised result."""
        result, new_global_state = self._query.get_noised_result(sample_state, global_state)
        # record sum queries for each client who was selected
        for i in range(self.M):
            if i in selected_indices:
                self.ledgers[i].record_sum_query(global_state.l2_norm_clip, global_state.noise_stddev)
                self.ledgers[i].finalise_sample()

        if isinstance(result, torch.Tensor):
            op = lambda tensor: tensor.clone().detach()
            return op(result), new_global_state
        else:
            return result, new_global_state

    def get_record_derived_data(self):
        return self._query.get_record_derived_data()

    def get_formatted_ledgers(self):
        """ Returns a formatted version of the ledger for use in privacy accounting """
        return [x.get_formatted_ledger() for x in self.ledgers]
