"""Utility methods for testing private queries.

Utility methods for testing private queries.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from numpy.testing import assert_, assert_raises
import math

def run_query(query, records, global_state=None, weights=None):
    """Executes query on the given set of records as a single sample.

    Args:
        query: A PrivateQuery to run.
        records: An iterable containing records to pass to the query.
        global_state: The current global state. If None, an initial global state is
            generated.
        weights: An optional iterable containing the weights of the records.

    Returns:
        A tuple (result, new_global_state) where "result" is the result of the
            query and "new_global_state" is the updated global state.
    """
    if not global_state:
        global_state = query.initial_global_state()
    params = query.derive_sample_params(global_state)
    sample_state = query.initial_sample_state(next(iter(records)))
    if weights is None:
        for record in records:
            sample_state = query.accumulate_record(params, sample_state, record)
    else:
        for weight, record in zip(weights, records):
            sample_state = query.accumulate_record(
                    params, sample_state, record, weight)
    return query.get_noised_result(sample_state, global_state)

def assert_allclose(result, expected):
    return np.testing.assert_allclose(result, expected)

def assert_near(f1, f2, err, msg=None):
    assert_(
        f1 == f2 or math.fabs(f1 - f2) <= err, "%f != %f +/- %f%s" %
        (f1, f2, err, " (%s)" % msg if msg is not None else ""))
