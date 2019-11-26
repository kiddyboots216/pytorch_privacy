# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Command-line script for computing privacy of a model trained with DP-SGD.

The script applies the RDP accountant to estimate privacy budget of an iterated
Sampled Gaussian Mechanism. The mechanism's parameters are controlled by flags.

Example:
    compute_dp_sgd_privacy
        --N=60000 \
        --batch_size=256 \
        --noise_multiplier=1.12 \
        --epochs=60 \
        --delta=1e-5

The output states that DP-SGD with these parameters satisfies (2.92, 1e-5)-DP.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import sys

from absl import app
from absl import flags
import argparse
import numpy as np

from torchprivacy.analysis.rdp_accountant import compute_rdp, get_privacy_spent

FLAGS = flags.FLAGS



def apply_dp_sgd_analysis(q, sigma, steps, orders, delta):
    """Compute and print results of DP-SGD analysis."""

    # compute_rdp requires that sigma be the ratio of the standard deviation of
    # the Gaussian noise to the l2-sensitivity of the function to which it is
    # added. Hence, sigma here corresponds to the `noise_multiplier` parameter
    # in the DP-SGD implementation found in privacy.optimizers.dp_optimizer
    rdp = compute_rdp(q, sigma, steps, orders)

    eps, _, opt_order = get_privacy_spent(orders, rdp, target_delta=delta)

    print('DP-SGD with sampling rate = {:.3g}% and noise_multiplier = {} iterated'
                ' over {} steps satisfies'.format(100 * q, sigma, steps), end=' ')
    print('differential privacy with eps = {:.3g} and delta = {}.'.format(
            eps, delta))
    print('The optimal RDP order is {}.'.format(opt_order))

    if opt_order == max(orders) or opt_order == min(orders):
        print('The privacy estimate is likely to be improved by expanding '
                    'the set of orders.')

    return eps, opt_order


def compute_dp_sgd_privacy(n, batch_size, noise_multiplier, epochs, delta):
    """Compute epsilon based on the given hyperparameters."""
    q = batch_size / n  # q - the sampling ratio.
    if q > 1:
        raise ValueError('n must be larger than the batch size.')
    orders = ([1.25, 1.5, 1.75, 2., 2.25, 2.5, 3., 3.5, 4., 4.5] +
                        list(range(5, 64)) + [128, 256, 512])
    steps = int(math.ceil(epochs * n / batch_size))

    return apply_dp_sgd_analysis(q, noise_multiplier, steps, orders, delta)

def compute_noise_multiplier(args):
    """Compute noise multiplier based on given params"""
    #return (8 * args.l2_norm_clip ** 2 * np.log(1.25 / args.delta)) / (np.log( (np.exp(-args.epsilon / np.sqrt(args.num_epochs / args.participation)) - 1) / (args.participation) + 1) ** 2)
    const = 8
    num = const * args.l2_norm_clip ** 2 * np.log(1.25 / args.delta)
    n_iters = args.num_epochs / args.participation
    eps_num = np.exp(args.epsilon / np.sqrt(n_iters)) - 1
    eps_denom = args.participation
    inner = (eps_num / eps_denom) + 1
    eps_bar = np.log(inner)
    denom = eps_bar ** 2 * args.num_worker ** 2
    sigma_squared = num / denom
    return np.sqrt(sigma_squared)

def main(args):
    if args.noise_multiplier is None:
        args.noise_multiplier = compute_noise_multiplier(args)
    print(f"Adding noise: {args.noise_multiplier}")
    compute_dp_sgd_privacy(args.num_clients, args.num_workers, args.noise_multiplier,
                                                  args.num_epochs, args.delta)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clients", type=int, help='Total number of samples')
    parser.add_argument("--num_workers", type=int, help='Batch size or number of workers')
    parser.add_argument('--noise_multiplier', type=float, help='Noise multiplier for DP-SGD')
    parser.add_argument('--num_epochs', type=float, help='Number of epochs (may be fractional)')
    parser.add_argument('--delta', type=float, help='Target delta')
    parser.add_argument("--epsilon", type=float, default=2.0, help='target epsilon')
    parser.add_argument("--l2_norm_clip", type=float, default=1.0, help='l2 norm to clip to')
    args = parser.parse_args()
    args.participation = args.num_workers / args.num_clients
    main(args)
