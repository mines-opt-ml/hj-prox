[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Docs](https://github.com/mines-opt-ml/hj-prox/actions/workflows/ci.yml/badge.svg)

# A Hamilton-Jacobi-based Proximal Operator 

_Stanley Osher, Samy Wu Fung, Howard Heaton_

## Abstract

First-order optimization algorithms are widely used today. Two standard building blocks in these algorithms are proximal operators (proximals) and gradients. Although gradients can be computed for a wide array of functions, explicit proximal formulas are only known for limited classes of functions. We provide an algorithm, HJ-Prox, for accurately approximating such proximals. This is derived from a collection of relations between proximals, Moreau envelopes, Hamilton-Jacobi (HJ) equations, heat equations, and importance sampling. In particular, HJ-Prox smoothly approximates the Moreau envelope and its gradient. The smoothness can be adjusted to act as a denoiser. Our approach applies even when functions are only accessible by (possibly noisy) blackbox samples. We show HJ-Prox is effective numerically via several examples.

## Publication

A Hamilton-Jacobi-based proximal operator ([arXiv Link](https://arxiv.org/abs/2211.12997)).
    
    @article{osher2023hamilton,
             title={{A Hamilton-Jacobi-based proximal operator}},
             author={Osher, Stanley and Heaton, Howard and Fung, Samy Wu},
             journal={{Proceedings of the National Academy of Sciences}},
             year={2023},
             volume={120},
             number={14}
    }

See the [documentation side](https://hj-prox.research.typal.academy) for more details.
