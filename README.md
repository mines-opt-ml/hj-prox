[![Documentation Status](https://readthedocs.org/projects/docs-hj-prox/badge/?version=latest)](https://docs-hj-prox.readthedocs.io/en/latest/?badge=latest)

# A Hamilton-Jacobi-based Proximal Operator 

## Abstract

First-order optimization algorithms are widely used today. Two standard building blocks in these algorithms are proximal operators (proximals) and gradients. Although gradients can be computed for a wide array of functions, explicit proximal formulas are only known for limited classes of functions. We provide an algorithm, HJ-Prox, for accurately approximating such proximals. This is derived from a collection of relations between proximals, Moreau envelopes, Hamilton-Jacobi (HJ) equations, heat equations, and importance sampling. In particular, HJ-Prox smoothly approximates the Moreau envelope and its gradient. The smoothness can be adjusted to act as a denoiser. Our approach applies even when functions are only accessible by (possibly noisy) blackbox samples. We show HJ-Prox is effective numerically via several examples.

**See the [Typal Research page](https://research.typal.llc/zeroth-order-methods/hj-prox) for an overview of the algorithm.**

## Publication

A Hamilton-Jacobi-based proximal operator ([arXiv Link](https://arxiv.org/abs/2211.12997)).
    
    @article{osher2022hamilton,
      title={A Hamilton-Jacobi-based Proximal Operator},
      author={Osher, Stanley and Heaton, Howard and Fung, Samy Wu},
      journal={arXiv preprint arXiv:2211.12997},
      year={2022}
    }
