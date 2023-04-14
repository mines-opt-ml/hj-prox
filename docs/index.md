# HJ Prox

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Docs](https://github.com/mines-opt-ml/hj-prox/actions/workflows/ci.yml/badge.svg)

:material-draw-pen: Stanley Osher, Howard Heaton, and Samy Wu Fung

!!! note "Summary"
    We give a formula for estimating proximal operators from (possibly noisy) observations of objective function values.

!!! success "Key Steps"
    - [x] Sample points $\mathsf{y^i}$ (via a Gaussian) about the input $\mathsf{x}$
    - [x] Evaluate function $\mathsf{f}$ at each point $\mathsf{y^i}$
    - [x] Estimate proximal by using softmax to combine the values for $\mathsf{f(y^i)}$ and $\mathsf{y^i}$

[Preprint :fontawesome-solid-file-lines:](assets/hj-prox-preprint.pdf){ .md-button .md-button--primary }
[Reprint :fontawesome-solid-file-lines:](https://www.pnas.org/doi/10.1073/pnas.2220469120){ .md-button .md-button--primary }
[Slides :fontawesome-solid-file-image:](assets/hj-prox-slides.pdf){ .md-button .md-button--primary }

<center>
    <img src="assets/hj-prox-animation.gif" alt="HJ-Prox Animation" width="500"/>
</center>

<div style="padding:56.25% 0 0 0;position:relative;"><iframe src="https://player.vimeo.com/video/778967515?h=57c80f74a7&amp;badge=0&amp;autopause=0&amp;player_id=0&amp;app_id=58479" frameborder="0" allow="autoplay; fullscreen; picture-in-picture" allowfullscreen style="position:absolute;top:0;left:0;width:100%;height:100%;" title="A Hamilton-Jacobi-based Proximal Operator"></iframe></div><script src="https://player.vimeo.com/api/player.js"></script>

!!! abstract "Abstract"

    First-order optimization algorithms are widely used today. Two standard building blocks in these algorithms are proximal operators (proximals) and gradients. Although gradients can be computed for a wide array of functions, explicit proximal formulas are known for only limited classes of functions. We provide an algorithm, HJ-Prox, for accurately approximating such proximals. This is derived from a collection of relations between proximals, Moreau envelopes, Hamilton–Jacobi (HJ) equations, heat equations, and Monte Carlo sampling. In particular, HJ-Prox smoothly approximates the Moreau envelope and its gradient. The smoothness can be adjusted to act as a denoiser. Our approach applies even when functions are accessible only by (possibly noisy) black box samples. We show that HJ-Prox is effective numerically via several examples.

!!! quote "Citation"
    ```
    @article{osher2023hamilton,
             title={{A Hamilton-Jacobi-based proximal operator}},
             author={Osher, Stanley and Heaton, Howard and Fung, Samy Wu},
             journal={{Proceedings of the National Academy of Sciences}},
             year={2023},
             volume={120},
             number={14}
    }
    ```
