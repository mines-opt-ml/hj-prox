import torch
import numpy as np

def compute_hj_prox(x, t, f, delta=1e-1, int_samples=1000, alpha=2.0,
                    recursion_depth=0, alpha_decay=0.631, tol=1.0e-6, 
                    tol_underflow=0.9, device='cpu', verbose=False,
                    return_samples=False):
    """ Estimate proximals from function value sampling via HJ-Prox Algorithm.

        The output computes
        
        $\textsf{prox}_{tf}(x) = \textsf{argmin}_y f(y) + \frac{1}{2t} \| y - x \|^2,$
        
        where $\mathsf{x}$ = `x` is the input, $\mathsf{t}$=`t` is the time parameter, 
        and $\mathsf{f}$=`f` is the function of interest.
        
        - [x] Sample points $\mathsf{y^i}$ (via a Gaussian) about the input $\mathsf{x}$
        - [x] Evaluate function $\mathsf{f}$ at each point $\mathsf{y^i}$
        - [x] Estimate proximal by using softmax to combine the values for $\mathsf{f(y^i)}$ and $\mathsf{y^i}$            

        Numerical Consideration: 
            The computation for the proximal involves the exponential of a potentially
            large negative number, which can result in underflow in floating point
            arithmetic that renders a grossly inaccurate proximal calculation. To avoid
            this, the "large negative number" is reduced in size by using a smaller
            value of alpha, returning a result once the underflow is not considered
            significant (as defined by the tolerances "tol" and "tol_underflow").
            Utilizing a scaling trick with proximals, this is mitigated by using
            recursive function calls.

        Args:
            x (tensor): Input vector
            t (tensor): Time > 0
            f: Function to minimize

        Returns:
            tensor: Estimate of the proximal of f at x
    """
    valid_vector_shape = x.shape[1] == 1 and x.shape[0] >= 1
    assert valid_vector_shape

    recursion_depth +=1
    std_dev = np.sqrt(delta * t / alpha)
    dim     = x.shape[0]
    y       = std_dev * torch.randn(int_samples, dim, device=device) 
    y       = y + x.permute(1,0)
    z       = -f(y) * (alpha / delta)

    underflow         = torch.exp(z)  <= tol
    underflow_freq    = float(underflow.sum()) / underflow.shape[0]
    observe_underflow = underflow_freq > tol_underflow

    if observe_underflow: 
        alpha *= alpha_decay
        return compute_hj_prox(x, t, f, delta=delta, int_samples=int_samples,
                               alpha=alpha, recursion_depth=recursion_depth,
                               alpha_decay=alpha_decay, tol=tol, 
                               tol_underflow=tol_underflow, device=device,
                               verbose=verbose, return_samples=return_samples)         
    else:                
        soft_max = torch.nn.Softmax(dim=1)  
        HJ_prox  = soft_max(z.permute(1,0)).mm(y)   

        valid_prox_shape = HJ_prox.shape == x.shape
        assert valid_prox_shape

        prox_is_finite = (HJ_prox < np.inf).all()
        assert prox_is_finite 

        if verbose:
            envelope = - (delta / alpha) * torch.log(torch.mean(torch.exp(z)))
            return HJ_prox, recursion_depth, envelope
        elif return_samples:
            return HJ_prox, y, alpha
        else:
            return HJ_prox
