import numpy as np
import torch

def compute_prox(x, t, f, delta=1e-1, int_samples=100, alpha=1.0, linesearch_iters=0, device='cpu'):
  ''' 
      compute prox.
      input is a single vector x of size (dim x 1)
  '''
  assert(x.shape[1]==1)
  assert(x.shape[0]>=1)
  linesearch_iters +=1
  standard_dev = np.sqrt(delta*t/alpha)

  dim = x.shape[0]

  y = standard_dev * torch.randn(int_samples, dim, device=device) + x.permute(1,0) # here y has shape (n_samples x dim)

  z = -f(y)*(alpha/delta) # shape =  n_samples
  w = torch.softmax(z, dim=0) # shape = n_samples 

  softmax_overflow_check = (w < np.inf)
  

  softmax_overflow_check = (w < np.inf)
  if softmax_overflow_check.prod()==0.0:
    print('x = ', x)
    print('z = ', z)
    print('w = ', w)
    alpha = 0.5*alpha
    return compute_prox(x, t, f, delta=delta, int_samples=int_samples, alpha=alpha, linesearch_iters=linesearch_iters, device=device)
  else:
    prox_term = torch.matmul(w.t(), y)
    prox_term = prox_term.view(-1,1)

    prox_overflow = (prox_term < np.inf)
    if prox_overflow.prod() == 0.0:
      print('prox overflowed: ', prox_term)
    assert(prox_overflow.prod() == 1.0)
    envelope = f(prox_term.view(1,-1)) + (1/(2*t)) * torch.norm(prox_term - x.permute(1,0), p=2)**2

    return prox_term, linesearch_iters, envelope