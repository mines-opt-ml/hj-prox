import torch
import torch.nn as nn

# ------------------------------------------------------------------------------------------------------------
# L1 Norm
# ------------------------------------------------------------------------------------------------------------
def l1_norm(x):
  # assumes shape of x is (n_samples x dim)
  return torch.norm(x, dim=1, p=1)

def l1_norm_noisy(x):
  # assumes shape of x is (n_samples x dim)
  return torch.norm(x, dim=1, p=1) + 1e-1*torch.randn(x.shape[0], device=x.device)

def l1_norm_prox(x, t=0.5):
  # assumes x is (n_samples x dim)
  shrink = nn.Softshrink(lambd=t)
  return shrink(x)

def envelope_l1_norm(x, t=0.5):
  return l1_norm(l1_norm_prox(x, t=t)) + 1/(2*t)*torch.norm(l1_norm_prox(x, t=t) - x, dim=1, p=2)**2

# ------------------------------------------------------------------------------------------------------------
# Quadratic Norm
# ------------------------------------------------------------------------------------------------------------
def quadratic(x, A, b, c=torch.zeros(1)):
  # assume x is (n_samples x dim)
  # A must be SPD (dim x dim)
  # b is a vector of size (dim) NOT (dim x 1)
  # c is a scalar

  c = c.to(x.device)

  assert(len(b.shape)==1) 
  assert(len(c)==1)
  assert(A.shape[0]==A.shape[1] and A.shape[0]==x.shape[1])

  Ax = A.matmul(x.permute(1,0)).permute(1,0) # n_samples x dim
  xAx = torch.sum(x*Ax, dim=1) # n_samples x 1
  bx = x.matmul(b) # n_samples

  return 0.5*xAx + bx + c

def quadratic_noisy(x, A, b, c=torch.zeros(1)):
  # assume x is (n_samples x dim)
  # A must be SPD (dim x dim)
  # b is a vector of size (dim) NOT (dim x 1)
  # c is a scalar

  c = c.to(x.device)

  assert(len(b.shape)==1) 
  assert(len(c)==1)
  assert(A.shape[0]==A.shape[1] and A.shape[0]==x.shape[1])

  Ax = A.matmul(x.permute(1,0)).permute(1,0) # n_samples x dim
  xAx = torch.sum(x*Ax, dim=1) # n_samples x 1
  bx = x.matmul(b) # n_samples

  return 0.5*xAx + bx + c + 5e-2*torch.randn(x.shape[0], device=x.device)

def quadratic_prox(x, A, b,  t=0.5):
  # assume x is (n_samples x dim)
  # A must be SPD (dim x dim)
  # b is a vector of size (dim) NOT (dim x 1)
  # c is a scalar

  assert(len(b.shape)==1) 
  assert(A.shape[0]==A.shape[1] and A.shape[0]==x.shape[1])

  Id = torch.eye(A.shape[0], device=x.device)
  y = torch.linalg.solve(Id + t*A, x.permute(1,0) - t*b.unsqueeze(1))

  return y.permute(1,0) # return y with shape n_samples x dim 

# ------------------------------------------------------------------------------------------------------------
# Logarithmic Barrier
# ------------------------------------------------------------------------------------------------------------
def log_barrier(x):
  # assume x is (n_samples x dim)
  return -torch.sum(torch.log(x), dim=1)

def log_barrier_noisy(x):
  # assume x is (n_samples x dim)
  return -torch.sum(torch.log(x), dim=1) + 1e-1*torch.randn(x.shape[0], device=x.device)

def log_barrier_prox(x, t=0.5):
  # assume x is (n_samples x dim)
  return 0.5*(x + torch.sqrt(x**2 + 4*t))