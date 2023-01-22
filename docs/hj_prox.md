# HJ-Prox Algorithm

To execute HJ-Prox, you can run:

::: src.hj_prox.compute_hj_prox
    options:
      show_root_heading: true

<br>

Input is a single vector "x" of size (dim, 1)    

Proximal computation does *not* handle case for f with negative
values. That case must be handled by adding an offset and is
omitted her since all our functions of interest are nonnegative.

The computation for the proximal involves the exponential of a
potentially large negative number, which can result in underflow
in floating point arithmetic that renders a grossly inaccurate 
proximal calculation. To avoid this, the "large negative number" 
is reduced in size by using a smaller value of alpha, returning 
a result once the underflow is not considered significant
(as defined by the tolerances "tol" and "tol_underflow").
            
This is the end.
