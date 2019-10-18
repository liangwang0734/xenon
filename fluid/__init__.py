"""
## Dispersion relation solvers for multifluid plasmas.

Three versions are implemented:

- `k2w_es1d`: Electrostatic waves in an unmagnetized plasma.
- `kw2_es3d`: Electrostatic waves in a magnetized plasma.
- `kw2_em3d`: Electromagnetic waves in a magnetized plasma.

The plasmma pressure is allowed to be anisotropic in the magnetized case (es3d
and em3d).

Further extension planned (in order of priority from high to low):

- Relativistic effects
- Density/temperature gradients
- Collisions
"""
from .fluid import *
