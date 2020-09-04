r"""
## Dispersion relation solvers for kinetic plasmas.

Two versions are implemented:

- `k2w_es1d`: Electrostatic waves in an unmagnetized plasma.
- `kw2_es3d`: Electrostatic waves in a magnetized plasma.
"""
from .vlasov_utils import *
from .vlasov_dr_es import *
