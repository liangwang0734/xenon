## XEON: A matriX-based dispErsion relAtion solver for plasma physics

This is a collection of matrix-based plasma dispersion relation (DR) solvers written in Python. The algorithms are extended from [[1]] and [[2]] with a few changes. Currently, the electromagnetic (EM) kinetic DR solver has not been implemented.

If you prefer the Matlab flavor, or need more complete physics, please refer to the original Matlab implementation, [BO](https://github.com/hsxie/pdrk), by Dr. Huasheng Xie.

[1]:https://www.sciencedirect.com/science/article/pii/S0010465513003408
[2]:https://iopscience.iop.org/article/10.1088/1009-0630/18/2/01/pdf

### Package structure
- `fluid`: Multifluid dispersion relation solver supporting anisotropic pressure. ES1D, ES3D, and EM3D version are implemented.
- `vlasov`: Vlasov dispersion relation solver for a warm plasma. ES1D and ES3D versions are implemented.
- `common`: Coefficient generator, convenience tools for visulization, etc.

### Installation
Currently, the package is provided only as is. One option to use the package is to make Python aware of its path:
```python
import sys
sys.path.append("/path/to/xeon/")
import xeon
```
### References
[[1]] Xie, Hua-sheng. "PDRF: A general dispersion relation solver for magnetized multi-fluid plasma." Computer Physics Communications 185.2 (2014): 670-675.  
[[2]] Xie, Huasheng, and Xiao, Yong. "PDRK: A general kinetic dispersion relation solver for magnetized plasma." Plasma Science and Technology 18.2 (2016): 97.
