## XEON: A matriX-based dispErsion relAtion solver for plasma physics

XEON is a collection of matrix-based plasma dispersion relation (DR) solvers written in Python. The algorithms are extended from [[1]] and [[2]]. The linearized fluid or Vlasov equations are transformed into a matrix form, and complex frequencies `[w1, w2, w3, ...]` are computed as eigenvalues of this matrix for each wavenumber `k`. Currently, the electromagnetic (EM) kinetic DR solver has not been implemented.

If you prefer the Matlab flavor, please consider the original implementation, [BO](https://github.com/hsxie/pdrk), by Dr. Huasheng Xie.

[1]:https://www.sciencedirect.com/science/article/pii/S0010465513003408
[2]:https://iopscience.iop.org/article/10.1088/1009-0630/18/2/01/pdf

### Dependencies
- Modern [numpy](https://numpy.org/) and [scipy](https://www.scipy.org) are required for computation.
- [matplotlib](https://matplotlib.org/) is required for using the visualization tool.

### Installation
Currently, the package is in development stage and is provided only as is. One option to use the package is to make Python aware of its path:
```python
import sys
sys.path.append("/path/to/xeon/")
import xeon
```

### Package structure and usage
- `fluid`: Multifluid dispersion relation solver supporting anisotropic pressure. ES1D, ES3D, and EM3D version are implemented.
- `vlasov`: Vlasov dispersion relation solver for a warm plasma. ES1D and ES3D versions are implemented.
- `common`: Coefficient generator, convenience tools for visulization, etc.
- `notebooks`: Demos compiled as Jupyter notebooks. This folder is being consolidated.

For the API, see https://liangwang0734.github.io/xeon/

### References
[[1]] Xie, Hua-sheng. "PDRF: A general dispersion relation solver for magnetized multi-fluid plasma." Computer Physics Communications 185.2 (2014): 670-675.  
[[2]] Xie, Huasheng, and Xiao, Yong. "PDRK: A general kinetic dispersion relation solver for magnetized plasma." Plasma Science and Technology 18.2 (2016): 97.
