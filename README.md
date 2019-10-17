## MIRROR: A Matrix-based dIspeRsion Relation sOlveR

This is a collection of matrix-based plasma dispersion relation (DR) solvers written in Python. The algorithms are extended from [[1]] and [[2]] with a few changes. Currently, the electromagnetic (EM) kinetic DR solver has not been implemented.

If you prefer the Matlab flavor, or need more complete physics, please refer to the original Matlab implementation, [BO](https://github.com/hsxie/pdrk), by Dr. Huasheng Xie.

[1]:https://www.sciencedirect.com/science/article/pii/S0010465513003408
[2]:https://iopscience.iop.org/article/10.1088/1009-0630/18/2/01/pdf

### Package structure
- `fluid`: multifluid-Maxwell dispersion relation solver with anisotropic pressure.
- `vlasov_es1d`: Electrostatic 1d Vlasov dispersion relation solver for a warm, unmagnetized plasma.
- `vlasov_es3d`: Electrostatic 3d Vlasov dispersion relation solver for a warm, magnetized plasma.
- `common`: Coefficient generator, convenience tools for visulization, etc.
- `notebooks`: Examples compiled as Jupyter notebooks.

### Installation
Currently, the package is provided only as is. One option to use the package is to make Python aware of its path:
```python
import sys
sys.path.append("/path/to/drPlasma/")
import drPlasma
```

### Usage
Please refer to the notebooks for dispersion relation computation and visualization examples.

### References
[[1]] Xie, Hua-sheng. "PDRF: A general dispersion relation solver for magnetized multi-fluid plasma." Computer Physics Communications 185.2 (2014): 670-675.  
[[2]] Xie, Huasheng, and Xiao, Yong. "PDRK: A general kinetic dispersion relation solver for magnetized plasma." Plasma Science and Technology 18.2 (2016): 97.
