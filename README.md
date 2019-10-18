### XEON: A matriX-based dispErsion relatiON solver for plasma physics

<img src="https://github.com/liangwang0734/xeon/blob/master/docs/bump-on-tail.png" align="right"
     title="Bump-on-tail instability" width="240">
XEON is a collection of matrix-based plasma dispersion relation (DR) solvers written in Python 3. For a quick taste of the flavor, following is a snippet to compute the DR for the Bump-on-tail instability:

```python
species = np.array([
    # parameters of each species
    # q, m, n,   v,     p
    [-1, 1, 0.9, 0,     0.9],  # background electron
    [-1, 1, 0.1, 7.071, 0.1],  # beam electron
])
params = dict(epsilon0=1)  # other parameters

ks = np.linspace(0.0001, 0.6, 50)  # an array of wavenumbers
# for each wavenumber, compute the complex frequencies
ws = xeon.vlasov.k2w_es1d(ks, species, params)

fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharex=True)
plot_dr(ks, ws, ax0=axs[0], ax1=axs[1])
```

##### How does it work?
- The linearized fluid or Vlasov equations are transformed into a matrix form, and complex frequencies `[w1, w2, w3, ...]` are computed as eigenvalues of this coefficient matrix for each wavenumber `k`.
  - The transformation of the Vlasov equations relies on the fact that the Z function of a bi-Maxwellian distribution can be written as a J-pole expansion (Padé approximation).
- In this matrix-based method, no initial guess is necessary, and all solutions, including eigenvectors, are captured.
- The algorithms are extended from [[1]] and [[2]]. If you prefer Matlab over Python, please consider the original implementation, [BO](https://github.com/hsxie/pdrk), by Dr. Huasheng Xie.

##### How do I cite?
- Please cite the code at https://doi.org/10.5281/zenodo.3497597 or  
[![DOI](https://zenodo.org/badge/215848704.svg)](https://zenodo.org/badge/latestdoi/215848704)
- Please submit feature request and code discussion through the [Issues Page](https://github.com/liangwang0734/xeon/issues).
- Any ideas of academic collaboration is welcome at **liang dot wang dot phys AT gmail.com**.

[1]:https://www.sciencedirect.com/science/article/pii/S0010465513003408
[2]:https://iopscience.iop.org/article/10.1088/1009-0630/18/2/01/pdf

#### Dependencies
- Modern [numpy](https://numpy.org/) and [scipy](https://www.scipy.org) are required for computation.
- [matplotlib](https://matplotlib.org/) is required for using the visualization tool.

#### Installation
Currently, the package is in development stage and is provided only as is. One option to use the package is to make Python aware of its path:
```python
import sys
sys.path.append("/path/to/xeon/")
import xeon
```
PyPI and Anaconda distributions will be used starting release 0.2.0.

#### Package structure
- [`fluid`](https://github.com/liangwang0734/xeon/tree/master/fluid): Multifluid dispersion relation solver supporting anisotropic pressure. ES1D, ES3D, and EM3D version are implemented.
- [`vlasov`](https://github.com/liangwang0734/xeon/tree/master/vlasov): Vlasov dispersion relation solver for a warm plasma. ES1D and ES3D versions are implemented.
- [`common`](https://github.com/liangwang0734/xeon/tree/master/common): Coefficient generator, convenience tools for visulization, etc.
- [`notebooks`](https://github.com/liangwang0734/xeon/tree/master/notebooks): Demos compiled as Jupyter notebooks. This folder is being consolidated.

#### Usage
- For the complete API, see https://liangwang0734.github.io/xeon/
- For examples with both computation and visualization, refer to [`notebooks`](https://github.com/liangwang0734/xeon/tree/master/notebooks).

#### References
[[1]] Xie, Hua-sheng. "PDRF: A general dispersion relation solver for magnetized multi-fluid plasma." Computer Physics Communications 185.2 (2014): 670-675.  
[[2]] Xie, Huasheng, and Xiao, Yong. "PDRK: A general kinetic dispersion relation solver for magnetized plasma." Plasma Science and Technology 18.2 (2016): 97.
