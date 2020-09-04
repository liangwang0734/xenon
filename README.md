### XENON: A matriX-based dispErsioN relatiON solver for plasma physics
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3497597.svg)](https://doi.org/10.5281/zenodo.3497597)

XENON is a collection of matrix-based plasma dispersion relation (DR) solvers written in Python 3. Currently, electrostatic/electromagnetic multifluid plasmas, and electrostatic kinetic plasmas are supported. The algorithms are extended from [[1]] and [[2]] pioneered by Dr. Huasheng Xie. 

The motivation of this work is to facilitate multifluid and Vlasov simulations using the [Gkeyll v2 code](https://github.com/ammarhakim/gkyl).

<img src="demos/images/bump-on-tail.png" align="right"
     title="Bump-on-tail instability" width="300">
For a quick taste, following is a snippet to compute the DR for the Bump-on-tail instability:
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
ws = xenon.vlasov.k2w_es1d(ks, species, params)

fig, axs = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
xenon.common.plot_dr(ks, ws, ax0=axs[0], ax1=axs[1])
```
For more examples, refer to [`demos`](demos).  
For the complete API, see https://liangwang0734.github.io/xenon/

#### How does xenon work?
- The linearized fluid or Vlasov equations are transformed into a matrix form, and complex frequencies `[w1, w2, w3, ...]` are computed as eigenvalues of this coefficient matrix for each wavenumber `k`.
- In this matrix-based method, no initial guess is necessary, and all solutions, including eigenvectors, are captured.
- If you prefer Matlab over Python, please consider the original implementation, [BO](https://github.com/hsxie/pdrk), by Dr. Huasheng Xie.

[1]:https://www.sciencedirect.com/science/article/pii/S0010465513003408
[2]:https://iopscience.iop.org/article/10.1088/1009-0630/18/2/01/pdf

#### Does xenon rely on other softwares to work?  
- Required: Modern [numpy](https://numpy.org/) and [scipy](https://www.scipy.org) are required for computation.
- Optional: [matplotlib](https://matplotlib.org/) is required for using the builtin visualization tools.

#### How do I load xenon?
Currently, the package is in development stage and is provided only as is. One option to use the package is to make Python aware of its path:
```python
import sys
sys.path.append("/path/to/parent/folder/of/xenon/")
import xenon
```
PyPI and Anaconda distributions will be used starting release 0.2.0.

#### What are the folders/files for?
- [`fluid`](fluid): Multifluid dispersion relation solver supporting anisotropic pressure. `ES1D`, `ES3D`, and `EM3D` version are implemented.
- [`vlasov`](vlasov): Vlasov dispersion relation solver for a warm plasma. `ES1D` and `ES3D` versions are implemented.
- [`common`](common): Coefficient generator, convenience tools for visulization, etc.
- [`demos`](demos): Demos compiled as markdown documents. This folder is being consolidated.

#### How do I use xenon like an expert?
- For the complete API, see https://liangwang0734.github.io/xenon/
- For examples with both computation and visualization, refer to [`demos`](demos), which will be consolidated in the future.

#### References
[[1]] Xie, Hua-sheng. "PDRF: A general dispersion relation solver for magnetized multi-fluid plasma." Computer Physics Communications 185.2 (2014): 670-675.  
[[2]] Xie, Huasheng, and Xiao, Yong. "PDRK: A general kinetic dispersion relation solver for magnetized plasma." Plasma Science and Technology 18.2 (2016): 97.

#### Funding support
- Air Force Office of Scientific Research Grant No. FA9550-15-1-0193.
