__all__ = ['k2w_es1d', 'k2w_es3d']

import numpy as np
import scipy.linalg
from scipy.special import ive
from .common_vlasov import bzs, czs
import logging


def k2w_es1d(ks, species, params, J=8, sort='real'):
    """Compute dispersion relation for the unmagnetized 1d Vlasov-Poisson
    system.
  
    The basic algorithm follows [1]. The Z function is approximated by a J-pole
    Pade expansion and the bi-Maxwellian distribution is approximated by a
    truncated summation of Bessel functions. This way, a linear system is
    generated that can be solved for complex frequencies as eigenvalues of the
    equivalent matrix.

    [1]:https://iopscience.iop.org/article/10.1088/1009-0630/18/2/01/pdf
  
    Args:
        ks (np.ndarray): An array of `k` values.
        species (np.ndarray of list): A list of parameters of each plasma
            species. `species[s]` is the parameters of the sth species,
            including `q, m, n0, v0, p0`. An example for a
            plasma with isothermal electrons and adiabatic:

                species = np.array([  
                    [-1,  1, 1, 0, 0.25],  # electron  
                    [+1, 25, 1, 0, 0.81],  # ion  
                ])

        params (dict): A dictionary of relevant parameters. Here, the used ones
            are `epsilon0`.
        J (int): Order of Pade polynomials to be used. Supported values are `8`
            and `12`.
        sort (str): `'real'` or `'imag'` or `'none'`. Order to sort results.

    Returns:
        ws (np.ndarray): `ws[ik, :]` is the maginary frequency for the `ik`'th
            `kx` value.
    """
    q, m, n, v, p = np.rollaxis(species, axis=1)
    eps0 = params['epsilon0']
    vt = np.sqrt(2. * p / n / m)
    lambdaD = np.sqrt(eps0 * p / (n * q)**2)
    S = len(q)

    bz, cz = bzs[J], czs[J]
    B = np.zeros([S * J], dtype=np.complex128)
    C = np.zeros([S * J], dtype=np.complex128)
    m = 0
    for s in range(S):
        for j in range(J):
            B[m] = bz[j] * cz[j] * vt[s] / (lambdaD[s]**2)
            C[m] = v[s] + cz[j] * vt[s]
            m += 1

    SJ = S * J
    ws = np.empty((len(ks), SJ), dtype=np.complex128)
    M = np.empty((SJ, SJ), dtype=np.complex128)
    for ik, k in enumerate(ks):
        for m in range(SJ):
            M[m, :] = -B[m] / k
            M[m, m] += C[m] * k
        w = scipy.linalg.eigvals(M)

        sort_idx = slice(None)
        if sort in ['imag']:
            sort_idx = np.argsort(w.imag + 1j * w.real)
        elif sort in ['real']:
            sort_idx = np.argsort(w)
        elif sort != 'none':
            raise ValueError('`sort` value {} not recognized'.format(sort))
        w = w[sort_idx]

        ws[ik, :] = w

    return np.array(ws)


def k2w_es3d(
        kxs,
        kzs,
        species,
        params,
        isMag=None,
        J=8,
        N=10,
        check_convergence=True,
        convergence_thresh=0.975,
        sort="real",
        dry_run=False,
):
    """
    Compute electrostatic dispersion relation for a magnetized Vlasov-Poisson
    system assuming the background magnetic field along `z` and the wavevector
    `k` can have components along `x` and `z`, i.e., the perpendicular and
    parallel directions.
  
    The basic algorithm follows [1]. The Z function is approximated by a J-pole
    Pade expansion and the bi-Maxwellian distribution is approximated by a
    truncated summation of Bessel functions. This way, a linear system is
    generated that can be solved for complex frequencies as eigenvalues of the
    equivalent matrix.

    [1]:https://iopscience.iop.org/article/10.1088/1009-0630/18/2/01/pdf
  
    Args:
        kxs (np.ndarray): Wavevector component values along `x`.
        kzs (np.ndarray): Wavevector component values along `z`.
        species (np.ndarray): A list of parameters of each plasma species.
          `species[s]` are the parameters of the `s`th species, including
          `q, m, n, v0x, v0z, p0x, p0z`.
        params (dict): A dictionary of relevant parameters. Here, the used
          ones are `epsilon0` and `Bz`.
        isMag (list or None): A list of booleans to indicate wether each species
            is magnetized. If not set, all species are assumed to be magnetized.
        J (int): Order of Pade approximation. One of (`8`, `12`).
        N (int): Highest order of cylctron harmonic to include in the Bessel
          expansion of the bi-Maxwellian distribution. If `check_convergence`
          is set to True, `N` will be increased until the expansion coefficients
          add up to greater than a threshold value set by `convergence_thresh`.
        check_convergence (bool): See option `N`. Default is True.
        convergence_thresh (float): See option `N`. Default is 0.975.
        sort (str): Order for sorting computed frequencies. Default is "real".
        sort (str): `'real'` or `'imag'` or `'none'`. Order to sort results.
  
    Returns:
        ws (np.ndarray or list): Computed frequencies in a list. `ws[ik]` is the
          solution for the `ik`th wavevector. The result is packed as a
          `np.ndarray` if the `N` values used for all `k` are the same.
    """
    q, m, n, vx, vz, px, pz = np.rollaxis(np.array(species), axis=1)
    eps0 = params['epsilon0']
    Bz = params['Bz']

    S = len(q)  # number of species
    lamDz = np.sqrt(eps0 * pz / (n * q)**2)
    vtz = np.sqrt(2. * pz / n / m)
    vtx = np.sqrt(2. * px / n / m)
    wc = q * Bz / m

    if isMag is None:
        isMag = [True] * S
    assert (len(isMag) == S)

    bz, cz = bzs[J], czs[J]

    Tz_Tx = (vtz / vtx)**2
    rc = vtx / np.sqrt(2) / np.abs(wc)

    ks = np.sqrt(kzs**2 + kxs**2)
    nk = len(ks)
    ws = []
    Ns = np.zeros((nk, ), dtype=np.int32)

    if check_convergence:
        if np.any(isMag):
            for ik in range(nk):
                Nmax = N
                kz = kzs[ik]
                kp = kxs[ik]
                kp_rc2m = ((kp * rc[isMag])**2).max()
                found = False
                while (not found):
                    bessSum = ive(range(-Nmax, Nmax + 1), kp_rc2m).sum()
                    if bessSum < convergence_thresh:
                        Nmax = Nmax + 1
                    else:
                        found = True
                        if ik == len(ks) - 1:
                            logging.debug(
                                "(kp*rc)^2 {}, N {}, bessSum {}".format(
                                    kp_rc2m, Nmax, bessSum))
                Ns[ik] = Nmax
    logging.debug("Ns {}".format(Ns))
    print('Ns', Ns)
    if dry_run:
        return

    for ik, k in enumerate(ks):
        kz = kzs[ik]
        kp = kxs[ik]

        N4k = Ns[ik]  # the number of terms to keep for k
        SNJ = S * (2 * N4k + 1) * J
        M = np.zeros((SNJ, SNJ), dtype=np.complex128)
        B = np.zeros([SNJ], dtype=np.complex128)
        C = np.zeros([SNJ], dtype=np.complex128)
        m = 0
        for s in range(S):
            if isMag[s]:
                for n in range(-N4k, N4k + 1):
                    bessel = ive(n, (kp * rc[s])**2)  # modified Bessel function
                    for j in range(J):
                        B[m] = bessel * bz[j] \
                            * (cz[j] * kz * vtz[s] + n * wc[s] * Tz_Tx[s]) \
                            / (lamDz[s] * k)**2
                        C[m] = kz * vz[s] + kp * vx[s] + n * wc[s] \
                             + cz[j] * kz * vtz[s]
                        m += 1
            else:
                bessel = 1. / (2. * N4k + 1.)
                for n in range(-N4k, N4k + 1):
                    for j in range(J):
                        B[m] = bessel * bz[j] * (cz[j] * k * vtz[s]) \
                             / (lamDz[s] * k)**2
                        C[m] = kz * vz[s] + kp * vx[s] + cz[j] * k * vtz[s]
                        m += 1

        for m in range(SNJ):
            M[m, :] = -B[m]
            M[m, m] += C[m]

        w = scipy.linalg.eigvals(M)

        sort_idx = slice(None)
        if sort in ["imag"]:
            sort_idx = np.argsort(w.imag + 1j * w.real)
        elif sort in ["real"]:
            sort_idx = np.argsort(w)
        elif sort != 'none':
            raise ValueError('`sort` value {} not recognized'.format(sort))
        ws.append(w[sort_idx])

    if Ns.min() == Ns.max():
        # if different k has different N, the size of ws are different
        return np.array(ws)
    else:
        return ws
