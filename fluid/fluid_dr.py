__all__ = ['k2w_es1d', 'k2w_es3d', 'k2w_em3d', 'k2w']

import numpy as np
import scipy.linalg


def k2w_es1d(kxs, species, params, sort='real', eigenvector=False):
    """Compute dispersion relation for a 1d multifluid-Poisson system.
    
    This function perhaps is simple enough to be incorporated into
    `k2w_es3d` and is implemented here to demonstrate the basic algorithm.

    Args:
        kxs (np.ndarray): An array of wavevector component values along `x`.
        species (np.ndarray or list): A `nSpecies*nComponents` matrix.
            The components are: `q, m, n0, v0x, p0, gamma`. An example:

                species = np.array([  
                    [q_e, m_e, n0_e, v0x_e, p0_e, gamma_e],  # electron  
                    [q_i, m_i, n0_i, v0x_i, p0_i, gamma_i],  # ion  
                ])

        params (dict): A dictionary with keys `epsilon0`.
        sort (str): `'real'` or `'imag'` or `'none'`. Order to sort results.
        eigenvector (bool): Switch to return eigenvector in addition to
            eigenvalues (frequencies).

    Returns:
        ws: `ws[ik, :]` contains imaginary frequencies for the `ik`'th
            `(kx, kz)` value.
        vrs: `vrs[ik, :, iw]` is the normalized right eigenvector
            corresponding to the eigenvalue `ws[ik, iw]`. Only returned if
            ``eigenvector=True``.
    """
    NN, VX = range(2)

    q, m, n, vx, p, gamma = np.rollaxis(species, axis=1)
    epsilon0 = params['epsilon0']

    rho = n * m
    cs2 = gamma * p / rho

    nSpecies = len(q)
    nSolutions = 2 * nSpecies
    nk = len(kxs)

    M = np.zeros((nSolutions, nSolutions), dtype=np.complex128)
    ws = np.zeros((nk, nSolutions), dtype=np.complex128)
    if eigenvector:
        vrs = np.zeros((nk, nSolutions, nSolutions), dtype=np.complex128)

    for ik in range(nk):
        M.fill(0)
        kx = kxs[ik]
        for j in range(nSpecies):
            idx = j * 2  # first index of fluid variables, i.e., number density

            # dn due to dn
            M[idx + NN, idx + NN] = -1j * kx * vx[j]
            # dn due to dvx
            M[idx + NN, idx + VX] = -1j * kx * n[j]

            # dvi due to dvi
            M[idx + VX, idx + VX] = -1j * kx * vx[j]

            # dvx due to dn (from dp, adiabatic EoS)
            M[idx + VX, idx + NN] = -1j * kx * cs2[j] / n[j]

            # dvx due to dn of all species (from phi or Ex, Gauss's law)
            for s in range(nSpecies):
                M[idx + VX, s * 2 + NN] += (q[j] / m[j]) * q[s] \
                                         * (-1j / kx /epsilon0)

        if eigenvector:
            d, vr = scipy.linalg.eig(M, right=True)
        else:
            d = scipy.linalg.eigvals(M)

        w = 1j * d

        sort_idx = slice(None)
        if sort in ['imag']:
            sort_idx = np.argsort(w.imag + 1j * w.real)
        elif sort in ['real']:
            sort_idx = np.argsort(w)
        elif sort != 'none':
            raise ValueError('`sort` value {} not recognized'.format(sort))
        ws[ik, :] = w[sort_idx]

        if eigenvector:
            vrs[ik, ...] = vr[:, sort_idx]

    if eigenvector:
        return ws, vrs
    else:
        return ws


def k2w_es3d(kxs, kzs, species, params, isMag=None, sort='real', eigenvector=False):
    """Compute dispersion relation for a 3d multifluid-Poisson system with
    background magnetic field along `z` and wavevector along `x` and `z`.

    Args:
        kxs (np.ndarray): An array of wavevector component along `x`.
        kzs (np.ndarray): An array of wavevector component along `z`.
        species (np.ndarray): A `nSpecies*nComponents` matrix. The components
            are: `q, m, n0, v0x, v0z, p0perp, p0para, gamma_perp, gamma_para`.
            An example:

                species = np.array([
                    [q_e, m_e, n0_e, v0x_e, v0z_e, p0perp_e, p0para_e,
                     gamma_perp_e, gamma_para_e],  # electron
                    [q_i, m_i, n0_i, v0x_i, v0z_i, p0perp_i, p0para_i,
                     gamma_perp_i, gamma_para_i],  # ion
                ])
        params (dict): A dictionary with keys `Bz`, `epsilon0`.
        isMag (list or None): A list of booleans to indicate wether each species
            is magnetized. If not set, all species are assumed to be magnetized.
        sort (str): `'real'` or `'imag'` or `'none'`. Order to sort results.
        eigenvector (bool): Switch to return eigenvector in addition to
            eigenvalues (frequencies).

    Returns:
        ws: `ws[ik, :]` contains imaginary frequencies for the `ik`'th
            `(kx, kz)` value.
        vrs: `vrs[ik, :, iw]` is the normalized right eigenvector
            corresponding to the eigenvalue `ws[ik, iw]`. Only returned if
            ``eigenvector=True``.
    """
    NN, VX, VY, VZ = range(4)

    q, m, n, vx, vz, p_perp, p_para, gamma_perp, gamma_para = \
            np.rollaxis(species, axis=1)
    nSpecies = len(q)
    B = params['Bz']
    epsilon0 = params['epsilon0']
    
    if isMag is None:
        isMag = [True] * nSpecies
    assert (len(isMag) == nSpecies)
    isMag = np.array(isMag, dtype=int)

    rho = n * m
    cs_para2 = gamma_para * p_para / rho
    cs_perp2 = gamma_perp * p_perp / rho
    wc = q * B / m * isMag

    nSolutions = 4 * nSpecies  # EM
    nk = len(kxs)

    M = np.zeros((nSolutions, nSolutions), dtype=np.complex128)
    ws = np.zeros((nk, nSolutions), dtype=np.complex128)
    if eigenvector:
        vrs = np.zeros((nk, nSolutions, nSolutions), dtype=np.complex128)

    for ik in range(nk):
        M.fill(0)
        kx = kxs[ik]
        kz = kzs[ik]
        k2 = kx**2 + kz**2
        k2 = max(k2, np.finfo(np.float32).eps)
        for j in range(nSpecies):
            idx = j * 4  # first index of fluid variables, i.e., number density

            # dn due to dn
            M[idx + NN, idx + NN] = -1j * (kx * vx[j] + kz * vz[j])
            # dn due to dvx
            M[idx + NN, idx + VX] = -1j * kx * n[j]
            # dn due to dvz
            M[idx + NN, idx + VZ] = -1j * kz * n[j]

            # dvi due to dvi
            M[idx + VX, idx + VX] = -1j * (kx * vx[j] + kz * vz[j])
            M[idx + VY, idx + VY] = -1j * (kx * vx[j] + kz * vz[j])
            M[idx + VZ, idx + VZ] = -1j * (kx * vx[j] + kz * vz[j])

            # dvx due to dn (from dp, adiabatic EoS)
            M[idx + VX, idx + NN] = -1j * kx * cs_perp2[j] / n[j]
            # dvz due to dn (from dp, adiabatic EoS)
            M[idx + VZ, idx + NN] = -1j * kz * cs_para2[j] / n[j]
            # dvx due to dvy (from vxB force)
            M[idx + VX, idx + VY] = wc[j]
            # dvy due to dvx (from vxB force)
            M[idx + VY, idx + VX] = -wc[j]

            # dvx due to dn of all species (from phi or E, Gauss's law)
            for s in range(nSpecies):
                idxs = s * 4
                M[idx + VX, idxs + NN] += (q[j] / m[j]) * q[s] \
                                        * (-1j * kx / k2 /epsilon0)
                M[idx + VZ, idxs + NN] += (q[j] / m[j]) * q[s] \
                                        * (-1j * kz / k2 /epsilon0)

        if eigenvector:
            d, vr = scipy.linalg.eig(M, right=True)
        else:
            d = scipy.linalg.eigvals(M)

        w = 1j * d

        sort_idx = slice(None)
        if sort in ['imag']:
            sort_idx = np.argsort(w.imag + 1j * w.real)
        elif sort in ['real']:
            sort_idx = np.argsort(w)
        elif sort != 'none':
            raise ValueError('`sort` value {} not recognized'.format(sort))
        ws[ik, :] = w[sort_idx]

        if eigenvector:
            vrs[ik, ...] = vr[:, sort_idx]

    if eigenvector:
        return ws, vrs
    else:
        return ws


def k2w_em3d(kxs, kzs, species, params, sort='real', eigenvector=False):
    """Compute dispersion relation for a 3d multifluid-Maxwell system with the
    background magnetic field along `z` and wavevector `k` along `x` and `z`.

    The basic algorithm is, for each `(kx, kz)`, find
    `w = -1j * (eigenvalues of M)`. Here, `M` is the matrix for the right hand
    side of linearized equations. Assume background magnetic field to be along
    `z` and `ky=0`.

    Relativistic effects, collisions, and spatial gradients are ignored.

    Args:
        kxs (np.ndarray): An array of wavevector component along `x`.
        kzs (np.ndarray): An array of wavevector component along `z`.
        species (np.ndarray): A `nSpecies*nComponents` matrix. The components
            are: `q, m, n0, v0x, v0y, v0z, p0perp, p0para, gamma_perp, gamma_para`.
            An example for plasma with isothermal electrons and adiabatic ions:  

                species = np.array([
                    [q_e, m_e, n0_e, v0x_e, v0y_e, v0z_e, p0perp_e, p0para_e,
                     gamma_perp_e, gamma_para_e],  # electron
                    [q_i, m_i, n0_i, v0x_i, v0y_i, v0z_i, p0perp_i, p0para_i,
                     gamma_perp_i, gamma_para_i],  # ion
                ])

        params (dict): A dictionary with keys `Bz`, `c`, `epsilon0`.
        sort (str): `'real'` or `'imag'` or `'none'`. Order to sort results.
        eigenvector (bool): Switch to return eigenvector in addition to
            eigenvalues (frequencies).

    Returns:
        ws: `ws[ik, :]` contains imaginary frequencies for the `ik`'th
            `(kx, kz)` value.
        vrs: `vrs[ik, :, iw]` is the normalized right eigenvector
            corresponding to the eigenvalue `ws[ik, iw]`. Only returned if
            ``eigenvector=True``.
    """
    NN, VX, VY, VZ = range(4)
    EX, EY, EZ, BX, BY, BZ = range(6)

    q, m, n, vx, vy, vz, p_perp, p_para, gamma_perp, gamma_para = \
            np.rollaxis(species, axis=1)
    B = params['Bz']
    c2 = params['c']**2  # light speed ^ 2
    epsilon0 = params['epsilon0']

    rho = n * m
    cs_para2 = gamma_para * p_para / rho
    cs_perp2 = gamma_perp * p_perp / rho
    if (np.abs(B) > np.finfo(np.float64).eps * 1e3):
        Delta = (p_para - p_perp) / B
    else:
        Delta = np.zeros((nSpecies))
    wc = q * B / m

    nSpecies = len(q)
    nSolutions = 4 * nSpecies + 6  # EM
    idxEM = 4 * nSpecies  # first index of EM field, i.e., Ex
    nk = len(kxs)

    M = np.zeros((nSolutions, nSolutions), dtype=np.complex128)
    ws = np.zeros((nk, nSolutions), dtype=np.complex128)
    if eigenvector:
        vrs = np.zeros((nk, nSolutions, nSolutions), dtype=np.complex128)

    for ik in range(nk):
        M.fill(0)
        kx = kxs[ik]
        kz = kzs[ik]
        for j in range(nSpecies):
            idx = j * 4  # first index of fluid variables, i.e., number density

            # dn due to dn
            M[idx + NN, idx + NN] = -1j * (kx * vx[j] + kz * vz[j])
            # dn due to dvx
            M[idx + NN, idx + VX] = -1j * kx * n[j]
            # dn due to dvz
            M[idx + NN, idx + VZ] = -1j * kz * n[j]

            # dvi due to dvi
            M[idx + VX, idx + VX] = -1j * (kx * vx[j] + kz * vz[j])
            M[idx + VY, idx + VY] = -1j * (kx * vx[j] + kz * vz[j])
            M[idx + VZ, idx + VZ] = -1j * (kx * vx[j] + kz * vz[j])

            # dvx due to dn (from dp, adiabatic EoS)
            M[idx + VX, idx + NN] = -1j * kx * cs_perp2[j] / n[j]
            # dvz due to dn (from dp, adiabatic EoS)
            M[idx + VZ, idx + NN] = -1j * kz * cs_para2[j] / n[j]
            # dvx due to dvy (from vxB force)
            M[idx + VX, idx + VY] = wc[j]
            # dvy due to dvx (from vxB force)
            M[idx + VY, idx + VX] = -wc[j]

            # dvi due to dEi
            M[idx + VX, idxEM + EX] = q[j] / m[j]
            M[idx + VY, idxEM + EY] = q[j] / m[j]
            M[idx + VZ, idxEM + EZ] = q[j] / m[j]
            # dvx due to dBy and dBz
            M[idx + VX, idxEM + BY] = -q[j] / m[j] * vz[j]
            M[idx + VX, idxEM + BZ] = +q[j] / m[j] * vy[j]
            # dvy due to dBx and dBz
            M[idx + VY, idxEM + BX] = +q[j] / m[j] * vz[j]
            M[idx + VY, idxEM + BZ] = -q[j] / m[j] * vx[j]
            # dvz due to dBx and dBy
            M[idx + VZ, idxEM + BX] = -q[j] / m[j] * vy[j]
            M[idx + VZ, idxEM + BY] = +q[j] / m[j] * vx[j]

            # dvi due to dBi and anisotropy
            M[idx + VX, idxEM + BX] += -1j * kz * Delta[j] / rho[j]
            M[idx + VY, idxEM + BY] += -1j * kz * Delta[j] / rho[j]
            M[idx + VZ, idxEM + BX] += -1j * kx * Delta[j] / rho[j]

            # dEi due to dn
            M[idxEM + EX, idx] = -q[j] * vx[j] / epsilon0
            M[idxEM + EY, idx] = -q[j] * vy[j] / epsilon0
            M[idxEM + EZ, idx] = -q[j] * vz[j] / epsilon0

            # dEi due to dvi
            M[idxEM + EX, idx + VX] = -q[j] * n[j] / epsilon0
            M[idxEM + EY, idx + VY] = -q[j] * n[j] / epsilon0
            M[idxEM + EZ, idx + VZ] = -q[j] * n[j] / epsilon0

        # dE due to dB
        M[idxEM + EX, idxEM + BY] = -1j * kz * c2
        M[idxEM + EY, idxEM + BX] = +1j * kz * c2
        M[idxEM + EY, idxEM + BZ] = -1j * kx * c2
        M[idxEM + EZ, idxEM + BY] = +1j * kx * c2

        # dB due to dE
        M[idxEM + BX, idxEM + EY] = +1j * kz
        M[idxEM + BY, idxEM + EX] = -1j * kz
        M[idxEM + BY, idxEM + EZ] = +1j * kx
        M[idxEM + BZ, idxEM + EY] = -1j * kx

        if eigenvector:
            d, vr = scipy.linalg.eig(M, right=True)
        else:
            d = scipy.linalg.eigvals(M)

        w = 1j * d

        sort_idx = slice(None)
        if sort in ['imag']:
            sort_idx = np.argsort(w.imag + 1j * w.real)
        elif sort in ['real']:
            sort_idx = np.argsort(w)
        elif sort != 'none':
            raise ValueError('`sort` value {} not recognized'.format(sort))
        ws[ik, :] = w[sort_idx]

        if eigenvector:
            vrs[ik, ...] = vr[:, sort_idx]

    if eigenvector:
        return ws, vrs
    else:
        return ws

"""
Newer implementation to reduce code duplication
"""

NN, VX, VY, VZ, PXX, PXY, PXZ, PYY, PYZ, PZZ = range(10)

def fill_coeff_mat_nv(M, idx, k, n, v, wc):
    kx, ky, kz = k
    vx, vy, vz = v
    wcx, wcy, wcz = wc

    M[idx + NN, idx + NN] = kz*vz + ky*vy + kx*vx
    M[idx + NN, idx + VX] = kx*n
    M[idx + NN, idx + VY] = ky*n
    M[idx + NN, idx + VZ] = kz*n

    M[idx + VX, idx + VX] = kz*vz + ky*vy + kx*vx
    M[idx + VX, idx + VY] = (+wcz) * 1j
    M[idx + VX, idx + VZ] = (-wcy) * 1j

    M[idx + VY, idx + NN] = 0
    M[idx + VY, idx + VX] = (-wcz) * 1j
    M[idx + VY, idx + VY] = kz*vz + ky*vy + kx*vx
    M[idx + VY, idx + VZ] = (+wcx) * 1j

    M[idx + VZ, idx + NN] = 0
    M[idx + VZ, idx + VX] = (+wcy) * 1j
    M[idx + VZ, idx + VY] = (-wcx) * 1j
    M[idx + VZ, idx + VZ] = kz*vz + ky*vy + kx*vx


def fill_coeff_mat_5m(M, idx, k, n, v, cs2, wc):
    kx, ky, kz = k

    fill_coeff_mat_nv(M, idx, k, n, v, wc)
    M[idx + VX, idx + NN] = kx * cs2 / n
    M[idx + VY, idx + NN] = ky * cs2 / n
    M[idx + VZ, idx + NN] = kz * cs2 / n
    

def fill_coeff_mat_10m(M, idx, k, m, n, v, p, wc):
    kx, ky, kz = k
    vx, vy, vz = v
    pxx, pxy, pxz, pyy, pyz, pzz = p
    wcx, wcy, wcz = wc

    fill_coeff_mat_nv(M, idx, k, n, v, wc)

    M[idx + NN, idx + PXX] = 0
    M[idx + NN, idx + PXY] = 0
    M[idx + NN, idx + PXZ] = 0
    M[idx + NN, idx + PYY] = 0
    M[idx + NN, idx + PYZ] = 0
    M[idx + NN, idx + PZZ] = 0
    M[idx + VX, idx + NN] = 0

    M[idx + VX, idx + PXX] = kx/(m*n)
    M[idx + VX, idx + PXY] = ky/(m*n)
    M[idx + VX, idx + PXZ] = kz/(m*n)
    M[idx + VX, idx + PYY] = 0
    M[idx + VX, idx + PYZ] = 0
    M[idx + VX, idx + PZZ] = 0

    M[idx + VY, idx + PXX] = 0
    M[idx + VY, idx + PXY] = kx/(m*n)
    M[idx + VY, idx + PXZ] = 0
    M[idx + VY, idx + PYY] = ky/(m*n)
    M[idx + VY, idx + PYZ] = kz/(m*n)
    M[idx + VY, idx + PZZ] = 0

    M[idx + VZ, idx + PXX] = 0
    M[idx + VZ, idx + PXY] = 0
    M[idx + VZ, idx + PXZ] = kx/(m*n)
    M[idx + VZ, idx + PYY] = 0
    M[idx + VZ, idx + PYZ] = ky/(m*n)
    M[idx + VZ, idx + PZZ] = kz/(m*n)

    M[idx + PXX, idx + NN] = 0
    M[idx + PXX, idx + VX] = 2*kz*pxz+2*ky*pxy+3*kx*pxx
    M[idx + PXX, idx + VY] = pxx*ky
    M[idx + PXX, idx + VZ] = pxx*kz
    M[idx + PXX, idx + PXX] = kz*vz+ky*vy+kx*vx
    M[idx + PXX, idx + PXY] = (2*wcz) * 1j
    M[idx + PXX, idx + PXZ] = (-2*wcy) * 1j
    M[idx + PXX, idx + PYY] = 0
    M[idx + PXX, idx + PYZ] = 0
    M[idx + PXX, idx + PZZ] = 0

    M[idx + PXY, idx + NN] = 0
    M[idx + PXY, idx + VX] = kz*pyz+ky*pyy+2*kx*pxy
    M[idx + PXY, idx + VY] = kz*pxz+2*ky*pxy+kx*pxx
    M[idx + PXY, idx + VZ] = pxy*kz
    M[idx + PXY, idx + PXX] = (-wcz) * 1j
    M[idx + PXY, idx + PXY] = kz*vz+ky*vy+kx*vx
    M[idx + PXY, idx + PXZ] = (wcx) * 1j
    M[idx + PXY, idx + PYY] = (wcz) * 1j
    M[idx + PXY, idx + PYZ] = (-wcy) * 1j
    M[idx + PXY, idx + PZZ] = 0

    M[idx + PXZ, idx + NN] = 0
    M[idx + PXZ, idx + VX] = kz*pzz+ky*pyz+2*kx*pxz
    M[idx + PXZ, idx + VY] = ky*pxz
    M[idx + PXZ, idx + VZ] = 2*kz*pxz+ky*pxy+kx*pxx
    M[idx + PXZ, idx + PXX] = (wcy) * 1j
    M[idx + PXZ, idx + PXY] = (-wcx) * 1j
    M[idx + PXZ, idx + PXZ] = kz*vz+ky*vy+kx*vx
    M[idx + PXZ, idx + PYY] = 0
    M[idx + PXZ, idx + PYZ] = (wcz) * 1j
    M[idx + PXZ, idx + PZZ] = (-wcy) * 1j

    M[idx + PYY, idx + NN] = 0
    M[idx + PYY, idx + VX] = kx*pyy
    M[idx + PYY, idx + VY] = 2*kz*pyz+3*ky*pyy+2*kx*pxy
    M[idx + PYY, idx + VZ] = kz*pyy
    M[idx + PYY, idx + PXX] = 0
    M[idx + PYY, idx + PXY] = (-2*wcz) * 1j
    M[idx + PYY, idx + PXZ] = 0
    M[idx + PYY, idx + PYY] = kz*vz+ky*vy+kx*vx
    M[idx + PYY, idx + PYZ] = (2*wcx) * 1j
    M[idx + PYY, idx + PZZ] = 0

    M[idx + PYZ, idx + NN] = 0
    M[idx + PYZ, idx + VX] = kx*pyz
    M[idx + PYZ, idx + VY] = kz*pzz+2*ky*pyz+kx*pxz
    M[idx + PYZ, idx + VZ] = 2*kz*pyz+ky*pyy+kx*pxy
    M[idx + PYZ, idx + PXX] = 0
    M[idx + PYZ, idx + PXY] = (wcy) * 1j
    M[idx + PYZ, idx + PXZ] = (-wcz) * 1j
    M[idx + PYZ, idx + PYY] = (-wcx) * 1j
    M[idx + PYZ, idx + PYZ] = kz*vz+ky*vy+kx*vx
    M[idx + PYZ, idx + PZZ] = (wcx) * 1j

    M[idx + PZZ, idx + NN] = 0
    M[idx + PZZ, idx + VX] = kx*pzz
    M[idx + PZZ, idx + VY] = ky*pzz
    M[idx + PZZ, idx + VZ] = 3*kz*pzz+2*ky*pyz+2*kx*pxz
    M[idx + PZZ, idx + PXX] = 0
    M[idx + PZZ, idx + PXY] = 0
    M[idx + PZZ, idx + PXZ] = (2*wcy) * 1j
    M[idx + PZZ, idx + PYY] = 0
    M[idx + PZZ, idx + PYZ] = (-2*wcx) * 1j
    M[idx + PZZ, idx + PZZ] = kz*vz+ky*vy+kx*vx


def k2w(kxs, kys, kzs, species, params, isMag=None, 
                moment='10m', em_or_es='es', sort='real', eigenvector=False):
    if moment == '10m':
        nComps = 10
        q, m, n, vx, vy, vz, pxx, pxy, pxz, pyy, pyz, pzz = \
                np.rollaxis(species, axis=1)
        v = np.array((vx, vy, vz))
        p = np.array((pxx, pxy, pxz, pyy, pyz, pzz))
    elif moment == '5m':
        nComps = 4
        q, m, n, vx, vy, vz, p, gamma = \
                np.rollaxis(species, axis=1)
        v = np.array((vx, vy, vz))
        cs2 = gamma * p / n / m
    else:
        raise ValueError(f'moment {moment} not supported')
    nSpecies = len(q)

    epsilon0 = params['epsilon0']
    Bx = params.setdefault('Bx', 0)
    By = params.setdefault('By', 0)
    Bz = params.setdefault('Bz', 0)
   
    if isMag is None:
        isMag = [True] * nSpecies
    assert (len(isMag) == nSpecies)
    isMag = np.array(isMag, dtype=int)

    wcx = q * Bx / m * isMag
    wcy = q * By / m * isMag
    wcz = q * Bz / m * isMag
    wc = np.array((wcx, wcy, wcz))

    if em_or_es == 'es':
        nSolutions = nComps * nSpecies
    elif em_or_es == 'em':
        nSolutions = nComps * nSpecies + 6
    else:
        raise ValueError(f'em_or_es {em_or_es} not supported')
    nk = len(kxs)

    M = np.zeros((nSolutions, nSolutions), dtype=np.complex128)
    ws = np.zeros((nk, nSolutions), dtype=np.complex128)
    if eigenvector:
        vrs = np.zeros((nk, nSolutions, nSolutions), dtype=np.complex128)

    for ik in range(nk):
        M.fill(0)
        kx = kxs[ik]
        ky = kys[ik]
        kz = kzs[ik]
        k = (kx, ky, kz)
        k2 = kx*kx + ky*ky + kz*kz

        for j in range(nSpecies):
            # first index of fluid variables, i.e., number density
            idx = j * nComps
            
            if moment == '10m':
                fill_coeff_mat_10m(M, idx, k, m[j], n[j], v[:, j], p[:, j], wc[:, j])
            elif moment == '5m':
                fill_coeff_mat_5m(M, idx, k, n[j], v[:, j], cs2[j], wc[:, j])
            else:
                raise ValueError(f'moment {moment} not supported')

            if em_or_es == 'es':
                # dvx due to dn of all species (from phi or E, Gauss's law)
                for s in range(nSpecies):
                    idxs = s * nComps
                    phi = q[s] / k2 /epsilon0
                    M[idx + VX, idxs + NN] += (q[j] / m[j]) * kx * phi
                    M[idx + VY, idxs + NN] += (q[j] / m[j]) * ky * phi
                    M[idx + VZ, idxs + NN] += (q[j] / m[j]) * kz * phi
            elif em_or_es == 'em':
                raise NotImplementedError('em not implemented yet')
            else:
                raise ValueError(f'em_or_es {em_or_es} not supported')

        if eigenvector:
            w, vr = scipy.linalg.eig(M, right=True)
        else:
            w = scipy.linalg.eigvals(M)

        sort_idx = slice(None)
        if sort in ['imag']:
            sort_idx = np.argsort(w.imag + 1j * w.real)
        elif sort in ['real']:
            sort_idx = np.argsort(w)
        elif sort != 'none':
            raise ValueError('`sort` value {} not recognized'.format(sort))
        ws[ik, :] = w[sort_idx]

        if eigenvector:
            vrs[ik, ...] = vr[:, sort_idx]

    if eigenvector:
        return ws, vrs
    else:
        return ws

