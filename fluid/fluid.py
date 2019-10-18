__all__ = ['k2w_es1d', 'k2w_es3d', 'k2w_em3d']

import numpy as np
import scipy.linalg


def k2w_es1d(kxs, species, params, sort='real', eigenvector=False):
    """Compute dispersion relation for a 1d multifluid-Poisson system.

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
        ws (np.ndarray): `ws[ik, :]` is the maginary frequency for the `ik`'th
            `kx` value.
        vrs (np.ndarray): `vrs[ik, ...]` is the normalized right eigenvector
            corresponding to the eigenvalue `w[ik, is]` is the column
            `vr[ik, :, is]`. Only returned if `eigenvector=True`.
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


def k2w_es3d(kxs, kzs, species, params, sort='real', eigenvector=False):
    """Compute dispersion relation for a 3d multifluid-Poisson system with
    background magnetic field along `z` and wavevector along `x` and `z`.

    Args:
        kxs (np.ndarray): An array of wavevector component along `x`.
        kzs (np.ndarray): An array of wavevector component along `z`.
        species (np.ndarray): A `nSpecies*nComponents` matrix. The components
            are: `q, m, n0, v0x, v0y, v0z, p0perp, p0para, gamma_perp, gamma_para`.
            An example:

                species = np.array([
                    [q_e, m_e, n0_e, v0x_e, v0z_e, p0perp_e, p0para_e,
                     gamma_perp_e, gamma_para_e],  # electron
                    [q_i, m_i, n0_i, v0x_i, v0z_i, p0perp_i, p0para_i,
                     gamma_perp_i, gamma_para_i],  # ion
                ])
        params (dict): A dictionary with keys `Bz`, `epsilon0`.
        sort (str): `'real'` or `'imag'` or `'none'`. Order to sort results.
        eigenvector (bool): Switch to return eigenvector in addition to
            eigenvalues (frequencies).

    Returns:
        ws: `ws[ik, :]` is the maginary frequency for the `ik`'th `(kx, kz)`
            value.
        vrs: `vrs[ik, ...]` is the normalized right eigenvector corresponding to
            the eigenvalue `w[ik, is]` is the column `vr[ik, :, is]`. Only
            returned if ``eigenvector=True``.
    """
    NN, VX, VZ = range(3)

    q, m, n, vx, vz, p_perp, p_para, gamma_perp, gamma_para = \
            np.rollaxis(species, axis=1)
    B = params['Bz']
    epsilon0 = params['epsilon0']

    rho = n * m
    cs_para2 = gamma_para * p_para / rho
    cs_perp2 = gamma_perp * p_perp / rho

    nSpecies = len(q)
    nSolutions = 3 * nSpecies  # EM
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
        for j in range(nSpecies):
            idx = j * 3  # first index of fluid variables, i.e., number density

            # dn due to dn
            M[idx + NN, idx + NN] = -1j * (kx * vx[j] + kz * vz[j])
            # dn due to dvx
            M[idx + NN, idx + VX] = -1j * kx * n[j]
            # dn due to dvz
            M[idx + NN, idx + VZ] = -1j * kz * n[j]

            # dvi due to dvi
            M[idx + VX, idx + VX] = -1j * (kx * vx[j] + kz * vz[j])
            M[idx + VZ, idx + VZ] = -1j * (kx * vx[j] + kz * vz[j])

            # dvx due to dn (from dp, adiabatic EoS)
            M[idx + VX, idx + NN] = -1j * kx * cs_perp2[j] / n[j]
            # dvz due to dn (from dp, adiabatic EoS)
            M[idx + VZ, idx + NN] = -1j * kz * cs_para2[j] / n[j]

            # dvx due to dn of all species (from phi or E, Gauss's law)
            for s in range(nSpecies):
                idxs = s * 3
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
        ws (np.ndarray): `ws[ik, :]` is the maginary frequency for the `ik`'th
            `(kx, kz)` value.
        vrs (np.ndarray): `vrs[ik, ...]` is the normalized right eigenvector
            corresponding to the eigenvalue `w[ik, is]` is the column
            `vr[ik, :, is]`. Only returned if `eigenvector=True`.
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
            M[idx + VX, idx + VY] += wc[j]
            # dvy due to dvx (from vxB force)
            M[idx + VY, idx + VX] -= wc[j]

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
