__all__ = ['k2w_es1d', 'k2w_es3d', 'k2w_em3d']

import numpy as np
import scipy.linalg

NN, VX, VY, VZ, PXX, PXY, PXZ, PYY, PYZ, PZZ = range(10)

def fill_coeff_mat_nv(M, idx, k, n, v, wc):
    kx, ky, kz = k
    vx, vy, kz = v
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


def fill_coeff_mat_adiabatic(M, idx, k, n, v, cs2, wc):
    kx, ky, kz = k
    cs2x, cs2y, kz = cs2

    fill_coeff_mat_nv(M, idx, k, n, v, wc)
    M[idx + VX, idx + NN] = kx * cs2x / n
    M[idx + VY, idx + NN] = kx * cs2y / n
    M[idx + VZ, idx + NN] = kz * cs2z / n
    

def fill_coeff_mat_10m(M, idx, k, m, n, v, p, wc):
    kx, ky, kz = k
    vx, vy, vz = v
    pxx, pxy, pxz, pyy, pyz, pzz = p

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
    elif moment == '6m':
        nComps = 6
        q, m, n, vx, vy, vz, p_para, p_perp, gamma_para, gamma_perp = \
                np.rollaxis(species, axis=1)
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
    wc = (wcx, wcy, wcz)

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

            fill_coeff_mat_10m(M, idx, k, m, n, v, p, wc)

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

