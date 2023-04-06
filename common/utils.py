__all__ = [
    'plot_dr',
    'extra_params',
    'draw_extra_params',
]

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def plot_dr(
        ks,
        ws,
        knorm=1.,
        wnorm=1.,
        wi_mask_funcs=[],
        wri_mask_funcs=[],
        knorm_name='',
        wnorm_name='',
        wrmin=-np.inf,
        wrmax=np.inf,
        wimin=-np.inf,
        wimax=np.inf,
        w_min_max_is_normed=True,
        ax0=None,
        ax1=None,
        pargs0={},
        pargs1={},
):
    r"""Plot dispersion relation as `k-wr` and `k-wi` dots, where `wr` and `wi`
    are real and imaginary parts of a complex frequency `w`.

    Args:
        ks: An array of `k` values.
        ws: A 2d array of `w` values of shape `[nk, nSolutions]`, or a list of
            length `len(k)` with different number of `w` values at different
            `k`.
        ax0: Axes to plot `k-wr`. If it is None, `k-wr` will not be plotted.
        ax1: Axes to plot `k-wi`. If it is None, `k-wi` will not be plotted.
        knorm: `k` values will be normalized by `knorm` before being used.
        wnorm: `w` values will be normalized by `wnorm` before being used.
        knorm_name: For example, `'/k'`.
        wnorm_name: For example, `'/\omega_{ci}'`.
        w_min_max_is_normed: If `False`, the `wimin`, `wimax`, etc. will be
            multiplied by `wnorm` before being used. The default is `True`,
            i.e., the `wimin`, etc. are values after normalization.
        wi_mask_funcs: A list of functions that return mask arrays using
            normalized `k` and `wi` as input. For example, to remove heavily
            damped solutions below the line `((0, 0), (1, -0.5))` on the `k-wi`
            plane, use

                def wi_mask_func0(k, wi):
                    return wi > k* (0+0.5) / (0-1)
            or

                def wi_mask_func0(k, wi):
                    k0, wi0 = 1, -0.5
                    k1, wi1 = 0, 0
                    return wi > (k-k0) * (wi1-wi0) / (k1-k0) + wi0

            Another example is to keep modes within a region:

                def wi_mask_func1(k, wi):
                    k0, wi0 = 0, 0
                    k1, wi1 = 5, -0.75
                    k2, wi2 = 5, -4
                    return (wi < (k-k0) * (wi1-wi0) / (k1-k0) + wi0) \  
                         & (wi > (k-k0) * (wi2-wi0) / (k2-k0) + wi0)

            A more complex example:

                def above_border(k, wi, border_line):
                    "Check if points are above or below a border_line
                    Args:
                    k: 1d array of k coordinates
                    wi: 1d array of wi coordinates
                    border_line: n*2 array, n is the # of points defining the
                        line
                    
                    Returns:
                    mask: 1d array
                    "
                    n_nodes = border_line.shape[0]

                    # find the indices of the points with the smallest k
                    # coordinate that is greater than each k coordinate in the
                    # to check array
                    j_values = np.searchsorted(border_line[:, 0], k)
                    j_values[j_values >= n_nodes] = n_nodes - 1

                    # get the two points on either side of each point in the k
                    # array
                    p1 = border_line[j_values - 1]
                    p2 = border_line[j_values]

                    # calculate the slope and y-intercept of the line connecting
                    # the two points
                    m_values = (p2[:, 1] - p1[:, 1]) / (p2[:, 0] - p1[:, 0])
                    b_values = p1[:, 1] - m_values * p1[:, 0]

                    # calculate the y-coordinate of the point on the line with
                    # the same k
                    # coordinate as each point in the k array
                    y_line_values = m_values * k + b_values

                    # create a mask array to store whether each point is above
                    # or below the border line
                    mask = wi >= y_line_values

                    return mask

                def wi_mask_func1(k, wi):
                    border_line = np.array([
                        [0, -0.1],
                        [0.8, -1],
                        [1, -1.56],
                    ])
                    mask = above_border(k, wi, border_line)
                    return mask

        wri_mask_funcs: A list of functions that return mask arrays using
            normalized wr and wi as input. For example, to remove heavily damped
            modes, use

                def wri_mask_func0(wr, wi):
                    return wr >= - abs(wr) / (2.*np.pi)
    """
    if ax0 is None and ax1 is None:
        return

    if not w_min_max_is_normed:
        wimin *= wnorm
        wimax *= wnorm
        wrmin *= wnorm
        wrmax *= wnorm

    ks_ = ks / knorm
    if isinstance(ws, np.ndarray):
        ws_ = ws / wnorm
    elif isinstance(ws, list):
        ws_ = [w / wnorm for w in ws]
    else:
        raise TypeError('type(ws) {} =/= ndarray or list'.format(type(ws)))

    if isinstance(ws, np.ndarray):
        for iSol in range(ws.shape[1]):
            w = ws_[:, iSol]
            wi = np.imag(w)
            wr = np.real(w)

            mask = (wi > wimin) & (wi < wimax) & (wr > wrmin) & (wr < wrmax)
            if len(wi_mask_funcs) + len(wri_mask_funcs) > 0:
                for wi_mask_func in wi_mask_funcs:
                    mask = mask & (wi_mask_func(ks_, wi))
                for wri_mask_func in wri_mask_funcs:
                    mask = mask & (wri_mask_func(wr, wi))
            wr = wr[mask]
            wi = wi[mask]
            k = ks_[mask]
            if len(k) == 0:
                continue

            if ax0 is not None:
                sc0 = ax0.scatter(k, wr, **pargs0)
            if ax1 is not None:
                sc1 = ax1.scatter(k, wi, **pargs1)
    elif isinstance(ws, list):
        nk = len(ks_)
        for ik, kk in enumerate(ks_):
            wreal = ws_[ik].real
            wimag = ws_[ik].imag

            mask = (wimag > wimin) & (wimag < wimax) & (wreal > wrmin) & (wreal < wrmax)
            wr = wreal[mask]
            wi = wimag[mask]

            nSols = len(ws_[ik])
            k = np.full((nSols), kk)[mask]
            if len(k) == 0:
                continue

            if ax0 is not None:
                sc0 = ax0.scatter(k, wr, **pargs0)
            if ax1 is not None:
                sc1 = ax1.scatter(k, wi, **pargs1)

    if ax0 is not None:
        ax0.set_ylabel(r'$\omega_R{}$'.format(wnorm_name))
        ax0.set_xlabel(r'$k{}$'.format(knorm_name))
        ax0.set_xlim(ks_.min(), ks_.max())
    if ax1 is not None:
        ax1.set_ylabel(r'$\gamma{}$'.format(wnorm_name))
        ax1.set_xlabel(r'$k{}$'.format(knorm_name))
        ax1.set_xlim(ks_.min(), ks_.max())


class extra_params():
    """A class to compute useful variables from basic parameters for repeated
    usage."""
    def __init__(self, species, params, problem_type=None):
        """
        Args:
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
        """
        if species.shape[1] == 10:  # fluid em3d
            q, m, n, vx, vy, vz, p_perp, p_para, gamma_perp, gamma_para = np.rollaxis(
                species, axis=1)
            if problem_type is None:
                problem_type = 'em3d'
            else:
                assert problem_type == 'em3d'

        elif species.shape[1] == 9:  # fluid es3d
            q, m, n, vx, vz, p_perp, p_para, gamma_perp, gamma_para = np.rollaxis(
                species, axis=1)
            vy = 0
            if problem_type is None:
                problem_type = 'es3d'
            else:
                assert problem_type == 'es3d'

        elif species.shape[1] == 6:  # fluid es1d
            q, m, n, vx, p, gamma = np.rollaxis(species, axis=1)
            vy = 0
            vz = 0
            p_perp = p
            p_para = p
            gamma_perp = gamma
            gamma_para = gamma
            if problem_type is None:
                problem_type = 'es1d'
            else:
                assert problem_type == 'es1d'

        elif species.shape[1] == 7:  # vlasov es3d
            q, m, n, vx, vz, p_perp, p_para = np.rollaxis(
                    np.array(species), axis=1)
            vy = 0
            if problem_type is None:
                problem_type = 'es3d'
            else:
                assert problem_type == 'es3d'

        elif species.shape[1] == 5:  # vlasov es1d
            q, m, n, vx, p = np.rollaxis(species, axis=1)
            gamma = 1
            vy = 0
            vz = 0
            p_perp = p
            p_para = p
            gamma_perp = gamma
            gamma_para = gamma
            if problem_type is None:
                problem_type = 'es1d'
            else:
                assert problem_type == 'es1d'

        else:
            raise ValueError(
                    '`species` must have 6 (fluid es1d), 9 (fluid es3d) 10 '
                    '(fluid em3d) , 5 (vlasov es1d), or 7 (vlasov es3d) '
                    'components.')

        nSpecies = len(q)

        B, c = 0, 1
        if problem_type in ['es3d', 'em3d']:
            B = params['Bz']
        if problem_type in ['em3d']:
            c = params['c']
        epsilon0 = params['epsilon0']

        c2 = c**2
        mu0 = 1 / c2 / epsilon0
        wp = np.sqrt(n * q**2. / epsilon0 / m)
        wp_tot = np.linalg.norm(wp)
        wc = q * B / m
        cs = np.sqrt(gamma_para * p_para / n / m)  # sound speeds
        rho_m = (n * m).sum()
        vAlf = np.sqrt(B**2 / mu0 / (m * n))
        if abs(B) > 0:
            vAlf_tot = np.sqrt(1 / (1 / vAlf**2).sum())
        else:
            vAlf_tot = 0
        vAlf2 = vAlf_tot**2
        # magnetosonic speed
        if (np.abs(B) > np.finfo(np.float64).eps * 1e3):
            vMS = np.sqrt(c2 * vAlf2 * (1 + (cs**2 / vAlf**2).sum()) /
                          (vAlf2 + c2))
        else:
            vMS = c2 * vAlf2 / (c2 + vAlf2)

        self.q = q
        self.m = m
        self.n = n
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.p_para = p_para
        self.p_perp = p_perp
        self.gamma_para = gamma_para
        self.gamma_perp = gamma_perp
        self.nSpecies = nSpecies

        # kB * T
        T_para = p_para / n
        T_perp = p_perp / n
        T = p_para / n  # FIXME more rigorous

        lambdaD = np.sqrt(epsilon0 * T / n / q**2)

        self.T_para = T_para
        self.T_perp = T_perp
        self.T = T
        self.lambdaD = lambdaD

        for val, key in enumerate(params):
            setattr(self, key, val)

        self.c = np.sqrt(c2)
        self.wp = wp
        self.wp_tot = wp_tot
        self.wc = wc
        self.cs = cs
        self.rho_m = rho_m
        self.vAlf = vAlf
        self.vAlf_tot = vAlf_tot
        self.vMS = vMS

        if nSpecies == 2 and q[0] < 0. and q[1] > 0:
            wpe = self.wp[0]
            wpi = self.wp[1]
            wce = self.wc[0]
            wci = self.wc[1]
            wp = self.wp_tot
            # exact lower and higher hybrid frequencies; solutions of S=0
            w2_sum = 0.5 * (wpe**2 + wpi**2 + wce**2 + wci**2)
            w2_perm = wpe**2 * wci**2 + wpi**2 * wce**2 + wce**2 * wci**2
            self.wLH = np.sqrt(w2_sum - np.sqrt(w2_sum**2 - w2_perm))
            self.wUH = np.sqrt(w2_sum + np.sqrt(w2_sum**2 - w2_perm))
            # exact cutoff frequencies left- and right-handed
            # circularly-polarized waves solutions of L=0 and R=0
            self.wR = 0.5 * (abs(wce) - wci) + np.sqrt((0.5 *
                                                        (abs(wce) + wci))**2 +
                                                       wp**2)
            self.wL = 0.5 * (wci - abs(wce)) + np.sqrt((0.5 *
                                                        (abs(wce) + wci))**2 +
                                                       wp**2)

            self.wpe = wpe
            self.wpi = wpi
            self.wce = wce
            self.wci = wci
            self.cse = self.cs[0]
            self.csi = self.cs[1]
            self.vAlfe = self.vAlf[0]
            self.vAlfi = self.vAlf[1]
            self.pe = self.p_para[0]
            self.pi = self.p_para[1]
            self.rhoe = self.n[0] * self.m[0]
            self.rhoi = self.n[1] * self.m[1]


def draw_extra_params(params, ax, ks=None, what=[], cost=None):
    """

    Args:
        params: A extra_params instance.
        what: A list of various curves to draw. Eligible elements are:
            vAlf, vMs, wp, wpe, wpi, wce, wci, wUH, wLH, wL, wR, wce, wci.
        cost: cos(theta) to be used with vAlf, vMs, wce, wci.
    """
    if 'vAlf' in what:
        ax.plot(ks, ks * params.vAlf_tot, label=r'$kv_{A}$', c='r')
    if 'vAlf*cost' in what:
        ax.plot(ks,
                ks * params.vAlf_tot * cost,
                label=r'$kv_{A}\cos\theta$',
                lw=2,
                ls='dotted',
                c='r')
    if 'vMS' in what:
        ax.plot(ks, ks * params.vMS, label=r'$kv_{\rm{magsonic}}$', c='b')
    if 'vMS*cost' in what:
        ax.plot(ks,
                ks * params.vMS * cost,
                label=r'$kv_{\rm{magsonic}}\cos\theta$',
                lw=2,
                ls='dotted',
                c='b')

    if 'wp' in what:
        ax.axhline(params.wp_tot, label=r'$\omega_{p}$', c='r', ls='dotted')
    if 'wpi' in what:
        ax.axhline(params.wpi, label=r'$\omega_{pi}$', ls='dotted', c='g')
    if 'wpe' in what:
        ax.axhline(params.wpe, label=r'$\omega_{pe}$', ls='dotted', c='b')
    if 'wUH' in what:
        ax.axhline(params.wUH, label=r'$\omega_{UH}$', ls='--',
                   c='c')  # upper hybrid
    if 'wLH' in what:
        ax.axhline(params.wLH, label=r'$\omega_{LH}$', ls='--',
                   c='m')  # lower hybrid
    if 'wL' in what:
        ax.axhline(params.wL, label=r'$\omega_{L}$', ls='--', c='k')
    if 'wR' in what:
        ax.axhline(params.wR, label=r'$\omega_{R}$', ls='--', c='g')
    if 'wci' in what:
        ax.axhline(params.wci * cost,
                   label=r'$\omega_{ci}\cos\theta$',
                   ls='dotted',
                   c='steelblue')
    if 'wce' in what:
        ax.axhline(abs(params.wce * cost),
                   label=r'$\omega_{ce}\cos\theta$',
                   ls='dotted',
                   c='maroon')

    if 'acoustic' in what or 'aw' in what:
        line_kwargs = dict(alpha=0.8, lw=5, ls='dotted', c='b', label='AW')
        wp2 = params.wp_tot**2
        ci2 = params.csi**2
        ce2 = params.cse**2
        wpi2 = params.wpi**2
        wpe2 = params.wpe**2
        ks2 = ks**2
        ws = np.empty((len(ks), 4), dtype=np.complex128)
        for ik, k2 in enumerate(ks2):
            ws[ik, :] = np.roots(
                (1, 0, -(wp2 + k2 * ci2 + k2 * ce2), 0,
                 k2 * ci2 * wpe2 + k2 * ce2 * wpi2 + k2**2 * ce2 * ci2))
        ax.plot(
            ks,
            ws[:, 3].real,  # FIXME sort result?
            **line_kwargs)
    if 'acoustic*cost' in what or 'aw*cost' in what:
        line_kwargs = dict(
            alpha=0.8,
            lw=5,
            ls='dotted',
            c='orange',
            label=
            r'$k\sqrt{\frac{\omega_{pe}^{2}c_{i}^{2}+\omega_{pi}^{2}c_{e}^{2}}{\omega_{pi}^{2}+\omega_{pe}^{2}}}\cos{\theta}$'
        )
        ci2 = params.csi**2
        ce2 = params.cse**2
        wpi2 = params.wpi**2
        wpe2 = params.wpe**2
        v_acoustic = np.sqrt((wpe2 * ci2 + wpi2 * ce2) / (wpi2 + wpe2))
        ax.plot(ks, ks * v_acoustic * cost, **line_kwargs)

    if 'ion acoustic' in what or 'iaw' in what:
        # valid for mi >> me
        line_kwargs = dict(alpha=0.8, lw=5, ls='dotted', c='c', label='IAW')
        if params.pe > 0:
            kLambdaD2 = ks**2 * params.pe / params.wpe**2 / params.rhoe
            ax.plot(ks, params.wpi / np.sqrt(1 + 1 / kLambdaD2), **line_kwargs)
        else:
            ax.plot((np.nan), (np.nan), **line_kwargs)


def calc_wh(wpe, wpi, wce, wci):
    w2_sum = 0.5 * (wpe**2 + wpi**2 + wce**2 + wci**2)
    w2_perm = wpe**2 * wci**2 + wpi**2 * wce**2 + wce**2 * wci**2
    wLH = np.sqrt(w2_sum - np.sqrt(w2_sum**2 - w2_perm))
    wUH = np.sqrt(w2_sum + np.sqrt(w2_sum**2 - w2_perm))
    return wLH, wUH


def calc_wLH(wpe, wpi, wce, wci):
    """Compute exact lower hybrid frequency for an electron-ion plasma.
    """
    wLH, wUH = calc_wh(wpe, wpi, wce, wci)
    return wLH


def calc_wUH(wpe, wpi, wce, wci):
    """Compute exact upper hybrid frequency for an electron-ion plasma.
    """
    wLH, wUH = calc_wh(wpe, wpi, wce, wci)
    return wUH


def calc_wR(wpe, wpi, wce, wci):
    """Compute exact cutoff frequencies left-handed circularly-polarized waves.

    The wave is solution of L=0 in Stix's notations.
    """
    wce = abs(wce)
    return 0.5 * (wce - wci) + np.sqrt((0.5 * (wce + wci))**2 + wp**2)


def calc_wL(wpe, wpi, wce, wci):
    """Compute exact cutoff frequencies right-handed circularly-polarized waves.

    The wave is solution of R=0 in Stix's notations.
    """
    wce = abs(wce)
    return 0.5 * (wci - wce) + np.sqrt((0.5 * (wce + wci))**2 + wp**2)

