__all__ = [
    'plot_dr',
]

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from ..fluid import fluid


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
