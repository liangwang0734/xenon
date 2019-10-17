__all__ = ['bzs', 'czs']

import numpy as np

bzs = {}
czs = {}

def prep_bcz(J):
    """Compute J-pole approximation coefficients for Z(zeta).
    
    Args:
        J: Order.
    Return:
        bz and cz.
    """
    if J in bzs and J in czs:
        return bzs[J], czs[J]

    Jhalf = J // 2
    bz = np.zeros(J, dtype=np.complex128)
    cz = np.zeros(J, dtype=np.complex128)

    if J == 8:
        bz[:Jhalf] = [
            -1.734012457471826E-2 - 4.630639291680322E-2j,
            -7.399169923225014E-1 + 8.395179978099844E-1j,
            5.840628642184073 + 9.536009057643667E-1j,
            -5.583371525286853 - 1.120854319126599E1j,
        ]
        cz[:Jhalf] = [
            2.237687789201900 - 1.625940856173727j,
            1.465234126106004 - 1.789620129162444j,
            .8392539817232638 - 1.891995045765206j,
            .2739362226285564 - 1.941786875844713j,
        ]
    elif J == 12:
        bz[:Jhalf] = [
            -0.00454786121654587 - 0.000621096230229454j,
            0.215155729087593 + 0.201505401672306j,
            0.439545042119629 + 4.16108468348292j,
            -20.2169673323552 - 12.8855035482440j,
            67.0814882450356 + 20.8463458499504j,
            -48.0146738250076 + 107.275614092570j,
        ]

        cz[:Jhalf] = [
            -2.97842916245164 - 2.04969666644050j,
            2.25678378396682 - 2.20861841189542j,
            -1.67379985617161 - 2.32408519416336j,
            -1.15903203380422 - 2.40673940954718j,
            0.682287636603418 - 2.46036501461004j,
            -0.225365375071350 - 2.48677941704753j,
        ]

    bz[Jhalf:] = np.conjugate(bz[:Jhalf])
    cz[Jhalf:] = -np.conjugate(cz[:Jhalf])

    bz.flags.writeable = False
    cz.flags.writeable = False

    # print("J", J)
    # print("sum(bz)       = {} ~ -1".format(sum(bz)))  # -1
    # print("sum(bz*cz)    = {} ~ 0".format(sum(bz * cz)))  # 0
    # print("sum(bz*cz**2) = {} ~ -0.5".format(sum(bz * cz**2)))  # -0.5
    # print("sum(bz*cz**3) = {} ~ 0".format(sum(bz * cz**3)))  # 0
    return bz, cz

for J in [8, 12]:
    bzs[J], czs[J] = prep_bcz(J)


