# coding: utf-8

import numpy as np
from numba import njit, prange


# ==========================================================
# Direction Cosines
# ==========================================================

@njit
def di2xyz(dec, inc):
    """
    偏角(dec: degree)・伏角(inc: degree)
    → 単位ベクトル
    """

    dec_rad = dec * np.pi / 180.0
    inc_rad = inc * np.pi / 180.0

    x = np.cos(inc_rad) * np.sin(dec_rad)
    y = np.cos(inc_rad) * np.cos(dec_rad)
    z = -np.sin(inc_rad)

    return np.array([x, y, z])


# ==========================================================
# Prism kernel
# ==========================================================

@njit
def _prism_kernel_func_(mgz, x, y, z):
    """
    磁化ブロックによる磁気異常の磁場3成分(不定積分)を計算

    mgz: 磁化ベクトル (jx, jy, jz)
    x: 観測点〜ソースの位置ベクトル x座標 (xsrc - xobs)
    y: y座標
    z: z座標
    """

    r = np.sqrt(x**2 + y**2 + z**2)

    jx = mgz[0]
    jy = mgz[1]
    jz = mgz[2]

    eps = np.finfo(np.float64).eps

    # ------------------------------------------------------
    # log(r + x)
    # ------------------------------------------------------

    lnx = np.empty(len(r))

    for i in range(len(r)):

        if np.abs(r[i] + x[i]) > eps:
            lnx[i] = np.log(r[i] + x[i])
        else:
            lnx[i] = -np.log(r[i] - x[i])

    # ------------------------------------------------------
    # log(r + y)
    # ------------------------------------------------------

    lny = np.empty(len(r))

    for i in range(len(r)):

        if np.abs(r[i] + y[i]) > eps:
            lny[i] = np.log(r[i] + y[i])
        else:
            lny[i] = -np.log(r[i] - y[i])

    # ------------------------------------------------------
    # log(r + z)
    # ------------------------------------------------------

    lnz = np.empty(len(r))

    for i in range(len(r)):

        if np.abs(r[i] + z[i]) > eps:
            lnz[i] = np.log(r[i] + z[i])
        else:
            lnz[i] = -np.log(r[i] - z[i])

    # ------------------------------------------------------
    # Magnetic field
    # ------------------------------------------------------

    hx = (
        - jx * np.arctan2(y * z, x * r)
        + jy * lnz
        + jz * lny
    )

    hy = (
        jx * lnz
        - jy * np.arctan2(x * z, y * r)
        + jz * lnx
    )

    hz = (
        jx * lny
        + jy * lnx
        - jz * np.arctan2(x * y, z * r)
    )

    h = np.empty((len(r), 3))

    h[:, 0] = hx
    h[:, 1] = hy
    h[:, 2] = hz

    return 100.0 * h


# ==========================================================
# Prism response
# ==========================================================

@njit
def prism(
    mgz,
    xobs,
    yobs,
    zobs,
    xsrc,
    ysrc,
    zsrc,
    dim
):
    """
    Rectangular prism magnetic field
    """

    dx = dim[0]
    dy = dim[1]
    dz = dim[2]

    xs1 = xsrc - dx[0]
    xs2 = xsrc + dx[1]

    ys1 = ysrc - dy[0]
    ys2 = ysrc + dy[1]

    zs1 = zsrc - dz[0]
    zs2 = zsrc + dz[1]

    x1 = xs1 - xobs
    x2 = xs2 - xobs

    y1 = ys1 - yobs
    y2 = ys2 - yobs

    z1 = zs1 - zobs
    z2 = zs2 - zobs

    h = np.zeros((xobs.shape[0], 3))

    h += _prism_kernel_func_(mgz, x1, y1, z1)
    h -= _prism_kernel_func_(mgz, x1, y1, z2)

    h -= _prism_kernel_func_(mgz, x1, y2, z1)
    h += _prism_kernel_func_(mgz, x1, y2, z2)

    h -= _prism_kernel_func_(mgz, x2, y1, z1)
    h += _prism_kernel_func_(mgz, x2, y1, z2)

    h += _prism_kernel_func_(mgz, x2, y2, z1)
    h -= _prism_kernel_func_(mgz, x2, y2, z2)

    return -h


# ==========================================================
# Total force anomaly
# ==========================================================

@njit
def total_force(exf, h):

    return (
        exf[0] * h[:, 0]
        + exf[1] * h[:, 1]
        + exf[2] * h[:, 2]
    )


# ==========================================================
# Kernel Matrix
# ==========================================================

@njit(parallel=True, fastmath=True)
def kernel_matrix3_extended(
    exf,
    mgz,
    xobs,
    yobs,
    zobs,
    xgrd,
    ygrd,
    zgrd,
    dim,
    L=1000.0
):
    """
    Kernel matrix for magnetic inversion

    Parameters
    ----------
    exf : (3,)
        Geomagnetic field unit vector

    mgz : (3,)
        Magnetization direction

    xobs,yobs,zobs : (ndata,)
        Observation locations

    xgrd,ygrd,zgrd : (ncell,)
        Cell centers

    dim : (ncell,3,2)

    L : float
        Boundary extension length
    """

    nobs = xobs.shape[0]
    ncell = xgrd.shape[0]

    xmin = np.min(xgrd)
    xmax = np.max(xgrd)

    ymin = np.min(ygrd)
    ymax = np.max(ygrd)

    K = np.zeros((nobs, ncell))

    for s in prange(ncell):

        dx1 = dim[s, 0, 0]
        dx2 = dim[s, 0, 1]

        dy1 = dim[s, 1, 0]
        dy2 = dim[s, 1, 1]

        dz1 = dim[s, 2, 0]
        dz2 = dim[s, 2, 1]

        # -------------------------
        # Boundary extension
        # -------------------------

        if abs(xgrd[s] - xmin) < 1e-10:
            dx1 += L

        elif abs(xgrd[s] - xmax) < 1e-10:
            dx2 += L

        if abs(ygrd[s] - ymin) < 1e-10:
            dy1 += L

        elif abs(ygrd[s] - ymax) < 1e-10:
            dy2 += L

        dim_new = np.empty((3, 2))

        dim_new[0, 0] = dx1
        dim_new[0, 1] = dx2

        dim_new[1, 0] = dy1
        dim_new[1, 1] = dy2

        dim_new[2, 0] = dz1
        dim_new[2, 1] = dz2

        h = prism(
            mgz,
            xobs,
            yobs,
            zobs,
            xgrd[s],
            ygrd[s],
            zgrd[s],
            dim_new
        )

        K[:, s] = total_force(exf, h)

    return K