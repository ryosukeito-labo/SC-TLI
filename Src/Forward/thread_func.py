import numpy as np
import magcal

def thread_func(p0, p1, mgz,
                xobs, yobs, zobs,
                xgrd, ygrd, zgrd,
                dim, L):
    """
    Compute the forward kernel (sensitivity) matrix for 3D magnetic inversion.

    This function builds a partial kernel matrix K of size (N x M_partial),
    where each column represents the response of one rectangular prism cell.

    The computation is typically parallelized along the z-direction.

    Parameters
    ----------
    p0, p1 : int
        Index range along the z-direction (for parallel processing).

    mgz : float or ndarray
        Magnetization (scalar or vector depending on implementation).

    xobs, yobs, zobs : ndarray of shape (N,)
        Observation point coordinates.

    xgrd, ygrd, zgrd : ndarray of shape (M,)
        Center coordinates of model cells.

    dim : list
        Cell dimensions:
            [[xmin_offset, xmax_offset],
             [ymin_offset, ymax_offset],
             [zmin_offset, zmax_offset]]

    L : float
        Extension length applied to boundary cells to mitigate edge effects.

    Returns
    -------
    K : ndarray of shape (N, M_partial)
        Partial forward kernel matrix.
    """

    # ----------------------------------------
    # Grid size
    # ----------------------------------------
    x_unique = np.unique(xgrd)
    y_unique = np.unique(ygrd)

    nx = len(x_unique)
    ny = len(y_unique)

    # ----------------------------------------
    # Observation size
    # ----------------------------------------
    N = len(xobs)

    # ----------------------------------------
    # Number of cells in this thread
    # ----------------------------------------
    nz_local = int(p1 - p0)
    M_partial = nx * ny * nz_local

    # Preallocate kernel matrix (IMPORTANT)
    K = np.zeros((N, M_partial))

    col = 0  # column index

    for iz in range(int(p0), int(p1)):
        for iy in range(ny):
            for ix in range(nx):

                # ----------------------------------------
                # Flattened cell index
                # ----------------------------------------
                j = ix + iy * nx + iz * nx * ny

                # ----------------------------------------
                # Copy original dimensions
                # ----------------------------------------
                dim0_0, dim0_1 = dim[0]
                dim1_0, dim1_1 = dim[1]
                dim2_0, dim2_1 = dim[2]

                # ----------------------------------------
                # Extend boundary cells (edge effect mitigation)
                # ----------------------------------------
                if ix == 0:
                    dim0_0 += L
                elif ix == nx - 1:
                    dim0_1 += L

                if iy == 0:
                    dim1_0 += L
                elif iy == ny - 1:
                    dim1_1 += L

                dim_new = [
                    [dim0_0, dim0_1],
                    [dim1_0, dim1_1],
                    [dim2_0, dim2_1]
                ]

                # ----------------------------------------
                # Forward modeling (prism response)
                # ----------------------------------------
                c = magcal.prism(
                    mgz,
                    xobs, yobs, zobs,
                    xgrd[j], ygrd[j], zgrd[j],
                    dim_new
                )

                f = magcal.total_force(mgz, c).reshape(-1)

                # ----------------------------------------
                # Store into kernel matrix
                # ----------------------------------------
                K[:, col] = f
                col += 1

                # ----------------------------------------
                # Progress logging
                # ----------------------------------------
                if j % 1000 == 0:
                    print(f"Processed cell index: {j}")

    return K