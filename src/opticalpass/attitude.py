"""Quaternions, the LVLH frame and nadir-pointing geometry.

Quaternion convention
---------------------
* Scalar first: q = [w, x, y, z] with w = cos(theta/2), [x, y, z] = axis sin(theta/2).
* Hamilton product (i^2 = j^2 = k^2 = ijk = -1).
* ``rotate(q, v)`` is an ACTIVE rotation: it returns q v q*, the vector v turned
  by theta about the axis, with components in the same frame.
* ``quat_to_dcm(q)`` returns the matrix R with R @ v == rotate(q, v).
* Frame reading: if q_AB is the active rotation that turns frame A's axes onto
  frame B's axes, then R(q_AB) converts components from B to A:
  v_A = R(q_AB) @ v_B. This is the convention stated in every docstring below
  that names two frames.

Vectorised: arrays of quaternions have shape (..., 4), vectors (..., 3).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

_EZ = np.array([0.0, 0.0, 1.0])


def _q(q: ArrayLike) -> NDArray[np.float64]:
    a = np.asarray(q, dtype=float)
    if a.shape[-1] != 4:
        raise ValueError(f"Quaternion array must have last dimension 4, got shape {a.shape}")
    return a


def quat_normalize(q: ArrayLike) -> NDArray[np.float64]:
    a = _q(q)
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    if np.any(n < 1e-12):
        raise ValueError("Cannot normalise a zero quaternion")
    return a / n


def quat_conjugate(q: ArrayLike) -> NDArray[np.float64]:
    a = _q(q).copy()
    a[..., 1:] *= -1.0
    return a


def quat_multiply(p: ArrayLike, q: ArrayLike) -> NDArray[np.float64]:
    """Hamilton product p * q. Applying the result rotates by q first, then p."""
    a, b = _q(p), _q(q)
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=-1)


def quat_from_axis_angle(axis: ArrayLike, angle_rad: ArrayLike) -> NDArray[np.float64]:
    ax = np.asarray(axis, dtype=float)
    n = np.linalg.norm(ax, axis=-1, keepdims=True)
    if np.any(n < 1e-15):
        raise ValueError("Rotation axis must be non-zero")
    half = 0.5 * np.asarray(angle_rad, dtype=float)[..., None]
    return np.concatenate([np.cos(half), np.sin(half) * ax / n], axis=-1)


def rotate(q: ArrayLike, v: ArrayLike) -> NDArray[np.float64]:
    """Active rotation q v q* of vector(s) v."""
    qn = quat_normalize(q)
    vv = np.asarray(v, dtype=float)
    vq = np.concatenate([np.zeros(vv.shape[:-1] + (1,)), vv], axis=-1)
    return quat_multiply(quat_multiply(qn, vq), quat_conjugate(qn))[..., 1:]


def quat_to_dcm(q: ArrayLike) -> NDArray[np.float64]:
    """Rotation matrix R (shape (..., 3, 3)) such that R @ v equals rotate(q, v)."""
    w, x, y, z = np.moveaxis(quat_normalize(q), -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], axis=-2)


def dcm_to_quat(R: ArrayLike) -> NDArray[np.float64]:
    """Inverse of quat_to_dcm, using Shepperd's method for numerical stability.

    Picks the largest of w, x, y, z to divide by, so no branch ever divides by a
    small number. Returns w >= 0 (q and -q are the same rotation).
    """
    M = np.asarray(R, dtype=float)
    if M.shape[-2:] != (3, 3):
        raise ValueError(f"DCM must have shape (..., 3, 3), got {M.shape}")
    det = np.linalg.det(M)
    orth = np.max(np.abs(M @ np.swapaxes(M, -1, -2) - np.eye(3)))
    if np.any(np.abs(det - 1.0) > 1e-6) or orth > 1e-6:
        raise ValueError("Matrix is not a proper rotation (orthonormal, det +1)")
    flat = M.reshape(-1, 3, 3)
    out = np.empty((flat.shape[0], 4))
    for k, m in enumerate(flat):
        tr = np.trace(m)
        cand = np.array([tr, m[0, 0], m[1, 1], m[2, 2]])
        i = int(np.argmax(cand))
        if i == 0:
            w = 0.5 * np.sqrt(1.0 + tr)
            q = [w, (m[2, 1] - m[1, 2]) / (4 * w), (m[0, 2] - m[2, 0]) / (4 * w),
                 (m[1, 0] - m[0, 1]) / (4 * w)]
        elif i == 1:
            x = 0.5 * np.sqrt(1.0 + 2 * m[0, 0] - tr)
            q = [(m[2, 1] - m[1, 2]) / (4 * x), x, (m[0, 1] + m[1, 0]) / (4 * x),
                 (m[0, 2] + m[2, 0]) / (4 * x)]
        elif i == 2:
            y = 0.5 * np.sqrt(1.0 + 2 * m[1, 1] - tr)
            q = [(m[0, 2] - m[2, 0]) / (4 * y), (m[0, 1] + m[1, 0]) / (4 * y), y,
                 (m[1, 2] + m[2, 1]) / (4 * y)]
        else:
            z = 0.5 * np.sqrt(1.0 + 2 * m[2, 2] - tr)
            q = [(m[1, 0] - m[0, 1]) / (4 * z), (m[0, 2] + m[2, 0]) / (4 * z),
                 (m[1, 2] + m[2, 1]) / (4 * z), z]
        q = np.asarray(q)
        out[k] = q if q[0] >= 0 else -q
    return quat_normalize(out.reshape(M.shape[:-2] + (4,)))


# ------------------------------------------------------------------------------ LVLH


def lvlh_dcm(r_m: ArrayLike, v_inertial_m_s: ArrayLike) -> NDArray[np.float64]:
    """Matrix whose columns are the LVLH axes expressed in the input frame.

    +Z towards nadir (-r), +Y opposite the orbit normal (-r x v), +X = Y x Z,
    which is close to the velocity direction for a near-circular orbit.
    v must be inertial, otherwise the orbit normal is wrong by the Earth's spin.
    Returns M with v_frame = M @ v_lvlh, so v_lvlh = M.T @ v_frame.
    """
    r = np.asarray(r_m, dtype=float)
    v = np.asarray(v_inertial_m_s, dtype=float)
    h = np.cross(r, v)
    rn = np.linalg.norm(r, axis=-1, keepdims=True)
    hn = np.linalg.norm(h, axis=-1, keepdims=True)
    if np.any(rn == 0) or np.any(hn == 0):
        raise ValueError("LVLH undefined: zero position or position parallel to velocity")
    z = -r / rn
    y = -h / hn
    x = np.cross(y, z)
    return np.stack([x, y, z], axis=-1)


def to_body(vec_frame: ArrayLike, r_m: ArrayLike, v_inertial_m_s: ArrayLike
            ) -> NDArray[np.float64]:
    """Express vector(s) in the body frame, assuming body axes equal LVLH axes."""
    M = lvlh_dcm(r_m, v_inertial_m_s)
    return np.einsum("...ji,...j->...i", M, np.asarray(vec_frame, dtype=float))


def off_nadir_angle_rad(los_body: ArrayLike) -> NDArray[np.float64]:
    """Angle between body +Z (nadir boresight) and the line of sight [rad]."""
    u = np.asarray(los_body, dtype=float)
    u = u / np.linalg.norm(u, axis=-1, keepdims=True)
    return np.arccos(np.clip(u[..., 2], -1.0, 1.0))


def shortest_slew_quat(los_body: ArrayLike) -> NDArray[np.float64]:
    """Quaternion of the smallest rotation that turns body +Z onto ``los_body``.

    Active rotation in body axes: rotate(q, [0, 0, 1]) == unit(los_body). The axis
    is z x los and the angle is the off-nadir angle. At zero angle the result is
    the identity; at 180 deg any perpendicular axis works and body +X is used.
    """
    u = np.asarray(los_body, dtype=float)
    u = u / np.linalg.norm(u, axis=-1, keepdims=True)
    axis = np.cross(_EZ, u)
    s = np.linalg.norm(axis, axis=-1)
    angle = np.arccos(np.clip(u[..., 2], -1.0, 1.0))
    safe_axis = np.where((s > 1e-12)[..., None], axis, np.array([1.0, 0.0, 0.0]))
    return quat_from_axis_angle(safe_axis, angle)
