from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

Point2D = Sequence[float | int]


def angle_degrees(point_a: Point2D, point_b: Point2D, point_c: Point2D) -> float:
    """Return the angle ABC in degrees.

    The angle is measured at point_b. Returns NaN if the angle cannot be
    computed because a point is missing, non-finite, or a vector has zero length.
    """

    a = _as_array(point_a)
    b = _as_array(point_b)
    c = _as_array(point_c)

    if not _is_valid_point(a) or not _is_valid_point(b) or not _is_valid_point(c):
        return math.nan

    vector_ba = a - b
    vector_bc = c - b

    norm_ba = float(np.linalg.norm(vector_ba))
    norm_bc = float(np.linalg.norm(vector_bc))

    if norm_ba == 0.0 or norm_bc == 0.0:
        return math.nan

    cosine = float(np.dot(vector_ba, vector_bc) / (norm_ba * norm_bc))
    cosine = float(np.clip(cosine, -1.0, 1.0))

    return float(np.degrees(np.arccos(cosine)))


def line_angle_degrees(point_a: Point2D, point_b: Point2D) -> float:
    """Return the 2D line angle from point_a to point_b in degrees."""

    a = _as_array(point_a)
    b = _as_array(point_b)

    if not _is_valid_point(a) or not _is_valid_point(b):
        return math.nan

    delta = b - a

    if float(np.linalg.norm(delta)) == 0.0:
        return math.nan

    return float(np.degrees(np.arctan2(delta[1], delta[0])))


def absolute_angle_difference_degrees(angle_a: float, angle_b: float) -> float:
    """Return the smallest absolute difference between two angles in degrees."""

    if not math.isfinite(angle_a) or not math.isfinite(angle_b):
        return math.nan

    difference = abs((angle_a - angle_b + 180.0) % 360.0 - 180.0)
    return float(difference)


def _as_array(point: Point2D) -> np.ndarray:
    return np.asarray(point, dtype=float)


def _is_valid_point(point: np.ndarray) -> bool:
    return point.shape == (2,) and bool(np.isfinite(point).all())
