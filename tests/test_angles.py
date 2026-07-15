from __future__ import annotations

import math

import pytest

from cricform.features.joint_angles import (
    absolute_angle_difference_degrees,
    angle_degrees,
    line_angle_degrees,
)


def test_angle_degrees_right_angle() -> None:
    angle = angle_degrees((1, 0), (0, 0), (0, 1))

    assert angle == pytest.approx(90.0)


def test_angle_degrees_straight_line() -> None:
    angle = angle_degrees((-1, 0), (0, 0), (1, 0))

    assert angle == pytest.approx(180.0)


def test_angle_degrees_returns_nan_for_zero_length_vector() -> None:
    angle = angle_degrees((0, 0), (0, 0), (1, 0))

    assert math.isnan(angle)


def test_line_angle_degrees_horizontal_line() -> None:
    angle = line_angle_degrees((0, 0), (1, 0))

    assert angle == pytest.approx(0.0)


def test_line_angle_degrees_vertical_line() -> None:
    angle = line_angle_degrees((0, 0), (0, 1))

    assert angle == pytest.approx(90.0)


def test_absolute_angle_difference_wraps_correctly() -> None:
    difference = absolute_angle_difference_degrees(350.0, 10.0)

    assert difference == pytest.approx(20.0)
