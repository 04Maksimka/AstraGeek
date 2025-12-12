"""Module with astronomical time functions."""

from datetime import datetime, time, timedelta
import numpy as np


def get_sidereal_time(longitude: float, local: datetime):
    """
    Calculate local sidereal time

    :param longitude: place longitude in radians
    :param local: local time
    :return: LST (Local Sidereal Time)
    """

    # Shift from UTC
    timeshift = get_timeshift(longitude)
    gmt = local

    # From 0 January
    year = gmt.year
    origin = datetime(year, 1, 1, 0, 0, 0) - timedelta(days=1)
    shift = int((gmt - origin).total_seconds() / 86400.0)

    # Calculate julian date on 0 January
    jd = julian_date(origin)
    T = (jd - 2415020.0) / 36525.0
    R = 6.6460656 + 2400.051262 * T + 0.00002581 * T**2
    U = R - 24 * (year - 1900)

    A = 0.0657098
    B = 24 - U
    C = 1.002738
    T0 = shift * A - B

    lst = (C * (get_total_hours(local.time())) + T0) % 24
    time_lst = get_time(lst)

    return time_lst


def vequinox_hour_angle(longitude: float, local: datetime) -> float:
    """
    Calculate vernal equinox hour angle
    :param longitude: place longitude in radians
    :param local: local time
    :return: hour angle in radians
    """

    sidereal_time = get_sidereal_time(longitude, local)
    t = np.deg2rad(get_total_hours(sidereal_time) * 15.0)
    return t


def get_total_hours(t: time) -> float:
    """
    Calculate time in hours (with floating point)
    :param t: time hh:mm:ss
    :return: hours
    """

    return t.hour + t.minute / 60.0 + t.second / 3600.0


def get_timeshift(longitude: float) -> timedelta:
    """
    Calculate timeshift between local time and UTC
    :param longitude: longitude in radians
    :return: shift
    """

    total_hours = np.rad2deg(longitude) / 15.0
    hours = int(total_hours % 24)
    minutes = int((total_hours - hours) * 60)
    seconds = int((total_hours - hours - minutes / 60.0) * 3600)
    return timedelta(hours=hours, minutes=minutes, seconds=seconds)


def get_time(total_hours: float) -> time:
    """
    Calculate time object from total hours (with floating point)
    :param total_hours: hours with floating point
    :return: corresponding time
    """

    hours = int(total_hours % 24)
    minutes = int((total_hours - hours) * 60)
    seconds = int((total_hours - hours - minutes / 60.0) * 3600)

    return time(hour=hours, minute=minutes, second=seconds)


def julian_date(date_time: datetime):
    """
    Returns the Julian date, number of days since 1 January 4713 BC 12:00 UTC.
    :param date_time: datetime object
    :return: Julian date
    """

    year = date_time.year
    month = date_time.month
    day = date_time.day

    if month > 2:
        y = year
        m = month
    else:
        y = year - 1
        m = month + 12

    d = day

    if year <= 1582 and month <= 10 and day <= 4:
        # Julian calendar
        b = 0
    else:
        # Gregorian Calendar
        a = int(y / 100)
        b = 2 - a + int(a / 4)

    jd = int(365.25*y) + int(30.6001*(m+1)) + d + b + 1720994.5

    return jd