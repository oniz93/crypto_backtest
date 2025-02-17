"""
utils.py
--------
Utility functions for common tasks used throughout the project,
including date conversion, unit conversion, rounding, and normalization.
"""

from datetime import datetime, timezone
import numpy as np

def convert_date_to_time(date_str):
    """
    Convert a date string ('YYYY-MM-DD HH:MM:SS') to a datetime object.

    Parameters:
        date_str (str): Date string.

    Returns:
        datetime: Parsed datetime object.
    """
    return datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')

def sat_to_usd(sat, price):
    """
    Convert satoshis to USD given the Bitcoin price.

    Parameters:
        sat: Amount in satoshis.
        price: Price per Bitcoin.

    Returns:
        float: Value in USD.
    """
    return (price / 100000000) * sat

def usd_to_sat(usd, price):
    """
    Convert USD to satoshis given the Bitcoin price.

    Parameters:
        usd: Amount in USD.
        price: Price per Bitcoin.

    Returns:
        float: Equivalent satoshis.
    """
    return (100000000 / price) * usd

def round_to_100(qty):
    """
    Round a quantity to the nearest 100 units.

    Parameters:
        qty (float): Input quantity.

    Returns:
        float: Rounded quantity.
    """
    if qty < 100:
        return 100.0
    else:
        qty = round(qty, 0)
        remainder = qty % 100
        if remainder >= 50:
            qty = qty - remainder + 100
        else:
            qty = qty - remainder
    return qty

def get_next_rebuy_pct(step, n_rebuy, pct_stop_loss):
    """
    Calculate the next rebuy percentage.

    Parameters:
        step (int): Current rebuy step.
        n_rebuy (int): Total number of rebuy steps.
        pct_stop_loss (float): Stop-loss percentage.

    Returns:
        float: Next rebuy percentage.
    """
    x = (n_rebuy * pct_stop_loss) / (n_rebuy + 1)
    s = n_rebuy - (n_rebuy - step)
    return s * x / n_rebuy

def protected_div(left, right):
    """
    Perform division with protection against division by zero.

    Parameters:
        left: Numerator.
        right: Denominator.

    Returns:
        float: Division result, or 1 if denominator is zero.
    """
    try:
        return left / right
    except ZeroDivisionError:
        return 1

def timestamp_to_datetime(timestamp_ms):
    """
    Convert a timestamp in milliseconds to a UTC datetime object.

    Parameters:
        timestamp_ms (int): Timestamp in milliseconds.

    Returns:
        datetime: UTC datetime.
    """
    return datetime.fromtimestamp(timestamp_ms / 1000, timezone.utc)

def normalize_price(price, max_price=150000):
    """
    Normalize a price to the range [-1, 1].

    Parameters:
        price (float): Price value.
        max_price (float): Expected maximum price.

    Returns:
        float: Normalized price.
    """
    norm = 2 * (price / max_price) - 1
    return max(min(norm, 1), -1)

def normalize_volume(volume, max_volume=1000000):
    """
    Normalize a volume value to the range [-1, 1].

    Parameters:
        volume (float): Volume value.
        max_volume (float): Expected maximum volume.

    Returns:
        float: Normalized volume.
    """
    norm = 2 * (volume / max_volume) - 1
    return max(min(norm, 1), -1)

def normalize_diff(diff, max_diff=150000):
    """
    Normalize a difference value to the range [-1, 1].

    Parameters:
        diff (float): Difference value.
        max_diff (float): Expected maximum absolute difference.

    Returns:
        float: Normalized difference.
    """
    norm = diff / max_diff
    return max(min(norm, 1), -1)

def normalize_rsi(rsi):
    """
    Normalize RSI (0-100) to range [-1, 1] (with 50 -> 0).

    Parameters:
        rsi (float): RSI value.

    Returns:
        float: Normalized RSI.
    """
    return (rsi / 50) - 1

def normalize_price_vec(prices, max_price=150000):
    """
    Vectorized normalization for price values.

    Parameters:
        prices (np.ndarray): Array of prices.
        max_price (float): Maximum price value.

    Returns:
        np.ndarray: Normalized prices.
    """
    norm = 2 * (prices / max_price) - 1
    return np.clip(norm, -1, 1)

def normalize_volume_vec(volumes, max_volume=1000000):
    """
    Vectorized normalization for volume values.

    Parameters:
        volumes (np.ndarray): Array of volumes.
        max_volume (float): Maximum volume.

    Returns:
        np.ndarray: Normalized volumes.
    """
    norm = 2 * (volumes / max_volume) - 1
    return np.clip(norm, -1, 1)

def normalize_diff_vec(diffs, max_diff=150000):
    """
    Vectorized normalization for difference values.

    Parameters:
        diffs (np.ndarray): Array of differences.
        max_diff (float): Maximum absolute difference.

    Returns:
        np.ndarray: Normalized differences.
    """
    norm = diffs / max_diff
    return np.clip(norm, -1, 1)

def normalize_rsi_vec(rsis):
    """
    Vectorized normalization for RSI values.

    Parameters:
        rsis (np.ndarray): Array of RSI values.

    Returns:
        np.ndarray: Normalized RSI values.
    """
    return (rsis / 50) - 1
