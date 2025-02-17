# src/utils.py

from datetime import datetime, timezone
import numpy as np

def convert_date_to_time(date_str):
    """Converts a date string to a datetime object."""
    return datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')


def sat_to_usd(sat, price):
    """Converts satoshis to USD based on the price."""
    return (price / 100000000) * sat


def usd_to_sat(usd, price):
    """Converts USD to satoshis based on the price."""
    return (100000000 / price) * usd


def round_to_100(qty):
    """Rounds a quantity to the nearest 100 units."""
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
    """Calculates the next rebuy percentage."""
    x = (n_rebuy * pct_stop_loss) / (n_rebuy + 1)
    s = n_rebuy - (n_rebuy - step)
    return s * x / n_rebuy


def protected_div(left, right):
    """Performs a protected division to avoid ZeroDivisionError."""
    try:
        return left / right
    except ZeroDivisionError:
        return 1

def timestamp_to_datetime(timestamp_ms):
    """Converts a timestamp in milliseconds to a datetime object in UTC."""
    return datetime.fromtimestamp(timestamp_ms / 1000, timezone.utc)

def normalize_price(price, max_price=150000):
    """
    Normalize a price value to the range [-1, 1] given a maximum price.
    0 -> -1, max_price -> 1.
    """
    norm = 2 * (price / max_price) - 1
    return max(min(norm, 1), -1)

def normalize_volume(volume, max_volume=1000000):
    """
    Normalize a volume value to the range [-1, 1] given a chosen maximum volume.
    """
    norm = 2 * (volume / max_volume) - 1
    return max(min(norm, 1), -1)

def normalize_diff(diff, max_diff=150000):
    """
    Normalize a difference (which may be negative) to the range [-1, 1]
    assuming the expected maximum absolute difference is max_diff.
    """
    norm = diff / max_diff
    return max(min(norm, 1), -1)

def normalize_rsi(rsi):
    """
    Normalize RSI (normally in 0-100) so that 50 -> 0, 0 -> -1 and 100 -> 1.
    """
    return (rsi / 50) - 1

def normalize_price_vec(prices, max_price=150000):
    norm = 2 * (prices / max_price) - 1
    return np.clip(norm, -1, 1)

def normalize_volume_vec(volumes, max_volume=1000000):
    norm = 2 * (volumes / max_volume) - 1
    return np.clip(norm, -1, 1)

def normalize_diff_vec(diffs, max_diff=150000):
    norm = diffs / max_diff
    return np.clip(norm, -1, 1)

def normalize_rsi_vec(rsis):
    return (rsis / 50) - 1