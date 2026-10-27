import math
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

NPV_LOGISTIC_SCALE: float = 1e8
TARGET_IRR_FOR_EI: float = 0.30


def _is_positive_number(value: Any) -> bool:
    """Return True when the input is a finite number greater than zero.

    Parameters
    ----------
    value : Any
        Value to test for a positive numeric representation.

    Returns
    -------
    bool
        ``True`` if ``value`` can be converted to ``float`` and is strictly
        positive; otherwise ``False``.
    """
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return np.isfinite(numeric) and numeric > 0


def _is_non_negative_number(value: Any) -> bool:
    """Return True when the input is a finite number greater than or equal to zero.

    Parameters
    ----------
    value : Any
        Value to test for a non-negative numeric representation.

    Returns
    -------
    bool
        ``True`` if ``value`` can be converted to ``float`` and is not less than
        zero; otherwise ``False``.
    """
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return np.isfinite(numeric) and numeric >= 0


def npv(rate: float, cashflows: Sequence[float]) -> float:
    """
    Calculate the net present value of a series of cash flows.

    Parameters
    ----------
    rate : float
        Discount rate per period.
    cashflows : list of float
        Cash flows at each period, where `cashflows[0]` is time 0.

    Returns
    -------
    float
        Net present value.
    """
    if not math.isfinite(rate):
        raise ValueError("rate must be a finite number")
    if rate <= -1.0:
        raise ValueError("rate must be greater than -1 to compute NPV")

    cf_values = [float(cf) for cf in cashflows]
    if not cf_values:
        return 0.0

    total = cf_values[0]
    if len(cf_values) == 1:
        return total

    step = 1.0 / (1.0 + rate)
    discount_factor = 1.0
    for cf in cf_values[1:]:
        discount_factor *= step
        total += cf * discount_factor
    return total


def irr(
    cashflows: Sequence[float],
    guess: float = 0.1,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float | None:
    """
    Compute the internal rate of return for a series of cash flows.

    Uses the Newton–Raphson method to find the rate that zeroes NPV.

    Parameters
    ----------
    cashflows : list of float
        Cash flows at each period.
    guess : float, optional
        Initial guess for the IRR (default is 0.1).
    tol : float, optional
        Convergence tolerance (default is 1e-6).
    max_iter : int, optional
        Maximum number of iterations (default is 100).

    Returns
    -------
    float or None
        Estimated IRR if converged within `max_iter`, otherwise `None`.
    """
    cf_values = [float(cf) for cf in cashflows]
    if len(cf_values) < 2 or not any(cf > 0 for cf in cf_values) or not any(cf < 0 for cf in cf_values):
        return None

    def npv_and_derivative(rate: float) -> Tuple[float, float]:
        if not math.isfinite(rate) or rate <= -1.0:
            raise ValueError("rate must be greater than -1 for IRR computation")
        total = cf_values[0]
        derivative = 0.0
        if len(cf_values) == 1:
            return total, derivative
        step = 1.0 / (1.0 + rate)
        discount_factor = 1.0
        for t, cf in enumerate(cf_values[1:], start=1):
            discount_factor *= step
            total += cf * discount_factor
            derivative -= t * cf * discount_factor * step
        return total, derivative

    # Attempt Newton-Raphson first for fast convergence
    rate = guess
    for _ in range(max_iter):
        try:
            value, derivative = npv_and_derivative(rate)
        except ValueError:
            break
        if abs(value) < tol:
            return rate
        if abs(derivative) < 1e-12:
            break
        new_rate = rate - value / derivative
        if not math.isfinite(new_rate) or new_rate <= -0.999999:
            break
        if abs(new_rate - rate) < tol:
            return new_rate
        rate = new_rate

    # Fallback to bisection to guarantee convergence if IRR exists
    def npv_only(r: float) -> float:
        try:
            return npv(r, cf_values)
        except ValueError:
            return math.copysign(math.inf, cf_values[0])

    lower = -0.999999
    upper = max(rate, 0.1)
    value_lower = npv_only(lower)
    value_upper = npv_only(upper)

    if abs(value_lower) < tol:
        return lower
    if abs(value_upper) < tol:
        return upper

    def _sign(val: float) -> int:
        if math.isnan(val):
            return 0
        if abs(val) < tol:
            return 0
        return 1 if val > 0 else -1

    expansion_attempts = 0
    while _sign(value_lower) == _sign(value_upper) and expansion_attempts < 60:
        upper = upper * 2.0 + 1.0
        value_upper = npv_only(upper)
        expansion_attempts += 1
        if not math.isfinite(value_upper):
            break

    if _sign(value_lower) == _sign(value_upper):
        return None

    for _ in range(200):
        mid = (lower + upper) / 2.0
        value_mid = npv_only(mid)
        if not math.isfinite(value_mid):
            return None
        if abs(value_mid) < tol:
            return mid
        sign_mid = _sign(value_mid)
        if sign_mid == 0:
            return mid
        if sign_mid == _sign(value_lower):
            lower = mid
            value_lower = value_mid
        else:
            upper = mid
            value_upper = value_mid

    if abs(value_mid) < 1e-4:
        return mid
    return None


def payback_period(rate: float, cashflows: Sequence[float]) -> float | None:
    """
    Calculate the discounted payback period for cash flows.

    The payback period is the time when cumulative discounted cash flows
    become non-negative.

    Parameters
    ----------
    rate : float
        Discount rate per period.
    cashflows : list of float
        Cash flows at each period.

    Returns
    -------
    float or None
        Discounted payback period in periods (may be fractional), or `None`
        if the investment is never recovered.
    """
    if not math.isfinite(rate):
        raise ValueError("rate must be a finite number")
    if rate <= -1.0:
        raise ValueError("rate must be greater than -1 to compute payback")

    cum = 0.0
    for t, cf in enumerate(cashflows):
        discounted = float(cf) / (1 + rate) ** t
        prev = cum
        cum += discounted
        if cum >= 0:
            if discounted == 0:
                return float(t)
            return float(t) - (prev / discounted)
    return None


def quantize(value: float | None, places: str = "0.01") -> Decimal | None:
    """
    Quantize a float to a fixed number of decimal places.

    Parameters
    ----------
    value : float or None
        Value to quantize.
    places : str, optional
        Decimal quantization format, e.g. "0.01" for two decimal places.
        Default is "0.01".

    Returns
    -------
    Decimal or None
        Quantized `Decimal` value, or `None` if `value` is `None`.
    """
    if value is None:
        return None
    return Decimal(value).quantize(Decimal(places), rounding=ROUND_HALF_UP)


def make_cashflow(
    lu: str,
    land_area: float,
    profile: Dict[str, Any]
) -> List[float]:
    """
    Generate a series of cash flows for a land-use profile.

    Parameters
    ----------
    lu : str
        Land-use profile key.
    land_area : float
        Land area in the same units used by `profile["density"]`.
    profile : dict
        Profile parameters containing at least:
        - "density": float
        - "cost_build": float
        Optionally:
        - "land_cost": float
        - "construction_years": int
        - "opex_rate": float
        And either:
        - "price_sale": float and "sale_years": int
        or
        - "rent_annual": float, "rent_years": int, "occupancy": float

    Returns
    -------
    list of float
        Cash flow list: initial negative outflows for land and construction,
        followed by net operating revenues or sales.

    Raises
    ------
    ValueError
        If `profile` lacks both "price_sale" and "rent_annual".
    """
    # Determine the effective gross floor area (GFA).
    land_area = float(land_area)
    if not np.isfinite(land_area) or land_area <= 0:
        raise ValueError(f"Profile '{lu}' requires a positive land_area")

    built_area_value = profile.get("built_area")
    gfa = float(built_area_value) if _is_positive_number(built_area_value) else math.nan
    if not np.isfinite(gfa) or gfa <= 0:
        density = profile.get("density")
        if _is_positive_number(density):
            gfa = land_area * float(density)
    if not np.isfinite(gfa) or gfa <= 0:
        gfa = max(land_area, 0.0)

    cost_build_value = profile.get("cost_build", 0)
    cost_build = float(cost_build_value) if _is_non_negative_number(cost_build_value) else 0.0

    land_cost_raw = profile.get("land_cost", 0)
    land_cost_per_area = float(land_cost_raw) if _is_non_negative_number(land_cost_raw) else 0.0
    land_cost = land_area * land_cost_per_area

    years_build_raw = profile.get("construction_years", 2)
    years_build = int(years_build_raw) if _is_positive_number(years_build_raw) else 2

    build_cost = gfa * cost_build
    capex_per_year = build_cost / years_build if years_build else 0.0

    cf: List[float] = [-land_cost - capex_per_year] + [-capex_per_year] * (years_build - 1)

    opex_rate_raw = profile.get("opex_rate", 0)
    opex_rate = float(opex_rate_raw) if _is_non_negative_number(opex_rate_raw) else 0.0
    opex = opex_rate * gfa
    if "price_sale" in profile:
        yrs_raw = profile.get("sale_years", 3)
        yrs = int(yrs_raw) if _is_positive_number(yrs_raw) else 3
        price_sale_raw = profile.get("price_sale", 0)
        price_sale = float(price_sale_raw) if _is_non_negative_number(price_sale_raw) else 0.0
        rev_total = gfa * price_sale
        rev_per_year = rev_total / yrs
        cf.extend(rev_per_year - opex for _ in range(yrs))
    elif "rent_annual" in profile:
        yrs_raw = profile.get("rent_years", 10)
        yrs = int(yrs_raw) if _is_positive_number(yrs_raw) else 10
        share_raw = profile.get("rent_share", 1.0)
        share = float(share_raw) if _is_non_negative_number(share_raw) else 1.0
        share = max(0.0, min(1.0, share))
        rent_annual_raw = profile.get("rent_annual", 0)
        rent_annual = float(rent_annual_raw) if _is_non_negative_number(rent_annual_raw) else 0.0
        rev_per_year = gfa * rent_annual * share
        cf.extend(rev_per_year - opex for _ in range(yrs))
    else:
        raise ValueError(f"Profile '{lu}' needs price_sale or rent_annual")

    return cf


def nanminmax(values: Iterable[float]) -> Tuple[float, float]:
    """
    Compute the minimum and maximum of values, ignoring NaNs.

    Parameters
    ----------
    values : iterable of float
        Input values, may contain NaNs.

    Returns
    -------
    (float, float)
        Tuple `(min, max)` ignoring NaNs.
    """
    arr = np.array(list(values), dtype=float)
    if arr.size == 0:
        return math.nan, math.nan
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return math.nan, math.nan
    return float(finite.min()), float(finite.max())


def normalize_series(
    s: Sequence[float],
    vmin: float,
    vmax: float
) -> pd.Series:
    """
    Normalize a sequence of values to a 0-100 scale.

    Parameters
    ----------
    s : sequence of float
        Input values.
    vmin : float
        Minimum value for normalization.
    vmax : float
        Maximum value for normalization.

    Returns
    -------
    pandas.Series
        Normalized values scaled to [0, 100], or zeros if `vmin == vmax`.
    """
    # Always preserve the input index to keep alignment with original data.
    ss = pd.Series(s, dtype=float)
    if vmax > vmin:
        return 100 * (ss - vmin) / (vmax - vmin)
    # Degenerate range: return zeros aligned to the original index
    return pd.Series(0.0, index=ss.index)


def economic_index(
    npv_val: float | None,
    irr_val: float | None,
    rate: float
) -> float:
    """
    Compute a bounded economic index based on NPV and IRR.

    Combines normalized NPV and IRR contributions into a score between 0 and 100.

    Parameters
    ----------
    npv_val : float or None
        Net present value.
    irr_val : float or None
        Internal rate of return.
    rate : float
        Discount rate for IRR performance normalization.

    Returns
    -------
    float
        Economic index in [0, 100], rounded to four decimal places.
    """
    ei = 0.0
    if npv_val is not None and math.isfinite(npv_val):
        arg = -npv_val / NPV_LOGISTIC_SCALE
        if arg > 700:
            sig = 0.0
        elif arg < -700:
            sig = 1.0
        else:
            sig = 1.0 / (1.0 + math.exp(arg))
        npv_component = 50.0 * max(0.0, 2.0 * sig - 1.0)
        ei += npv_component

    if irr_val is not None and math.isfinite(irr_val):
        if rate >= TARGET_IRR_FOR_EI:
            irr_component = 50.0 if irr_val > rate else 0.0
        else:
            irr_component = 50.0 * max(0.0, irr_val - rate) / (TARGET_IRR_FOR_EI - rate)
        ei += irr_component

    return round(min(max(ei, 0.0), 100.0), 4)
