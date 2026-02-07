# streamlit_app.py
import io
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="dV/dt & dI/dt – Quasi-Steady-State (IEC 60904-1:2020)",
                   layout="wide",
                   page_icon="⚡")

# --------------------- Utilities --------------------- #
def moving_average(x, window):
    if window <= 1:
        return x
    window = int(window)
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window) / window
    x_pad = np.pad(x, (window//2, window//2), mode="edge")
    return np.convolve(x_pad, kernel, mode="valid")


def robust_mad(x):
    x = np.asarray(x)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    return med, 1.4826 * mad  # approx std from MAD


def local_linear_slope(t, y, window_sec=0.3, min_points=5):
    """
    Robust slope estimate dy/dt via local linear regression in a time window centered at each sample.
    Handles irregular sampling.
    Returns: slope array (same length as y); NaN where insufficient points.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(t)
    slopes = np.full(n, np.nan)
    # Use searchsorted on a sorted copy
    order = np.argsort(t)
    t_sorted = t[order]
    y_sorted = y[order]

    for idx_sorted in range(n):
        t0 = t_sorted[idx_sorted]
        left = np.searchsorted(t_sorted, t0 - window_sec/2, side="left")
        right = np.searchsorted(t_sorted, t0 + window_sec/2, side="right")
        count = right - left
        if count >= min_points:
            tt = t_sorted[left:right]
            yy = y_sorted[left:right]
            # Center times for numerical stability
            tt_c = tt - tt.mean()
            try:
                p = np.polyfit(tt_c, yy, deg=1)  # yy ≈ p[0]*tt_c + p[1]
                slopes[idx_sorted] = p[0]
            except Exception:
                slopes[idx_sorted] = np.nan
        else:
            slopes[idx_sorted] = np.nan

    # Map back to original order
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(n)
    return slopes[inv_order]


def safe_gradient(y, t):
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        dy = np.gradient(y, t)  # handles irregular spacing
    return dy


def detect_steps(V, min_step_V=0.25):
    """
    Return indices where a new step likely begins (simple |ΔV| threshold).
    """
    V = np.asarray(V)
    dv = np.diff(V, prepend=V[0])
    step_idx = np.where(np.abs(dv) >= min_step_V)[0]
    if len(step_idx) == 0 or step_idx[0] != 0:
        step_idx = np.insert(step_idx, 0, 0)
    return np.unique(step_idx)


def dwell_pass_flags(t, dVdt, dIdt, thr_v, thr_i, dwell_sec=0.2, method="last_window"):
    """
    True if for each sample i, both |dV/dt| <= thr_v and |dI/dt| <= thr_i
    for the entire interval [t[i]-dwell_sec, t[i]].
    """
    t = np.asarray(t)
    dVdt = np.asarray(dVdt)
    dIdt = np.asarray(dIdt)
    n = len(t)
    flags = np.zeros(n, dtype=bool)

    for i in range(n):
        t0 = t[i] - dwell_sec
        left = np.searchsorted(t, t0, side="left")
        right = i + 1
        if right - left < 2:
            flags[i] = False
            continue
        ok_v = np.nanmax(np.abs(dVdt[left:right])) <= thr_v
        ok_i = np.nanmax(np.abs(dIdt[left:right])) <= thr_i
        flags[i] = bool(ok_v and ok_i)
    return flags


def pick_recommended_points(t, V, I, pass_flags, step_idx, min_separation_sec=0.05):
    """
    For each detected step, choose the first time index where pass_flags is True,
    and ensure selections are not too close in time (min_separation_sec).
    """
    t = np.asarray(t)
    n = len(t)
    chosen = []
    last_time = -np.inf

    step_idx = np.asarray(step_idx)
    step_idx = step_idx[step_idx < n]

    for s in step_idx:
        i = s
        while i < n:
            if pass_flags[i] and (t[i] - last_time >= min_separation_sec):
                chosen.append(i)
                last_time = t[i]
                break
            i += 1
    return np.array(chosen, dtype=int)


def suggest_thresholds(dVdt, dIdt, factor=3.0):
    """
    Suggest thresholds based on robust noise floor (MAD-based).
    """
    _, sigma_v = robust_mad(dVdt[np.isfinite(dVdt)])
    _, sigma_i = robust_mad(dIdt[np.isfinite(dIdt)])
    thr_v = factor * sigma_v if np.isfinite(sigma_v) and sigma_v > 0 else np.nan
    thr_i = factor * sigma_i if np.isfinite(sigma_i) and sigma_i > 0 else np.nan
    return thr_v, thr_i


def generate_synthetic_step_data(
    n_steps=15, step_V=(0.0, 45.0), step_hold_s=0.5, tau_s=0.08,
    sample_rate_hz=200, I0_A=9.0, G_S=0.18, C_F=0.04, noise_V=0.001, noise_I=0.02
):
    """
    Simulate a step-ramped IV acquisition with capacitive transient.
    V(t): first-order response to target with time constant tau_s
    I(t) = I0 - G*V + C * dV/dt + noise
    """
    total_time = n_steps * step_hold_s
    dt = 1.0 / sample_rate_hz
    t = np.arange(0, total_time, dt)

    # Create target step sequence
    V_targets = np.linspace(step_V[0], step_V[1], n_steps)
    V_target_t = np.zeros_like(t)
    step_times = np.arange(0, total_time, step_hold_s)
    for k, ts in enumerate(step_times):
        idx0 = int(ts / dt)
        V_target_t[idx0:] = V_targets[k]

    # First-order RC response for V
    V = np.zeros_like(t)
    for i in range(1, len(t)):
        dV = (V_target_t[i] - V[i-1]) * (dt / tau_s)
        V[i] = V[i-1] + dV

    # Derivative dV/dt (true)
    dVdt_true = np.gradient(V, t)

    # Current with conductance and capacitive term
    I = (I0_A - G_S * V) + C_F * dVdt_true

    # Add noise
    rng = np.random.default_rng(42)
    Vn = V + rng.normal(0, noise_V, size=len(V))
    In = I + rng.normal(0, noise_I, size=len(I))

    return pd.DataFrame({"time_s": t, "V_V": Vn, "I_A": In})


# --------------------- UI – Sidebar --------------------- #
st.sidebar.header("Configuration")

st.sidebar.subheader("Data Input")
demo = st.sidebar.toggle("Use synthetic demo data", value=True, help="Generate a realistic step-ramp with capacitive transients.")
uploaded = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

time_col_default = "time_s"
v_col_default = "V_V"
