import io
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Per‑point dV/dt & dI/dt – Quasi‑Steady‑State (IEC 60904‑1:2020)",
    layout="wide",
    page_icon="⚡"
)

# ===================== Utilities ===================== #
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
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    return med, 1.4826 * mad  # ≈ robust σ

def local_linear_slope(t, y, window_sec=0.3, min_points=5):
    """
    dy/dt via local linear regression in a TIME window centered at each sample.
    Works for irregular sampling. Returns NaN when insufficient points.
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    n = len(t)
    slopes = np.full(n, np.nan)

    order = np.argsort(t)
    ts, ys = t[order], y[order]

    for k in range(n):
        t0 = ts[k]
        L = np.searchsorted(ts, t0 - window_sec/2, side="left")
        R = np.searchsorted(ts, t0 + window_sec/2, side="right")
        if (R - L) >= min_points:
            tt = ts[L:R]
            yy = ys[L:R]
            tc = tt - tt.mean()
            try:
                a, b = np.polyfit(tc, yy, 1)  # yy ≈ a*tc + b
                slopes[k] = a
            except Exception:
                slopes[k] = np.nan
    inv = np.empty_like(order)
    inv[order] = np.arange(n)
    return slopes[inv]

def safe_gradient(y, t):
    y = np.asarray(y, float)
    t = np.asarray(t, float)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.gradient(y, t)

def effective_windows(t, window_sec, dwell_sec):
    """
    Clamp the local‑linear window & dwell to stay 'local' vs the record span.
    Prevents a global fit over the whole file on short traces.
    """
    t = np.asarray(t, float)
    if len(t) < 3:
        return float(window_sec), float(dwell_sec)
    ts = np.sort(t)
    dt = np.diff(ts)
    dt_med = float(np.median(dt)) if len(dt) else max(1e-6, float(window_sec)/50.0)
    span = float(np.ptp(ts)) if len(ts) else float(window_sec)

    # caps relative to record: allow at most 10% span for window, 20% for dwell
    max_window = max(5*dt_med, 0.10*span)
    max_dwell  = max(10*dt_med, 0.20*span)
    return min(float(window_sec), max_window), min(float(dwell_sec), max_dwell)

def dwell_pass_flags(t, dVdt, dIdt, thr_v, thr_i, dwell_sec=0.2):
    """
    IEC QSS rule: For each time t[i], both |dV/dt| and |dI/dt| must remain
    below thresholds THROUGHOUT the interval [t[i]-dwell_sec, t[i]].
    """
    t   = np.asarray(t, float)
    dV  = np.asarray(dVdt, float)
    dI  = np.asarray(dIdt, float)
    n   = len(t)
    out = np.zeros(n, dtype=bool)

    for i in range(n):
        left_t = t[i] - dwell_sec
        L = np.searchsorted(t, left_t, side="left")
        R = i + 1
        if R - L >= 2:
            ok_v = np.nanmax(np.abs(dV[L:R])) <= thr_v
            ok_i = np.nanmax(np.abs(dI[L:R])) <= thr_i
            out[i] = bool(ok_v and ok_i)
    return out

def detect_steps(V, t, min_step_V=0.25, min_step_gap=0.01):
    """
    Step starts by large |ΔV| and minimum time gap between starts.
    Always include index 0 as the first 'step'.
    """
    V = np.asarray(V, float)
    t = np.asarray(t, float)
    dv = np.diff(V, prepend=V[0])
    cand = np.where(np.abs(dv) >= float(min_step_V))[0]

    steps = [0]
    last_t = t[0]
    for idx in cand:
        if t[idx] - last_t >= float(min_step_gap):
            steps.append(int(idx))
            last_t = t[idx]
    return np.unique(np.array(steps, dtype=int))

def pick_first_qss_per_step(t, pass_flags, step_idx, min_separation_sec=0.0):
    """
    For each detected step start, pick the FIRST index at/after that start
    where QSS pass is True. Enforce minimal time spacing between picks.
    """
    t = np.asarray(t, float)
    n = len(t)
    picks = []
    last_t = -np.inf

    step_idx = np.asarray(step_idx, int)
    step_idx = step_idx[step_idx < n]
    for s in step_idx:
        i = s
        while i < n:
            if pass_flags[i] and (t[i] - last_t >= float(min_separation_sec)):
                picks.append(i)
                last_t = t[i]
                break
            i += 1
    return np.array(picks, dtype=int)

def suggest_thresholds(dVdt, dIdt, factor=3.0):
    """
    Suggest derivative thresholds from robust noise floors (MAD-based).
    """
    _, sig_v = robust_mad(dVdt[np.isfinite(dVdt)])
    _, sig_i = robust_mad(dIdt[np.isfinite(dIdt)])
    thr_v = factor * sig_v if (np.isfinite(sig_v) and (sig_v > 0)) else np.nan
    thr_i = factor * sig_i if (np.isfinite(sig_i) and (sig_i > 0)) else np.nan
    return thr_v, thr_i

# ===================== UI – Sidebar ===================== #
st.sidebar.header("Configuration")

# Data
demo = st.sidebar.toggle("Use synthetic demo data", value=True,
                         help="Generate a simple step‑ramp with transients.")
uploaded = st.sidebar.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"])

time_col_default, v_col_default, i_col_default = "time_s", "V_V", "I_A"

# Derivatives
st.sidebar.subheader("Derivative Estimation")
method = st.sidebar.selectbox("Method", ["Local linear (recommended)", "Gradient"], index=0)
window_sec = st.sidebar.number_input("Local linear window [s]", min_value=0.0001, max_value=2.0,
                                     value=0.25, step=0.0001, format="%.4f")
min_pts = st.sidebar.number_input("Min points per window", min_value=3, max_value=200, value=7, step=1)
smooth_win = st.sidebar.number_input("Smoothing window [samples] (for V & I BEFORE derivative)",
                                     min_value=1, max_value=501, value=5, step=2)
use_raw_for_deriv = st.sidebar.toggle("Use RAW V & I for derivatives (instead of smoothed)", value=False)

# QSS
st.sidebar.subheader("Quasi‑Steady‑State (QSS) Criteria")
dwell_sec = st.sidebar.number_input("Required dwell time [s]", min_value=0.0002, max_value=2.0,
                                    value=0.2, step=0.0001, format="%.4f")
thr_v_user = st.sidebar.text_input("Max |dV/dt| [V/s] (blank = auto)", value="")
thr_i_user = st.sidebar.text_input("Max |dI/dt| [A/s] (blank = auto)", value="")

# Steps
st.sidebar.subheader("Step Detection")
min_step_V  = st.sidebar.number_input("Min step size [V]", min_value=0.001, max_value=10.0, value=0.25, step=0.001)
min_step_gap = st.sidebar.number_input("Min time gap between steps [s]", min_value=0.0, max_value=1.0, value=0.02, step=0.001, format="%.3f")
min_sep_pick = st.sidebar.number_input("Min spacing between picked points [s]", min_value=0.0, max_value=1.0, value=0.01, step=0.001, format="%.3f")

# Export
want_export = st.sidebar.toggle("Enable CSV export", value=True)

# ===================== Load Data ===================== #
def generate_synthetic_step_data(n_steps=10, step_V=(0.0, 45.0), hold=0.02, tau=0.004, fs=10000,
                                 I0=10.0, G=0.18, C=0.02, noise_V=0.001, noise_I=0.02):
    t = np.arange(0, n_steps*hold, 1.0/fs)
    V_targets = np.linspace(step_V[0], step_V[1], n_steps)
    Vt = np.zeros_like(t)
    for k in range(n_steps):
        i0 = int(k*hold*fs)
        Vt[i0:] = V_targets[k]
    V = np.zeros_like(t)
    a = 1.0 - np.exp(-1.0/(fs*tau))
    for i in range(1, len(t)):
        V[i] = V[i-1] + a*(Vt[i] - V[i-1])
    dVdt_true = np.gradient(V, t)
    I = (I0 - G*V) + C*dVdt_true
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "time_s": t,
        "V_V": V + rng.normal(0, noise_V, size=len(t)),
        "I_A": I + rng.normal(0, noise_I, size=len(I))
    })

if demo and uploaded is None:
    df = generate_synthetic_step_data()
    st.success("Using synthetic demo data.")
else:
    if uploaded is None:
        st.info("Upload a file or enable demo data.")
        st.stop()
    try:
        if uploaded.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded, engine="openpyxl")
        st.success(f"Loaded: {uploaded.name}")
    except Exception as e:
        st.error(f"Failed to load file: {e}")
        st.stop()

# Column mapping
with st.expander("Map columns (if names differ)", expanded=False):
    cols = list(df.columns)
    time_col = st.selectbox("Time column [s]", cols, index=cols.index(time_col_default) if time_col_default in cols else 0)
    v_col    = st.selectbox("Voltage column [V]", cols, index=cols.index(v_col_default) if v_col_default in cols else 0)
    i_col    = st.selectbox("Current column [A]", cols, index=cols.index(i_col_default) if i_col_default in cols else 0)

df = df[[time_col, v_col, i_col]].rename(columns={time_col:"time_s", v_col:"V_V", i_col:"I_A"}).copy()
df = df.replace([np.inf, -np.inf], np.nan).dropna()
df = df.sort_values("time_s")
df = df[~df["time_s"].duplicated(keep="first")]
if len(df) < 10:
    st.error("Not enough points after cleaning (need ≥ 10).")
    st.stop()

# Optional smoothing
df["V_smooth"] = moving_average(df["V_V"].to_numpy(), smooth_win)
df["I_smooth"] = moving_average(df["I_A"].to_numpy(), smooth_win)

# ===================== Derivatives (per point) ===================== #
t = df["time_s"].to_numpy()
Vsrc = (df["V_V"] if use_raw_for_deriv else df["V_smooth"]).to_numpy()
Isrc = (df["I_A"] if use_raw_for_deriv else df["I_smooth"]).to_numpy()

# Keep local and enforce viable dwell on very short records
window_eff, dwell_eff = effective_windows(t, window_sec, dwell_sec)
if window_eff < window_sec:
    st.info(f"Local‑linear window clamped {window_sec:.6f}s → {window_eff:.6f}s "
            f"(record span {np.ptp(t):.6f}s).")
if dwell_eff < dwell_sec:
    st.info(f"Dwell clamped {dwell_sec:.6f}s → {dwell_eff:.6f}s to match record length.")

# Compute derivatives
if method.startswith("Local"):
    dVdt = local_linear_slope(t, Vsrc, window_sec=window_eff, min_points=int(min_pts))
    dIdt = local_linear_slope(t, Isrc, window_sec=window_eff, min_points=int(min_pts))
else:
    dVdt = safe_gradient(Vsrc, t)
    dIdt = safe_gradient(Isrc, t)

df["dVdt_Vps"] = dVdt
df["dIdt_Aps"] = dIdt

# ===================== Thresholds ===================== #
if thr_v_user.strip()=="" or thr_i_user.strip()=="":
    sv, si = suggest_thresholds(dVdt, dIdt, factor=3.0)
    if (not np.isfinite(sv)) or (sv <= 0):
        sv = np.nanmax([np.nanpercentile(np.abs(dVdt), 10), 1e-3])
    if (not np.isfinite(si)) or (si <= 0):
        si = np.nanmax([np.nanpercentile(np.abs(dIdt), 10), 1e-3])
    thr_v = float(thr_v_user) if thr_v_user.strip()!="" else float(sv)
    thr_i = float(thr_i_user) if thr_i_user.strip()!="" else float(si)
else:
    try:
        thr_v = float(thr_v_user)
        thr_i = float(thr_i_user)
    except Exception:
        st.error("Threshold inputs must be numeric.")
        st.stop()

# ===================== QSS Rule (IEC dwell) ===================== #
qss = dwell_pass_flags(t, dVdt, dIdt, thr_v=thr_v, thr_i=thr_i, dwell_sec=dwell_eff)
df["QSS_pass"] = qss

# ===================== Step detection & pick per‑step QSS point ===================== #
step_idx = detect_steps(df["V_V"].to_numpy(), t, min_step_V=min_step_V, min_step_gap=min_step_gap)
picks = pick_first_qss_per_step(t, qss, step_idx, min_separation_sec=min_sep_pick)
df["Recommended"] = False
if len(picks) > 0:
    df.loc[df.index[picks], "Recommended"] = True

# ===================== Power & dP/dt (informative) ===================== #
df["P_W"] = df["V_V"] * df["I_A"]
df["dPdt_Wps"] = df["I_A"]*df["dVdt_Vps"] + df["V_V"]*df["dIdt_Aps"]

# ===================== Summary ===================== #
c1, c2, c3, c4 = st.columns(4)
c1.metric("Samples", f"{len(df)}")
c2.metric("QSS pass (count)", f"{int(df['QSS_pass'].sum())}")
c3.metric("Thr |dV/dt| ≤ [V/s]", f"{thr_v:0.5f}")
c4.metric("Thr |dI/dt| ≤ [A/s]", f"{thr_i:0.5f}")

st.caption(
    "A point is QSS‑pass if both |dV/dt| and |dI/dt| remain below thresholds "
    f"throughout the preceding dwell (effective dwell = {dwell_eff:.6f} s)."
)

# ===================== Plots ===================== #
st.subheader("Time Series & Derivatives")

fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

# V and I
axs[0].plot(df["time_s"], df["V_V"], color="#1f77b4", lw=1.0, label="V [raw]")
axs[0].plot(df["time_s"], df["V_smooth"], color="#ff7f0e", lw=1.0, alpha=0.8, label="V [smooth]")
axs[0].set_ylabel("Voltage [V]"); axs[0].grid(True, alpha=0.3); axs[0].legend(loc="upper right")

axs[1].plot(df["time_s"], df["I_A"], color="#2ca02c", lw=1.0, label="I [raw]")
axs[1].plot(df["time_s"], df["I_smooth"], color="#d62728", lw=1.0, alpha=0.8, label="I [smooth]")
axs[1].set_ylabel("Current [A]"); axs[1].grid(True, alpha=0.3); axs[1].legend(loc="upper right")

# dV/dt and dI/dt
axs[2].plot(df["time_s"], df["dVdt_Vps"], color="#9467bd", lw=1.0, label="dV/dt")
axs[2].axhline(+thr_v, color="#9467bd", ls="--", alpha=0.5)
axs[2].axhline(-thr_v, color="#9467bd", ls="--", alpha=0.5)
axs[2].set_ylabel("dV/dt [V/s]"); axs[2].grid(True, alpha=0.3); axs[2].legend(loc="upper right")

axs[3].plot(df["time_s"], df["dIdt_Aps"], color="#8c564b", lw=1.0, label="dI/dt")
axs[3].axhline(+thr_i, color="#8c564b", ls="--", alpha=0.5)
axs[3].axhline(-thr_i, color="#8c564b", ls="--", alpha=0.5)
axs[3].set_ylabel("dI/dt [A/s]"); axs[3].set_xlabel("Time [s]")
axs[3].grid(True, alpha=0.3); axs[3].legend(loc="upper right")

# Mark per‑step recommended points
if len(picks) > 0:
    for ax in axs:
        ax.scatter(df["time_s"].iloc[picks], df[ax.get_ylabel().split()[0] if ax != axs[2] and ax != axs[3] else "dVdt_Vps"].iloc[picks] if ax==axs[2]
                   else df["dIdt_Aps"].iloc[picks] if ax==axs[3]
                   else (df["V_smooth"] if ax==axs[0] else df["I_smooth"]).iloc[picks],
                   color="k", s=35, marker="x", zorder=5)

st.pyplot(fig, use_container_width=True)

# IV with QSS status
st.subheader("IV with QSS status")
fig2, ax2 = plt.subplots(1,1, figsize=(6,5))
mask = df["QSS_pass"].to_numpy()
ax2.scatter(df["V_V"][~mask], df["I_A"][~mask], s=10, color="#ff9896", label="Not QSS")
ax2.scatter(df["V_V"][mask],  df["I_A"][mask],  s=10, color="#98df8a", label="QSS pass")
if len(picks) > 0:
    ax2.scatter(df["V_V"].iloc[picks], df["I_A"].iloc[picks], s=45, color="k", marker="x", label="Recommended")
ax2.set_xlabel("Voltage [V]"); ax2.set_ylabel("Current [A]")
ax2.grid(True, alpha=0.3); ax2.legend(loc="best")
st.pyplot(fig2, use_container_width=True)

# ===================== Tables & Export ===================== #
st.subheader("Per‑sample table (derivatives & QSS)")
st.dataframe(
    df[["time_s","V_V","I_A","P_W","dVdt_Vps","dIdt_Aps","dPdt_Wps","QSS_pass","Recommended"]].reset_index(drop=True),
    use_container_width=True, hide_index=True
)

if want_export:
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download full table (CSV)", data=csv_bytes,
                       file_name="qss_dvdt_didt_full.csv", mime="text/csv")

# ===================== IEC 60904‑1:2020 Validation Notes ===================== #
with st.expander("IEC 60904‑1:2020 – How this app validates the QSS requirement", expanded=False):
    st.markdown("""
**What IEC says (practical interpretation)**  
- Ideal steady‑state is **dV/dt = 0** and **dI/dt = 0** (capacitive current vanishes: **dQ/dt = 0**).  
- In practice, exact zero is not achievable → **use quasi‑steady‑state (QSS)**: both derivatives must be **sufficiently small** for a **sufficient time**.

**How this app enforces it**  
1) **Per‑point derivatives** (V/s, A/s) via **local linear regression** in a local time window (or gradient).  
2) **Dwell‑based QSS rule**: A sample passes only if **both |dV/dt| and |dI/dt| remain below thresholds** **throughout** the preceding dwell.  
3) **Step selection**: At each voltage step, the **first QSS‑pass** sample is the valid measurement (thus we 'wait sufficiently long').  
4) **Thresholds**: Auto‑suggested from the **noise floor** (MAD‑based, ≈ 3×σ), with option to manually tighten.  
5) **Short‑record protection**: The local window & dwell are auto‑clamped vs record span to avoid global fits and to keep the dwell meaningful.

**Evidence produced**  
- Per‑sample table with **dV/dt**, **dI/dt**, **dP/dt**, **QSS flags**, and **recommended** picks.  
- Time‑series plots with **thresholds** and per‑step recommended markers.  
- IV scatter colored by **QSS**.

This creates a defensible, audit‑ready workflow aligned with IEC 60904‑1:2020 for quasi‑steady‑state measurement.
""")
