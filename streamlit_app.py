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
                slopes[idx_sorted] = p[0]        # dy/dt
            except Exception:
                slopes[idx_sorted] = np.nan
        else:
            slopes[idx_sorted] = np.nan

    # Map back to original order
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(n)
    slopes_original = slopes[inv_order]
    return slopes_original


def safe_gradient(y, t):
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    # gradient with respect to time (handles irregular spacing)
    with np.errstate(divide='ignore', invalid='ignore'):
        dy = np.gradient(y, t)
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
    Returns boolean array indicating that *both* |dV/dt| <= thr_v and |dI/dt| <= thr_i
    for at least dwell_sec immediately before (and including) each sample.
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
        # From step start to end, pick first pass
        i = s
        found = False
        while i < n:
            if pass_flags[i] and (t[i] - last_time >= min_separation_sec):
                chosen.append(i)
                last_time = t[i]
                found = True
                break
            i += 1
        # If not found, skip
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

    Model:
      V(t) follows a first-order response to step target with time constant tau_s
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
i_col_default = "I_A"

# Derivative controls
st.sidebar.subheader("Derivative Estimation")
method = st.sidebar.selectbox("Method", ["Local linear (recommended)", "Gradient"], index=0,
                              help="Local linear fit in a time window is robust to noise and irregular sampling.")
window_sec = st.sidebar.number_input("Local linear window [s]", min_value=0.02, max_value=2.0, value=0.25, step=0.01)
min_pts = st.sidebar.number_input("Min points per window", min_value=3, max_value=100, value=7, step=1)
smooth_win = st.sidebar.number_input("Smoothing window [samples] (applied to V & I before derivative)", min_value=1, max_value=501, value=5, step=2)

st.sidebar.subheader("Quasi-Steady-State Criteria")
dwell_sec = st.sidebar.number_input("Required dwell time [s]", min_value=0.05, max_value=2.0, value=0.2, step=0.05)
thr_v_user = st.sidebar.text_input("Max |dV/dt| [V/s] (leave blank to auto-suggest)", value="")
thr_i_user = st.sidebar.text_input("Max |dI/dt| [A/s] (leave blank to auto-suggest)", value="")

st.sidebar.subheader("Step Detection")
min_step_V = st.sidebar.number_input("Minimum step size [V] to flag a step", min_value=0.01, max_value=5.0, value=0.25, step=0.01)
min_sep = st.sidebar.number_input("Min separation between picks [s]", min_value=0.01, max_value=1.0, value=0.05, step=0.01)

# Near-MPP band controls
st.sidebar.subheader("Near‑MPP Reporting")
near_pct = st.sidebar.number_input("Near‑MPP band [% of Pmax]", min_value=0.1, max_value=10.0, value=2.0, step=0.1,
                                   help="Show points with P >= (1 - band%) * Pmax, from the selected candidate set.")
near_max_rows = st.sidebar.number_input("Max rows in near‑MPP table", min_value=5, max_value=200, value=30, step=5)

st.sidebar.subheader("Export")
want_export = st.sidebar.toggle("Enable CSV export", value=True)

# --------------------- Load Data --------------------- #
if demo and uploaded is None:
    df = generate_synthetic_step_data()
    st.success("Using synthetic demo data.")
else:
    if uploaded is not None:
        try:
            if uploaded.name.lower().endswith(".csv"):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded, engine="openpyxl")
            st.success(f"Loaded file: {uploaded.name}")
        except Exception as e:
            st.error(f"Failed to load file: {e}")
            st.stop()
    else:
        st.info("Upload a file or enable synthetic demo data.")
        st.stop()

# Column selectors
with st.expander("Map columns (if names differ)", expanded=False):
    cols = list(df.columns)
    time_col = st.selectbox("Time column [s]", cols, index=cols.index(time_col_default) if time_col_default in cols else 0)
    v_col = st.selectbox("Voltage column [V]", cols, index=cols.index(v_col_default) if v_col_default in cols else 0)
    i_col = st.selectbox("Current column [A]", cols, index=cols.index(i_col_default) if i_col_default in cols else 0)

# Clean and sort
df = df[[time_col, v_col, i_col]].rename(columns={time_col: "time_s", v_col: "V_V", i_col: "I_A"}).copy()
df = df.replace([np.inf, -np.inf], np.nan).dropna()
df = df.sort_values("time_s")
df = df[~df["time_s"].duplicated(keep="first")]

if len(df) < 10:
    st.error("Not enough data points after cleaning. Need at least 10.")
    st.stop()

# Optional smoothing
df["V_smooth"] = moving_average(df["V_V"].to_numpy(), smooth_win)
df["I_smooth"] = moving_average(df["I_A"].to_numpy(), smooth_win)

# Derivatives
t = df["time_s"].to_numpy()
V = df["V_smooth"].to_numpy()
I = df["I_smooth"].to_numpy()

if method.startswith("Local"):
    dVdt = local_linear_slope(t, V, window_sec=float(window_sec), min_points=int(min_pts))
    dIdt = local_linear_slope(t, I, window_sec=float(window_sec), min_points=int(min_pts))
else:
    dVdt = safe_gradient(V, t)
    dIdt = safe_gradient(I, t)

df["dVdt_Vps"] = dVdt
df["dIdt_Aps"] = dIdt

# Suggest thresholds if user left blank
if thr_v_user.strip() == "" or thr_i_user.strip() == "":
    sv, si = suggest_thresholds(dVdt, dIdt, factor=3.0)
    if np.isnan(sv) or sv <= 0:
        sv = np.nanmax([np.nanpercentile(np.abs(dVdt), 10), 1e-3])
    if np.isnan(si) or si <= 0:
        si = np.nanmax([np.nanpercentile(np.abs(dIdt), 10), 1e-3])
    thr_v = float(thr_v_user) if thr_v_user.strip() != "" else float(sv)
    thr_i = float(thr_i_user) if thr_i_user.strip() != "" else float(si)
else:
    try:
        thr_v = float(thr_v_user)
        thr_i = float(thr_i_user)
    except Exception:
        st.error("Threshold inputs must be numeric.")
        st.stop()

# Pass/fail flags based on dwell (base thresholds)
pass_flags = dwell_pass_flags(t, dVdt, dIdt, thr_v=thr_v, thr_i=thr_i, dwell_sec=float(dwell_sec), method="last_window")
df["QSS_pass"] = pass_flags

# Step detection and recommended picks
step_idx = detect_steps(df["V_V"].to_numpy(), min_step_V=float(min_step_V))
picks = pick_recommended_points(t, V, I, pass_flags, step_idx, min_separation_sec=float(min_sep))
df["Recommended"] = False
if len(picks) > 0:
    df.loc[df.index[picks], "Recommended"] = True

# --------------------- Power & MPP determination (QSS-only enforced) --------------------- #
# Power per sample from raw V and I (not smoothed)
df["P_W"] = df["V_V"] * df["I_A"]

# Candidates are ONLY the samples that pass the base QSS criterion (dwell-based).
qss_candidates = df.index[df["QSS_pass"]]
if len(qss_candidates) == 0:
    st.error(
        "No quasi-steady-state (QSS) samples found under current thresholds/dwell. "
        "Relax thresholds or increase dwell to obtain QSS-pass data points."
    )
    st.stop()

cand_idx = qss_candidates
mpp_source = "QSS_pass (enforced)"

# Index of the MPP row within df (max power among QSS-only candidates)
mpp_idx = df.loc[cand_idx, "P_W"].idxmax()
df["is_MPP"] = False
df.at[mpp_idx, "is_MPP"] = True

# Values at MPP
Vpmax = float(df.at[mpp_idx, "V_V"])
Ipmax = float(df.at[mpp_idx, "I_A"])
Pmax  = float(df.at[mpp_idx, "P_W"])
dVdt_mpp = float(df.at[mpp_idx, "dVdt_Vps"])
dIdt_mpp = float(df.at[mpp_idx, "dIdt_Aps"])
t_mpp = float(df.at[mpp_idx, "time_s"])
qss_mpp = True  # by construction (picked only from QSS_pass)
# Informative power derivative at MPP
dPdt_mpp = Ipmax * dVdt_mpp + Vpmax * dIdt_mpp

# Near‑MPP band (within near_pct% of Pmax, restricted to same QSS-only candidate set)
near_threshold = (1.0 - near_pct / 100.0) * Pmax
cand_df = df.loc[cand_idx].copy()
near_df = cand_df[cand_df["P_W"] >= near_threshold].copy()
near_df["delta_P_W"] = Pmax - near_df["P_W"]
near_df = near_df.sort_values(["delta_P_W", "time_s"]).head(int(near_max_rows))

# --------------------- Summary --------------------- #
c1, c2, c3, c4 = st.columns(4)
c1.metric("Samples", f"{len(df)}")
c2.metric("QSS Pass (count)", f"{int(df['QSS_pass'].sum())}")
c3.metric("Suggested |dV/dt| ≤ [V/s]", f"{thr_v:0.5f}")
c4.metric("Suggested |dI/dt| ≤ [A/s]", f"{thr_i:0.5f}")

st.caption("QSS = Quasi-Steady-State; a sample passes if both |dV/dt| and |dI/dt| remain below thresholds for the last dwell window.")

# --------------------- MPP Block --------------------- #
st.subheader("Maximum Power Point (VPmax, IPmax, and derivatives at MPP)  —  QSS-only enforced")
mc1, mc2, mc3, mc4 = st.columns(4)
mc1.metric("Pmax [W]", f"{Pmax:0.3f}")
mc2.metric("Vpmax [V]", f"{Vpmax:0.3f}")
mc3.metric("Ipmax [A]", f"{Ipmax:0.3f}")
mc4.metric("dP/dt @MPP [W/s]", f"{dPdt_mpp:0.5f}")

mc5, mc6, mc7, mc8 = st.columns(4)
mc5.metric("dV/dt @MPP [V/s]", f"{dVdt_mpp:0.5e}")
mc6.metric("dI/dt @MPP [A/s]", f"{dIdt_mpp:0.5e}")
mc7.metric("QSS @MPP", "PASS ✅" if qss_mpp else "FAIL ❌")
mc8.metric("MPP selected from", mpp_source)

with st.expander("Near‑MPP points (showing dV/dt & dI/dt near Pmax)", expanded=True):
    show_cols = ["time_s", "V_V", "I_A", "P_W", "dVdt_Vps", "dIdt_Aps", "QSS_pass"]
    st.dataframe(near_df[show_cols].reset_index(drop=True),
                 use_container_width=True, hide_index=True)

# --------------------- Plots --------------------- #
st.subheader("Time Series")

fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
axs[0].plot(df["time_s"], df["V_V"], color="#1f77b4", label="V [raw]")
axs[0].plot(df["time_s"], df["V_smooth"], color="#ff7f0e", alpha=0.8, label="V [smooth]")
axs[0].set_ylabel("Voltage [V]")
axs[0].legend(loc="upper right")
axs[0].grid(True, alpha=0.3)

axs[1].plot(df["time_s"], df["I_A"], color="#2ca02c", label="I [raw]")
axs[1].plot(df["time_s"], df["I_smooth"], color="#d62728", alpha=0.8, label="I [smooth]")
axs[1].set_ylabel("Current [A]")
axs[1].legend(loc="upper right")
axs[1].grid(True, alpha=0.3)

axs[2].plot(df["time_s"], df["dVdt_Vps"], color="#9467bd", label="dV/dt")
axs[2].plot(df["time_s"], df["dIdt_Aps"], color="#8c564b", label="dI/dt")
axs[2].axhline(+thr_v, color="#9467bd", linestyle="--", alpha=0.5)
axs[2].axhline(-thr_v, color="#9467bd", linestyle="--", alpha=0.5)
axs[2].axhline(+thr_i, color="#8c564b", linestyle="--", alpha=0.5)
axs[2].axhline(-thr_i, color="#8c564b", linestyle="--", alpha=0.5)
axs[2].set_ylabel("Derivatives [V/s, A/s]")
axs[2].set_xlabel("Time [s]")
axs[2].legend(loc="upper right")
axs[2].grid(True, alpha=0.3)

# Mark recommended points (x)
if len(picks) > 0:
    axs[0].scatter(df["time_s"].iloc[picks], df["V_smooth"].iloc[picks], color="k", marker="x", s=40, label="Recommended")
    axs[1].scatter(df["time_s"].iloc[picks], df["I_smooth"].iloc[picks], color="k", marker="x", s=40)
    axs[2].scatter(df["time_s"].iloc[picks], df["dVdt_Vps"].iloc[picks], color="k", marker="x", s=40)
    axs[2].scatter(df["time_s"].iloc[picks], df["dIdt_Aps"].iloc[picks], color="k", marker="x", s=40)

# Highlight MPP (vertical line on all, star markers)
for ax in axs:
    ax.axvline(t_mpp, color="red", linestyle=":", alpha=0.6)
axs[0].scatter([t_mpp], [Vpmax], color="red", s=60, marker="*", zorder=5, label="MPP")
axs[1].scatter([t_mpp], [Ipmax], color="red", s=60, marker="*", zorder=5)
axs[2].scatter([t_mpp], [dVdt_mpp], color="red", s=60, marker="*", zorder=5)
axs[2].scatter([t_mpp], [dIdt_mpp], color="red", s=60, marker="*", zorder=5)

st.pyplot(fig, use_container_width=True)

# IV scatter with pass/fail
st.subheader("IV Curve with QSS Status")
fig2, ax2 = plt.subplots(1, 1, figsize=(6, 5))
mask_pass = df["QSS_pass"].to_numpy()
ax2.scatter(df["V_V"][~mask_pass], df["I_A"][~mask_pass], s=12, color="#ff9896", label="Not QSS")
ax2.scatter(df["V_V"][mask_pass], df["I_A"][mask_pass], s=12, color="#98df8a", label="QSS pass")
if len(picks) > 0:
    ax2.scatter(df["V_V"].iloc[picks], df["I_A"].iloc[picks], s=45, color="k", marker="x", label="Recommended")
# MPP star
ax2.scatter([Vpmax], [Ipmax], s=80, color="red", marker="*", label="MPP")
# Near‑MPP points (small gray markers), restricted to QSS-only candidates
if len(near_df) > 0:
    ax2.scatter(near_df["V_V"], near_df["I_A"], s=18, color="gray", alpha=0.6, label=f"Near‑MPP (≥{100-near_pct:.1f}% Pmax)")
ax2.set_xlabel("Voltage [V]")
ax2.set_ylabel("Current [A]")
ax2.grid(True, alpha=0.3)
ax2.legend(loc="best")
st.pyplot(fig2, use_container_width=True)

# --------------------- Recommended points table --------------------- #
st.subheader("Recommended Measurement Points (per step)")
if len(picks) > 0:
    out_cols = ["time_s", "V_V", "I_A", "P_W", "dVdt_Vps", "dIdt_Aps", "QSS_pass", "Recommended", "is_MPP"]
    table = df.iloc[picks][out_cols].reset_index(drop=True)
    st.dataframe(table, use_container_width=True, hide_index=True)
else:
    st.info("No recommended points found under the current thresholds/dwell time. Consider relaxing thresholds or increasing dwell.")

# --------------------- Export --------------------- #
if want_export:
    # Full QC export
    export_cols = ["time_s", "V_V", "I_A", "P_W", "V_smooth", "I_smooth",
                   "dVdt_Vps", "dIdt_Aps", "QSS_pass", "Recommended", "is_MPP"]
    export_df = df[export_cols].copy()
    export_df.attrs["notes"] = (
        f"Method={method}, window_sec={window_sec}, min_points={min_pts}, "
        f"smooth_win={smooth_win}, dwell_sec={dwell_sec}, thr_v={thr_v}, thr_i={thr_i}, "
        f"min_step_V={min_step_V}, MPP_source={mpp_source}, near_pct={near_pct}"
    )
    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download QC Report (CSV)",
        data=csv_bytes,
        file_name="qss_dvdt_didt_report.csv",
        mime="text/csv"
    )

    # Near‑MPP export
    if len(near_df) > 0:
        near_cols = ["time_s", "V_V", "I_A", "P_W", "dVdt_Vps", "dIdt_Aps", "QSS_pass", "delta_P_W"]
        near_csv = near_df[near_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"⬇️ Download Near‑MPP Table (≥{100-near_pct:.1f}% Pmax)",
            data=near_csv,
            file_name="near_mpp_points.csv",
            mime="text/csv"
        )

# --------------------- Guidance --------------------- #
with st.expander("Notes & Guidance", expanded=False):
    st.markdown(
        """
**What this app does**

- Computes **dV/dt** and **dI/dt** per sample from time-series IV acquisition.
- Applies a **quasi-steady-state (QSS)** check using dwell-based derivative thresholds.
- Detects steps and proposes the **first QSS-compliant** point per step as **Recommended**.
- Computes **power** per sample (**P = V×I**) and determines **Pmax**.
- **QSS-only enforced:** MPP and near‑MPP are selected **only** from **QSS-pass** samples.

**Tips**

- Use the **Local linear** derivative estimator; set the window a bit **shorter than dwell**.
- For high‑capacitance modules, consider larger dwell or tighter thresholds; **dP/dt at MPP** should be **very small** if QSS is satisfied.
        """
    )
