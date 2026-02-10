# streamlit_app.py
# Formula-based dV/dt & dI/dt per point with IEC 60904-1:2020 QSS enforcement
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Formula-based dV/dt & dI/dt per point – IEC 60904-1:2020 QSS",
    layout="wide",
    page_icon="⚡"
)

# ============================ Utilities ============================ #
def robust_mad(x):
    """Return (median, 1.4826*MAD) as a robust σ estimate."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    return med, 1.4826 * mad

def suggest_thresholds(dVdt, dIdt, factor=3.0):
    """Auto-suggest derivative thresholds from noise floor (MAD-based)."""
    _, sig_v = robust_mad(dVdt[np.isfinite(dVdt)])
    _, sig_i = robust_mad(dIdt[np.isfinite(dIdt)])
    thr_v = factor * sig_v if (np.isfinite(sig_v) and sig_v > 0) else np.nan
    thr_i = factor * sig_i if (np.isfinite(sig_i) and sig_i > 0) else np.nan
    return thr_v, thr_i

def finite_diff(y, t, mode="backward"):
    """
    Formula-based dy/dt per point with irregular t support.
    mode:
      - 'backward': dy/dt[i] = (y[i]-y[i-1])/(t[i]-t[i-1]); dy/dt[0] = NaN
      - 'forward' : dy/dt[i] = (y[i+1]-y[i])/(t[i+1]-t[i]); dy/dt[-1] = NaN
      - 'central' : dy/dt[i] = (y[i+1]-y[i-1])/(t[i+1]-t[i-1]) for 1..n-2;
                    ends use backward/forward
    NaN where Δt == 0 (division by zero avoided).
    """
    y = np.asarray(y, float)
    t = np.asarray(t, float)
    n = len(y)
    out = np.full(n, np.nan)

    if n < 2:
        return out

    if mode == "backward":
        dt = t[1:] - t[:-1]
        dy = y[1:] - y[:-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            out[1:] = np.where(dt != 0, dy / dt, np.nan)

    elif mode == "forward":
        dt = t[1:] - t[:-1]
        dy = y[1:] - y[:-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            out[:-1] = np.where(dt != 0, dy / dt, np.nan)

    else:  # central
        if n >= 3:
            dtc = t[2:] - t[:-2]
            dyc = y[2:] - y[:-2]
            with np.errstate(divide="ignore", invalid="ignore"):
                out[1:-1] = np.where(dtc != 0, dyc / dtc, np.nan)
        # end-point fallbacks:
        dt0 = t[1] - t[0]
        out[0] = (y[1] - y[0]) / dt0 if dt0 != 0 else np.nan
        dtn = t[-1] - t[-2]
        out[-1] = (y[-1] - y[-2]) / dtn if dtn != 0 else np.nan

    return out

def dwell_pass_flags(t, dVdt, dIdt, thr_v, thr_i, dwell_sec):
    """
    IEC QSS rule:
    For each sample i, both |dV/dt| and |dI/dt| must remain <= thresholds
    throughout the interval [t[i]-dwell_sec, t[i]].
    """
    t = np.asarray(t, float)
    dv = np.asarray(dVdt, float)
    di = np.asarray(dIdt, float)
    n = len(t)
    out = np.zeros(n, dtype=bool)
    for i in range(n):
        L = np.searchsorted(t, t[i] - dwell_sec, side="left")
        R = i + 1
        if R - L >= 2:  # need at least two points in dwell
            ok_v = np.nanmax(np.abs(dv[L:R])) <= thr_v
            ok_i = np.nanmax(np.abs(di[L:R])) <= thr_i
            out[i] = bool(ok_v and ok_i)
    return out

def detect_steps(V, t, min_step_V=0.25, min_step_gap=0.01):
    """
    Step detection by |ΔV|, ensuring minimal time between step starts.
    Always include index 0 as the first step.
    """
    V = np.asarray(V, float)
    t = np.asarray(t, float)
    dv = np.diff(V, prepend=V[0])
    cand = np.where(np.abs(dv) >= float(min_step_V))[0]
    steps = [0]
    last = t[0] if len(t) else 0.0
    for idx in cand:
        if t[idx] - last >= float(min_step_gap):
            steps.append(int(idx))
            last = t[idx]
    return np.unique(np.array(steps, dtype=int))

def pick_first_qss_per_step(t, qss_flags, step_starts, min_spacing_sec=0.0):
    """For each step start, pick FIRST index at/after start where QSS is True, with minimal time spacing enforced."""
    t = np.asarray(t, float)
    picks, last_t = [], -np.inf
    for s in step_starts:
        i = int(s)
        while i < len(t):
            if qss_flags[i] and (t[i] - last_t >= min_spacing_sec):
                picks.append(i)
                last_t = t[i]
                break
            i += 1
    return np.array(picks, dtype=int)

# ============================ Sidebar ============================ #
st.sidebar.header("Configuration")

st.sidebar.subheader("Data Input")
uploaded = st.sidebar.file_uploader("Upload CSV (tab/comma) or Excel", type=["csv", "xlsx"])
demo = st.sidebar.toggle("Use a small demo snippet (if no file uploaded)", value=(uploaded is None))

st.sidebar.subheader("Column Mapping")
time_default = "TimeOffset"  # your table header
v_default    = "Vraw"        # your table header
i_default    = "Iraw"        # your table header
p_default    = "Praw"        # optional

st.sidebar.subheader("Derivative Scheme")
scheme_label = st.sidebar.selectbox(
    "Finite difference formula",
    ["Backward (V[i]-V[i-1])/Δt", "Central (V[i+1]-V[i-1])/Δt", "Forward (V[i+1]-V[i])/Δt"],
    index=0
)
scheme_map = {
    "Backward (V[i]-V[i-1])/Δt": "backward",
    "Central (V[i+1]-V[i-1])/Δt": "central",
    "Forward (V[i+1]-V[i])/Δt": "forward",
}
scheme = scheme_map[scheme_label]

st.sidebar.subheader("QSS (IEC 60904-1:2020)")
dwell_sec = st.sidebar.number_input("Dwell time [s] (look-back window)", min_value=0.0001, max_value=2.0, value=0.0020, step=0.0001, format="%.4f")
thr_v_user = st.sidebar.text_input("Max |dV/dt| [V/s] (leave blank = auto)", value="")
thr_i_user = st.sidebar.text_input("Max |dI/dt| [A/s] (leave blank = auto)", value="")

st.sidebar.subheader("Step Detection")
min_step_V  = st.sidebar.number_input("Min step size [V]", min_value=0.001, max_value=10.0, value=0.25, step=0.001)
min_step_gap = st.sidebar.number_input("Min time gap between steps [s]", min_value=0.0, max_value=1.0, value=0.001, step=0.0005, format="%.4f")
min_pick_gap = st.sidebar.number_input("Min spacing between picked points [s]", min_value=0.0, max_value=1.0, value=0.0005, step=0.0005, format="%.4f")

want_export = st.sidebar.toggle("Enable CSV export", value=True)

# ============================ Load Data ============================ #
if demo:
    demo_text = """TimeOffset\tVimp\tIraw\tVraw\tPraw\tdV/dt\tdI/dt
0\t-0.81\t11.6009\t7.2096\t83.63784864\t-949\t-806
0.0001\t-2.7\t12.4615\t5.5231\t68.82611065\t-527\t-275
0.0002\t-4.41\t12.9625\t4.0053\t51.91870125\t-1054\t-177
0.0003\t-6.3\t13.1786\t2.1186\t27.92018196\t-738\t-20
0.0004\t-8.1\t13.2867\t0.3795\t5.04230265\t-738\t-118
0.0005\t-8.1406\t13.3142\t-0.4111\t-5.47346762\t-210\t39
0.0006\t-8.1406\t13.324\t-0.4427\t-5.8985348\t-105\t39
0.0007\t-8.1406\t13.3358\t-0.4427\t-5.90375866\t-105\t-98
0.0008\t-8.1406\t13.3456\t-0.4427\t-5.90809712\t0\t-138
0.0009\t-8.1406\t13.3653\t-0.4216\t-5.63481048\t0\t0
0.001\t-8.1406\t13.3692\t-0.4216\t-5.63645472\t-211\t-39
0.0011\t-8.1406\t13.3731\t-0.4216\t-5.63809896\t0\t-40
0.0012\t-8.1406\t13.3751\t-0.4005\t-5.35672755\t0\t0
0.0013\t-8.1406\t13.381\t-0.4111\t-5.5009291\t0\t-78
0.0014\t-8.1406\t13.381\t-0.4216\t-5.6414296\t-105\t20
0.0015\t-8.1406\t13.381\t-0.4111\t-5.5009291\t-105\t0
0.0016\t-8.1406\t13.3888\t-0.4111\t-5.50413568\t106\t19
0.0017\t-5.4369\t13.3692\t-0.3794\t-5.07227448\t1159\t216
0.0018\t-5.4369\t13.3653\t2.2873\t30.57045069\t105\t-39
0.0019\t-5.4369\t13.3771\t2.34\t31.302414\t-106\t-19
0.002\t-5.4369\t13.3731\t2.34\t31.293054\t-106\t-40
0.0021\t32.0036\t13.1983\t38.8727\t513.0535564\t1159\t649
0.0022\t32.0036\t13.1432\t39.8845\t524.2099604\t0\t117
0.0023\t32.0036\t13.1118\t40.0532\t525.1695478\t-211\t0
0.0024\t32.0036\t13.1197\t40.0848\t525.9005506\t-106\t0
0.0025\t32.0036\t13.1295\t40.0637\t526.0163492\t-316\t0
0.0026\t32.0036\t13.1413\t39.9899\t525.5192729\t106\t39
0.0027\t32.0036\t13.1491\t39.9478\t525.277617\t-317\t0
0.0028\t33.869\t12.3593\t41.4656\t512.4857901\t737\t-393
0.0029\t33.869\t12.9173\t41.5921\t537.2576333\t-106\t59
"""
    from io import StringIO
    df = pd.read_csv(StringIO(demo_text), sep="\t")
else:
    if uploaded is None:
        st.info("Upload a file or enable demo.")
        st.stop()
    try:
        if uploaded.name.lower().endswith(".csv"):
            # auto-detect separator
            try:
                df = pd.read_csv(uploaded, sep=None, engine="python")
            except Exception:
                df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded, engine="openpyxl")
    except Exception as e:
        st.error(f"Failed to load file: {e}")
        st.stop()

# ============================ Column Mapping (FIXED) ============================ #
with st.expander("Map columns", expanded=True):
    cols = list(df.columns)
    time_col = st.selectbox("Time column [s]", cols, index=cols.index(time_default) if time_default in cols else 0)
    v_col    = st.selectbox("Voltage column [V]", cols, index=cols.index(v_default) if v_default in cols else 1)
    i_col    = st.selectbox("Current column [A]", cols, index=cols.index(i_default) if i_default in cols else 2)
    p_col    = st.selectbox("Power column [W] (optional)", ["<none>"] + cols, index=(cols.index(p_default)+1 if p_default in cols else 0))

# Build selection list
select_cols = [time_col, v_col, i_col] + ([p_col] if p_col != "<none>" else [])
df = df[select_cols].copy()

# Safe renaming (DO NOT touch time unless mapping)
rename_map = {time_col: "time_s", v_col: "V", i_col: "I"}
if p_col != "<none>":
    rename_map[p_col] = "P"
df = df.rename(columns=rename_map).copy()

# Coerce numerics robustly
for c in ["time_s", "V", "I"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
if p_col != "<none>":
    df["P"] = pd.to_numeric(df["P"], errors="coerce")

# Drop non-numeric rows in required columns
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["time_s", "V", "I"])
if p_col == "<none>":
    df["P"] = df["V"] * df["I"]
else:
    df = df.dropna(subset=["P"])

# Enforce strictly increasing time & de-duplicate timestamps
df = df.sort_values("time_s")
before = len(df)
df = df[~df["time_s"].duplicated(keep="first")]
after = len(df)
if after < before:
    st.warning(f"Removed {before-after} duplicated time rows.")

# Δt inspector
if len(df) < 3:
    st.error("Need at least 3 time points after cleaning.")
    st.stop()
dt = np.diff(df["time_s"].to_numpy())
if np.any(dt <= 0):
    st.warning("Found non-increasing (or zero) time steps; those segments cannot produce valid derivatives.")
st.caption(f"Δt stats — min: {np.min(dt):.6e} s, median: {np.median(dt):.6e} s, max: {np.max(dt):.6e} s")

# ============================ Derivatives per point (your formula) ============================ #
t = df["time_s"].to_numpy().astype(float)
V = df["V"].to_numpy().astype(float)
I = df["I"].to_numpy().astype(float)

dVdt = finite_diff(V, t, mode=scheme)
dIdt = finite_diff(I, t, mode=scheme)

df["dVdt_Vps"] = dVdt
df["dIdt_Aps"] = dIdt

# ============================ Thresholds ============================ #
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

# ============================ QSS (IEC dwell rule) ============================ #
qss = dwell_pass_flags(t, dVdt, dIdt, thr_v=thr_v, thr_i=thr_i, dwell_sec=float(dwell_sec))
df["QSS_pass"] = qss

# ============================ Step detection & per-step pick ============================ #
step_starts = detect_steps(df["V"].to_numpy(), t, min_step_V=float(min_step_V), min_step_gap=float(min_step_gap))
picks = pick_first_qss_per_step(t, qss, step_starts, min_spacing_sec=float(min_pick_gap))
df["Recommended"] = False
if len(picks) > 0:
    df.loc[df.index[picks], "Recommended"] = True

# ============================ Summary ============================ #
c1, c2, c3, c4 = st.columns(4)
c1.metric("Samples", f"{len(df)}")
c2.metric("QSS pass (count)", f"{int(df['QSS_pass'].sum())}")
c3.metric("|dV/dt| ≤ [V/s]", f"{thr_v:0.5f}")
c4.metric("|dI/dt| ≤ [A/s]", f"{thr_i:0.5f}")

st.caption(
    "Per‑point derivatives are computed by finite differences. "
    "A sample passes QSS only if both |dV/dt| and |dI/dt| "
    f"remain ≤ thresholds throughout the preceding dwell of {dwell_sec:.4f} s."
)

# ============================ Plots ============================ #
st.subheader("Derivatives vs Time (with thresholds)")
fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

ax[0].plot(df["time_s"], df["dVdt_Vps"], color="#9467bd", lw=1.0, label="dV/dt")
ax[0].axhline(+thr_v, color="#9467bd", ls="--", alpha=0.5)
ax[0].axhline(-thr_v, color="#9467bd", ls="--", alpha=0.5)
ax[0].set_ylabel("dV/dt [V/s]")
ax[0].grid(True, alpha=0.3); ax[0].legend(loc="upper right")

ax[1].plot(df["time_s"], df["dIdt_Aps"], color="#8c564b", lw=1.0, label="dI/dt")
ax[1].axhline(+thr_i, color="#8c564b", ls="--", alpha=0.5)
ax[1].axhline(-thr_i, color="#8c564b", ls="--", alpha=0.5)
ax[1].set_ylabel("dI/dt [A/s]"); ax[1].set_xlabel("Time [s]")
ax[1].grid(True, alpha=0.3); ax[1].legend(loc="upper right")

# Mark recommended picks
if len(picks) > 0:
    ax[0].scatter(df["time_s"].iloc[picks], df["dVdt_Vps"].iloc[picks], color="k", s=35, marker="x", zorder=5)
    ax[1].scatter(df["time_s"].iloc[picks], df["dIdt_Aps"].iloc[picks], color="k", s=35, marker="x", zorder=5)

st.pyplot(fig, use_container_width=True)

st.subheader("IV with QSS status")
fig2, ax2 = plt.subplots(1, 1, figsize=(6, 5))
mask = df["QSS_pass"].to_numpy()
ax2.scatter(df["V"][~mask], df["I"][~mask], s=10, color="#ff9896", label="Not QSS")
ax2.scatter(df["V"][mask],  df["I"][mask],  s=10, color="#98df8a", label="QSS pass")
if len(picks) > 0:
    ax2.scatter(df["V"].iloc[picks], df["I"].iloc[picks], s=45, color="k", marker="x", label="Recommended")
ax2.set_xlabel("Voltage [V]"); ax2.set_ylabel("Current [A]")
ax2.grid(True, alpha=0.3); ax2.legend(loc="best")
st.pyplot(fig2, use_container_width=True)

# ============================ Table & Export ============================ #
st.subheader("Per‑sample results")
show_cols = ["time_s","V","I","P","dVdt_Vps","dIdt_Aps","QSS_pass","Recommended"]
st.dataframe(df[show_cols].reset_index(drop=True), use_container_width=True, hide_index=True)

if want_export:
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", data=csv_bytes, file_name="dvdt_didt_qss_table.csv", mime="text/csv")

# ============================ IEC Notes ============================ #
with st.expander("IEC 60904‑1:2020 – How this calculation validates QSS", expanded=False):
    st.markdown("""
**Practical interpretation of the standard**  
- Ideal steady‑state is **dV/dt = 0** and **dI/dt = 0** (capacitive current vanishes: **dQ/dt = 0**).  
- In practice, exact zero is unattainable → accept **quasi‑steady‑state (QSS)** when **both derivatives are sufficiently small** over a short **time interval** before acquisition.

**What this app enforces**  
- Computes **per‑point** derivatives with your finite‑difference formula (Backward/Central/Forward).  
- Applies a **dwell‑based** QSS rule: a sample passes **only if** both |dV/dt| and |dI/dt| stay **below thresholds** for the **entire dwell** immediately preceding that sample.  
- Detects steps and marks the **first QSS‑pass** per step—the operational equivalent of *“waiting sufficiently long”* for transients to decay.  
- Thresholds are **data‑driven** (robust noise‑based suggestion) with **manual override** for your lab‑validated limits.

**Evidence produced**  
- Per‑sample table with **dV/dt**, **dI/dt**, **QSS_pass**, **Recommended**; derivative plots with thresholds; IV plot with QSS coloring.
""")
