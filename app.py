"""
app.py  --  Multi-Organ-Chip Channel Sizing (web version of channel_sizing_gui.py)

Same physics as the desktop GUI:
    Rh = 12 * mu * L / (H^3 * W)      Q = dP / Rh
Qin is split into Q1 / Q2, one dimension (L, W or H) is fixed, another is swept,
and the third is solved in closed form. Includes plots, the combined fold
"Layout" design guide (Prev / Next), and CSV export (download button).

Run locally:   streamlit run app.py
"""

import csv
import io

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
import streamlit as st

# ---------------- physics (simplified: Rh = 12*mu*L / (H^3 * W)) ----------------

def resistance(mu, L, H, W):
    if H <= 0 or W <= 0 or L <= 0:
        return np.nan
    return (12.0 * mu * L) / (H**3 * W)

def solve_L(R_target, mu, H, W):
    if H <= 0 or W <= 0:
        return np.nan
    return (R_target * H**3 * W) / (12.0 * mu)

def solve_H(R_target, mu, L, W):
    if L <= 0 or W <= 0 or R_target <= 0:
        return np.nan
    return ((12.0 * mu * L) / (R_target * W)) ** (1.0 / 3.0)

def solve_W(R_target, mu, L, H):
    if L <= 0 or H <= 0 or R_target <= 0:
        return np.nan
    return (12.0 * mu * L) / (R_target * H**3)

UM = 1e-6
def um_to_m(x):
    return x * UM

def m_to_um(x):
    return x / UM if np.isfinite(x) else np.nan

def uLmin_to_m3s(q):
    return (q * 1e-9) / 60.0

def m3s_to_uLmin(q):
    return (q * 60.0) / 1e-9

MAX_DESIRABLE_UM = 1000.0

# ---------------- palette ----------------
BG = "#eaf6f6"
BANNER = "#0f3d5c"
ACCENT = "#1f7a8c"
C_L, C_W, C_H = "#1f7a8c", "#c96f2c", "#7a3ea3"
TEXT = "#0b2436"
GRID = "#c7dede"

MODE_DEFAULTS = {
    "height": dict(fixed=500.0, smin=200.0, smax=900.0),
    "width":  dict(fixed=800.0, smin=100.0, smax=600.0),
    "length": dict(fixed=3000.0, smin=200.0, smax=900.0),
}
SWEEP_NAME = {"height": "width", "width": "height", "length": "width"}

# ---------------- computation ----------------

def compute_rows_for_branch(branch, p):
    """Sweep table for 'Q1' or 'Q2'. Returns (rows, Q_channel, Q1, Q2, R_target)."""
    Q_m3s = uLmin_to_m3s(p["Q_uLmin"])
    Q1 = Q_m3s * (p["split"] / 100.0)
    Q2 = Q_m3s * (1.0 - p["split"] / 100.0)
    Q_channel = Q1 if branch == "Q1" else Q2

    if Q_channel <= 0 or p["sweep_max"] <= p["sweep_min"] or p["mu"] <= 0 or p["dP"] <= 0:
        return None, Q_channel, Q1, Q2, np.nan

    mu, fixed_var, fixed_val = p["mu"], p["fixed_var"], p["fixed_val"]
    R_target = p["dP"] / Q_channel

    rows = []
    for sv in np.linspace(p["sweep_min"], p["sweep_max"], 14):
        if fixed_var == "height":
            H_um, W_um = fixed_val, sv
            L_um = m_to_um(solve_L(R_target, mu, um_to_m(H_um), um_to_m(W_um)))
        elif fixed_var == "width":
            W_um, H_um = fixed_val, sv
            L_um = m_to_um(solve_L(R_target, mu, um_to_m(H_um), um_to_m(W_um)))
        else:  # fixed length -> sweep width, solve height
            L_um, W_um = fixed_val, sv
            H_um = m_to_um(solve_H(R_target, mu, um_to_m(L_um), um_to_m(W_um)))

        valid = all(np.isfinite(v) and v > 0 for v in (L_um, W_um, H_um))
        out_of_range = any((not np.isfinite(v)) or v > MAX_DESIRABLE_UM for v in (L_um, W_um, H_um))
        Rchk = resistance(mu, um_to_m(L_um), um_to_m(H_um), um_to_m(W_um)) if valid else np.nan
        rows.append(dict(sweep=sv, L_um=L_um, W_um=W_um, H_um=H_um, Rchk=Rchk,
                         valid=valid, out_of_range=out_of_range))
    return rows, Q_channel, Q1, Q2, R_target


def reference_row(rows):
    if not rows:
        return None
    valid_rows = [r for r in rows if r["valid"]]
    return valid_rows[len(valid_rows) // 2] if valid_rows else rows[0]

# ---------------- plots ----------------

def make_plots(rows, mu, R_target):
    ref = reference_row(rows)
    fig = Figure(figsize=(11, 3.6), dpi=100, facecolor=BG)
    axes = [fig.add_subplot(1, 3, i + 1) for i in range(3)]
    for ax in axes:
        ax.set_facecolor("white")
        ax.grid(True, color=GRID, linewidth=0.6)
    if ref is None:
        return fig

    Lref, Wref, Href = ref["L_um"], ref["W_um"], ref["H_um"]
    ax_l, ax_w, ax_h = axes

    Ls = np.linspace(max(Lref * 0.2, 1e-3), Lref * 3.0, 40)
    ax_l.plot(Ls, [resistance(mu, um_to_m(l), um_to_m(Href), um_to_m(Wref)) for l in Ls], "-", color=C_L)
    ax_l.axhline(R_target, color="#999", linestyle="--", linewidth=1)
    ax_l.plot(Lref, ref["Rchk"], "o", color=C_L, ms=6)
    ax_l.set_xlabel("Length (µm)", fontsize=9)
    ax_l.set_ylabel("Resistance R", fontsize=9)
    ax_l.set_title(f"R vs Length\n(W={Wref:.0f}µm, H={Href:.0f}µm)", fontsize=9)

    Ws = np.linspace(max(Wref * 0.2, 1e-3), Wref * 3.0, 40)
    ax_w.plot(Ws, [resistance(mu, um_to_m(Lref), um_to_m(Href), um_to_m(w)) for w in Ws], "-", color=C_W)
    ax_w.axhline(R_target, color="#999", linestyle="--", linewidth=1)
    ax_w.plot(Wref, ref["Rchk"], "o", color=C_W, ms=6)
    ax_w.set_xlabel("Width (µm)", fontsize=9)
    ax_w.set_ylabel("Resistance R", fontsize=9)
    ax_w.set_title(f"R vs Width\n(L={Lref:.0f}µm, H={Href:.0f}µm)", fontsize=9)

    Hs = np.linspace(max(Href * 0.2, 1e-3), Href * 3.0, 40)
    ax_h.plot(Hs, [resistance(mu, um_to_m(Lref), um_to_m(h), um_to_m(Wref)) for h in Hs], "-", color=C_H)
    ax_h.axhline(R_target, color="#999", linestyle="--", linewidth=1)
    ax_h.plot(Href, ref["Rchk"], "o", color=C_H, ms=6)
    ax_h.set_xlabel("Height (µm)", fontsize=9)
    ax_h.set_ylabel("Resistance R", fontsize=9)
    ax_h.set_title(f"R vs Height\n(L={Lref:.0f}µm, W={Wref:.0f}µm)", fontsize=9)

    fig.tight_layout()
    return fig

# ---------------- layout schematic (goalpost fold, both branches) ----------------

_AX_W_PTS = 9.4 * (0.97 - 0.05) * 72.0
_AX_H_PTS = 6.2 * (0.93 - 0.07) * 72.0
_LEG_LABEL_PTS = 9
_RUN_LABEL_PTS = 11
_BRACE_BASE_PTS = 24
_BRACE_BULGE_PTS = 11
_WH_LABEL_PTS = 58
_MARGIN_PTS_Y = 82
_MARGIN_PTS_X = 26


def _solve_margin_data(content_range, margin_pts, axis_pts):
    content_range = max(content_range, 1e-9)
    denom = axis_pts - 2.0 * margin_pts
    if denom <= 0:
        denom = axis_pts * 0.5
    return margin_pts * content_range / denom


def _brace(ax, x0, x1, y, size, text, above=True, color=TEXT):
    if x1 - x0 < 1e-6:
        x1 = x0 + 1e-6
    resolution = 101
    beta = 12.0 / (x1 - x0)
    x = np.linspace(x0, x1, resolution)
    half = resolution // 2 + 1
    x_half = x[:half]
    y_half = (1 / (1 + np.exp(-beta * (x_half - x_half[0]))) +
              1 / (1 + np.exp(-beta * (x_half - x_half[-1]))))
    y_curve = np.concatenate((y_half, y_half[-2::-1]))
    y_curve = y_curve - y_curve.min()
    if y_curve.max() > 0:
        y_curve = y_curve / y_curve.max()
    sign = 1.0 if above else -1.0
    ax.plot(x, y + sign * size * y_curve, "-", color=color, linewidth=1.1, solid_capstyle="round")
    ax.text((x0 + x1) / 2.0, y + sign * size * 1.35, text, fontsize=8.3, color=color, ha="center",
            va="bottom" if above else "top", fontweight="bold")


def _optimal_fold_pair(L1, L2):
    """One shared top-run for both branches so the two folds close into one rectangle."""
    if not (np.isfinite(L1) and L1 > 0 and np.isfinite(L2) and L2 > 0):
        return None
    run = min(L1, L2) / 3.0
    leg1 = (L1 - run) / 2.0
    leg2 = (L2 - run) / 2.0
    if leg1 <= 0 or leg2 <= 0:
        return None
    return dict(run=run, leg1=leg1, leg2=leg2, lead=run * 0.35)


def draw_combined_schematic(ax, r1, r2):
    ax.clear()
    ax.set_facecolor("white")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    if not (r1["valid"] and r2["valid"]):
        ax.text(0.5, 0.5,
                "One or both branches are non-physical at this sweep point\n"
                "(L, W, or H solved as \u2264 0 or infinite). Try Prev / Next\n"
                "to pick another combination.",
                ha="center", va="center", fontsize=11, color="#a33333", transform=ax.transAxes)
        return

    L1, W1, H1 = r1["L_um"], r1["W_um"], r1["H_um"]
    L2, W2, H2 = r2["L_um"], r2["W_um"], r2["H_um"]

    plan = _optimal_fold_pair(L1, L2)
    if plan is None:
        ax.text(0.5, 0.5, "Cannot compute a fold plan for this length.",
                ha="center", va="center", fontsize=11, color="#a33333", transform=ax.transAxes)
        return

    run, lead_in = plan["run"], plan["lead"]
    leg1, leg2 = plan["leg1"], plan["leg2"]

    jy, x0 = 0.0, 0.0
    jx = x0 + lead_in
    x_end = jx + run
    x_total = x_end + lead_in
    y1_top, y2_top = leg1, -leg2

    margin_y = _solve_margin_data(leg1 + leg2, _MARGIN_PTS_Y, _AX_H_PTS)
    margin_x = _solve_margin_data(x_total, _MARGIN_PTS_X, _AX_W_PTS)
    ax.set_xlim(x0 - margin_x, x_total + margin_x)
    ax.set_ylim(y2_top - margin_y, y1_top + margin_y)

    dpp_x = (2.0 * margin_x + x_total) / _AX_W_PTS
    dpp_y = (2.0 * margin_y + leg1 + leg2) / _AX_H_PTS

    # shared inlet
    ax.plot([x0, jx], [jy, jy], "-", color=TEXT, linewidth=6, solid_capstyle="butt")
    ax.annotate("", xy=(jx - lead_in * 0.1, jy), xytext=(x0 + lead_in * 0.35, jy),
                arrowprops=dict(arrowstyle="-|>", color=TEXT, lw=1.4))
    ax.plot(jx, jy, "o", color=TEXT, ms=5)

    def draw_branch(direction, leg, L, W, H, color, label):
        y_top = jy + direction * leg
        for xs, ys in (([jx, jx], [jy, y_top]), ([jx, x_end], [y_top, y_top]), ([x_end, x_end], [y_top, jy])):
            ax.plot(xs, ys, "-", color=color, linewidth=3.2, solid_capstyle="round", solid_joinstyle="round")

        leg_dx = _LEG_LABEL_PTS * dpp_x
        ax.text(jx - leg_dx, (jy + y_top) / 2.0, f"{leg:.1f}", fontsize=8.2, color=color,
                ha="right", va="center", rotation=90)
        ax.text(x_end + leg_dx, (jy + y_top) / 2.0, f"{leg:.1f}", fontsize=8.2, color=color,
                ha="left", va="center", rotation=90)

        run_dy = _RUN_LABEL_PTS * dpp_y
        ax.text((jx + x_end) / 2.0, y_top + direction * run_dy, f"{run:.1f}", fontsize=8.2,
                color=color, ha="center", va="bottom" if direction > 0 else "top")

        # outlet lead
        ax.plot([x_end, x_total], [jy, jy], "-", color=TEXT, linewidth=6, solid_capstyle="butt")
        ax.annotate("", xy=(x_total + leg_dx, jy), xytext=(x_total, jy),
                    arrowprops=dict(arrowstyle="-|>", color=TEXT, lw=1.4))

        brace_y = y_top + direction * (_BRACE_BASE_PTS * dpp_y)
        _brace(ax, jx, x_end, brace_y, _BRACE_BULGE_PTS * dpp_y,
               f"{label}: {L:.1f} µm total  =  {leg:.1f} + {run:.1f} + {leg:.1f}",
               above=(direction > 0), color=color)

        ax.text(jx, y_top + direction * (_WH_LABEL_PTS * dpp_y),
                f"W = {W:.1f} µm   ·   H = {H:.1f} µm", fontsize=7.8, color=color,
                ha="left", va="bottom" if direction > 0 else "top")

    draw_branch(+1, leg1, L1, W1, H1, C_L, "Branch 1 (Q1)")
    draw_branch(-1, leg2, L2, W2, H2, C_W, "Branch 2 (Q2)")


def make_layout_figure(r1, r2):
    fig = Figure(figsize=(9.4, 6.2), dpi=100, facecolor=BG)
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(top=0.93, bottom=0.07, left=0.05, right=0.97)
    draw_combined_schematic(ax, r1, r2)
    return fig

# ---------------- CSV export ----------------

def build_csv(p, rows1, rows2):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Channel Sizing Export"])
    w.writerow([])
    w.writerow(["Input parameters"])
    w.writerow(["Qin (uL/min)", p["Q_uLmin"]])
    w.writerow(["Q1 split (%)", p["split"]])
    w.writerow(["Q2 split (%)", 100.0 - p["split"]])
    w.writerow(["dP (Pa)", p["dP"]])
    w.writerow(["Viscosity mu (Pa.s)", p["mu"]])
    w.writerow(["Fixed dimension", p["fixed_var"]])
    w.writerow(["Fixed value (um)", p["fixed_val"]])
    w.writerow(["Sweep min (um)", p["sweep_min"]])
    w.writerow(["Sweep max (um)", p["sweep_max"]])
    w.writerow([])
    w.writerow(["L1 (um)", "H1 (um)", "W1 (um)", "L2 (um)", "H2 (um)", "W2 (um)"])
    f3 = lambda v: f"{v:.3f}" if np.isfinite(v) else ""
    for r1, r2 in zip(rows1, rows2):
        w.writerow([f3(r1["L_um"]), f3(r1["H_um"]), f3(r1["W_um"]),
                    f3(r2["L_um"]), f3(r2["H_um"]), f3(r2["W_um"])])
    return buf.getvalue()

# ---------------- Streamlit UI ----------------

st.set_page_config(page_title="Multi-Organ-Chip — Channel Sizing", layout="wide")

st.markdown(
    f"""
    <style>
      .banner {{background:{BANNER}; color:#fff; padding:14px 18px; border-radius:8px;
                font-size:1.25rem; font-weight:700; margin-bottom:12px;}}
      .summary {{color:{ACCENT}; font-family:Consolas, monospace; font-weight:700; font-size:0.9rem;}}
    </style>
    <div class="banner">Channel Sizing — Qin → Q1/Q2 → L, W, H</div>
    """,
    unsafe_allow_html=True,
)

# defaults
for k, v in dict(q_in=50.0, dp=100.0, mu=0.001, q1_split=50.0, q2_split=50.0,
                 branch="Q1", fixed_var="height", fixed_val=500.0,
                 sweep_min=200.0, sweep_max=900.0, view="Dashboard",
                 layout_idx=0, sig=None).items():
    st.session_state.setdefault(k, v)


def on_q1_change():
    st.session_state.q2_split = round(100.0 - st.session_state.q1_split, 4)

def on_q2_change():
    st.session_state.q1_split = round(100.0 - st.session_state.q2_split, 4)

def on_fix_change():
    d = MODE_DEFAULTS[st.session_state.fixed_var]
    st.session_state.fixed_val = d["fixed"]
    st.session_state.sweep_min = d["smin"]
    st.session_state.sweep_max = d["smax"]


c1, c2, c3, c4, c5 = st.columns(5)
c1.number_input("Qin (µL/min)", min_value=0.0, step=1.0, format="%.4f", key="q_in")
c2.number_input("ΔP (Pa)", min_value=0.0, step=1.0, format="%.4f", key="dp")
c3.number_input("Viscosity μ (Pa·s)", min_value=0.0, step=0.0001, format="%.6f", key="mu")
c4.number_input("Q1 split (%)", min_value=0.0, max_value=100.0, step=1.0, format="%.2f",
                key="q1_split", on_change=on_q1_change)
c5.number_input("Q2 split (%)", min_value=0.0, max_value=100.0, step=1.0, format="%.2f",
                key="q2_split", on_change=on_q2_change)

st.radio("Design channel for:", ["Q1", "Q2"], horizontal=True, key="branch",
         format_func=lambda b: "Branch 1 (Q1, R1)" if b == "Q1" else "Branch 2 (Q2, R2)")

st.radio("Fix one dimension:", ["length", "width", "height"], horizontal=True, key="fixed_var",
         format_func=lambda v: f"Fix {v}", on_change=on_fix_change)

fv = st.session_state.fixed_var
f1, f2, f3c = st.columns(3)
f1.number_input(f"Fixed {fv} (µm)", min_value=0.0, step=10.0, format="%.2f", key="fixed_val")
f2.number_input(f"Sweep {SWEEP_NAME[fv]} min (µm)", min_value=0.0, step=10.0, format="%.2f", key="sweep_min")
f3c.number_input(f"Sweep {SWEEP_NAME[fv]} max (µm)", min_value=0.0, step=10.0, format="%.2f", key="sweep_max")

params = dict(
    Q_uLmin=st.session_state.q_in,
    split=st.session_state.q1_split,
    dP=st.session_state.dp,
    mu=st.session_state.mu,
    fixed_var=fv,
    fixed_val=st.session_state.fixed_val,
    sweep_min=st.session_state.sweep_min,
    sweep_max=st.session_state.sweep_max,
)

branch = st.session_state.branch
rows, Q_channel, Q1, Q2, R_target = compute_rows_for_branch(branch, params)
rows1, *_ = compute_rows_for_branch("Q1", params)
rows2, *_ = compute_rows_for_branch("Q2", params)

if rows is None or rows1 is None or rows2 is None:
    st.error("Could not compute with the current inputs. Check that Qin, ΔP and μ are > 0, "
             "the split leaves both branches > 0, and sweep max > sweep min.")
    st.stop()

st.markdown(
    f"<div class='summary'>Designing: {branch} &nbsp; Q_channel = {m3s_to_uLmin(Q_channel):.3f} µL/min "
    f"(Q1={m3s_to_uLmin(Q1):.3f}, Q2={m3s_to_uLmin(Q2):.3f} µL/min) &nbsp; "
    f"R_required = {R_target:.4g} Pa·s/m³ &nbsp; [Rh = 12μL/(H³W)]</div>",
    unsafe_allow_html=True,
)

# reset layout index to the reference combo whenever the inputs change
n = min(len(rows1), len(rows2))
sig = tuple(params.values())
if st.session_state.sig != sig:
    st.session_state.sig = sig
    ref1 = reference_row(rows1)
    st.session_state.layout_idx = rows1.index(ref1) if ref1 in rows1 else n // 2
ref_idx = rows1.index(reference_row(rows1)) if reference_row(rows1) in rows1 else n // 2

st.download_button("⬇ Export CSV (Q1 + Q2)", data=build_csv(params, rows1, rows2),
                   file_name="channel_sizing_results.csv", mime="text/csv")

st.radio("View", ["Dashboard", "Layout guide"], horizontal=True, key="view",
         label_visibility="collapsed")

if st.session_state.view == "Dashboard":
    fmt = lambda v, s=".1f": format(v, s) if np.isfinite(v) else "—"
    df = pd.DataFrame([{
        "L (µm)": fmt(r["L_um"]),
        "W (µm)": fmt(r["W_um"]),
        "H (µm)": fmt(r["H_um"]),
        "R check (Pa·s/m³)": fmt(r["Rchk"], ".4g"),
        "≤1000µm": "yes" if not r["out_of_range"] else "no",
    } for r in rows])
    df.index = range(1, len(df) + 1)
    df.index.name = "#"
    st.dataframe(df, width="stretch")
    st.pyplot(make_plots(rows, params["mu"], R_target), width="stretch")
else:
    i = st.session_state.layout_idx % n
    b1, b2, b3, _ = st.columns([1, 1, 1.4, 4])
    if b1.button("◀ Prev"):
        st.session_state.layout_idx = (i - 1) % n
        st.rerun()
    if b2.button("Next ▶"):
        st.session_state.layout_idx = (i + 1) % n
        st.rerun()
    if b3.button("Reference combo"):
        st.session_state.layout_idx = ref_idx
        st.rerun()

    r1, r2 = rows1[i], rows2[i]
    st.markdown(
        f"<div class='summary'>Combination {i + 1} / {n} — sweep point = {r1['sweep']:.1f} µm "
        f"(fixed {fv} = {params['fixed_val']:.1f} µm)</div>",
        unsafe_allow_html=True,
    )
    st.pyplot(make_layout_figure(r1, r2), width="stretch")
    st.caption("Branch 1 = Q1 (loops up)  ·  Branch 2 = Q2 (loops down)")
