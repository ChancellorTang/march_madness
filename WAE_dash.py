import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
ACTUAL_PATH = Path("./cbb25.json")        # path to your actual results JSON
SIMS_DIR    = Path("./Sims/2025/")        # directory of simulation JSONs

st.set_page_config(
    page_title="Sim vs Reality",
    page_icon="🎯",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.title-block {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    border-radius: 12px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(255,255,255,0.07);
}
.title-block h1 {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3.2rem;
    letter-spacing: 3px;
    color: #f0c040;
    margin: 0; line-height: 1;
}
.title-block p { color: rgba(255,255,255,0.5); margin: 0.4rem 0 0; font-size: 0.9rem; }
.metric-card {
    background: linear-gradient(145deg, #1a1a2e, #16213e);
    border: 1px solid rgba(240,192,64,0.2);
    border-radius: 10px; padding: 1.2rem 1.5rem; text-align: center;
}
.metric-card .val {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.2rem; color: #f0c040; line-height: 1;
}
.metric-card .lbl {
    font-size: 0.75rem; color: rgba(255,255,255,0.4);
    text-transform: uppercase; letter-spacing: 1.5px; margin-top: 0.3rem;
}
section[data-testid="stSidebar"] { background: #0d0d1a; }
section[data-testid="stSidebar"] * { color: #e0e0f0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
ROUND_ORDER    = [64, 32, 16, 8, 4, 2, 1]
SCORED_ROUNDS  = [32, 16, 8, 4, 2, 1]   # R64 excluded — every team plays it
ROUND_POINTS   = {32: 10, 16: 20, 8: 40, 4: 80, 2: 160, 1: 320}  # bracket scoring
ROUND_META  = {
    64: {"name": "Round of 64",  "short": "R64", "size": 64},
    32: {"name": "Round of 32",  "short": "R32", "size": 32},
    16: {"name": "Sweet 16",     "short": "S16", "size": 16},
    8:  {"name": "Elite 8",      "short": "E8",  "size": 8},
    4:  {"name": "Final Four",   "short": "FF",  "size": 4},
    2:  {"name": "Championship", "short": "CG",  "size": 2},
    1:  {"name": "Champion",     "short": "WIN", "size": 1},
}
ROUND_DEPTH = {64: 1, 32: 2, 16: 3, 8: 4, 4: 5, 2: 6, 1: 7}
MAX_SCORE   = sum(ROUND_META[rk]["size"] * ROUND_POINTS[rk] for rk in SCORED_ROUNDS)
# = 32×10 + 16×20 + 8×40 + 4×80 + 2×160 + 1×320 = 1,920

# Expected round depth by seed (capped at FF since seeding is regional)
# WAE can still go positive beyond FF
SEED_EXPECTED_DEPTH = {
    1: 5,          # Final Four
    2: 4,          # Elite 8
    3: 3, 4: 3,    # Sweet 16
    5: 2, 6: 2, 7: 2, 8: 2,   # Round of 32
}
# Seeds 9-16 → expected depth = 1 (lose in R64)
for s in range(9, 17):
    SEED_EXPECTED_DEPTH[s] = 1

PLOT_LAYOUT = dict(
    plot_bgcolor="#0d0d1a", paper_bgcolor="#0d0d1a",
    font_color="#e0e0f0", margin=dict(t=20, b=40, l=10, r=20),
)

# ── Loaders ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_actual(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)

@st.cache_data
def load_sims(directory: Path):
    sims, filenames, errors = [], [], []
    for fp in sorted(directory.glob("*.json")):
        try:
            with open(fp) as f:
                data = json.load(f)
            if isinstance(data, list):
                sims.extend(data)
                filenames.extend([fp.name] * len(data))
            elif isinstance(data, dict):
                sims.append(data)
                filenames.append(fp.name)
        except Exception as e:
            errors.append(f"{fp.name}: {e}")
    return sims, filenames, errors

# ── Team set helpers ──────────────────────────────────────────────────────────
def teams_in_round(bracket: dict, round_key: int) -> set[str]:
    """Return set of team names that appear in a given round of a bracket."""
    return {t.get("TEAM") or "Unknown" for t in bracket.get(str(round_key), [])}

def round_reached_depth(bracket: dict) -> dict[str, int]:
    """Return {team: deepest round depth reached} for a bracket."""
    result = {}
    for rk in ROUND_ORDER:
        for t in bracket.get(str(rk), []):
            name = t.get("TEAM") or "Unknown"
            result[name] = max(result.get(name, 0), ROUND_DEPTH[rk])
    return result

# ── WAE calculation ───────────────────────────────────────────────────────────
def compute_wae(bracket: dict) -> dict[str, int]:
    """
    Wins Above Expected per team.
    WAE = actual_depth - expected_depth
    Positive = exceeded expectations, negative = underperformed.
    No cap — teams can go above FF expectation.
    Only uses seeds 1-16 in each region (not the full bracket / CG rounds
    for WAE purposes, since seeding context ends at FF).
    """
    # Build seed lookup from R64 (ground truth of seeds)
    seed_lookup = {}
    for t in bracket.get("64", []):
        name = t.get("TEAM") or "Unknown"
        seed_lookup[name] = int(t.get("SEED") or 0)

    depth_map = round_reached_depth(bracket)
    wae = {}
    for team, depth in depth_map.items():
        seed = seed_lookup.get(team)
        if seed is None or seed == 0:
            continue
        expected = SEED_EXPECTED_DEPTH.get(seed, 1)
        wae[team] = depth - expected
    return wae

# ── Build sim-level correct-teams-per-round table ─────────────────────────────
def sim_accuracy_table(actual: dict, sims: list[dict], filenames: list[str]) -> pd.DataFrame:
    """
    For each sim x round: how many teams did the sim correctly predict?
    Returns df with columns: sim, filename, round, correct, total, pct
    """
    rows = []
    for sim_idx, sim in enumerate(sims):
        fname = filenames[sim_idx] if sim_idx < len(filenames) else f"sim_{sim_idx}"
        for rk in SCORED_ROUNDS:
            actual_teams = teams_in_round(actual, rk)
            sim_teams    = teams_in_round(sim, rk)
            total   = len(actual_teams)
            correct = len(actual_teams & sim_teams)
            rows.append({
                "sim":      sim_idx,
                "filename": fname,
                "rnd":      rk,
                "round":    ROUND_META[rk]["short"],
                "correct":  correct,
                "total":    total,
                "points":   correct * ROUND_POINTS.get(rk, 0),
                "pct":      correct / total * 100 if total else 0,
            })
    return pd.DataFrame(rows)

# ── Build per-team WAE comparison ────────────────────────────────────────────
def wae_comparison_table(actual: dict, sims: list[dict]) -> pd.DataFrame:
    actual_wae = compute_wae(actual)

    # Sim average WAE per team
    sim_wae_rows = []
    for sim in sims:
        for team, wae in compute_wae(sim).items():
            sim_wae_rows.append({"team": team, "wae": wae})
    sim_wae_df = pd.DataFrame(sim_wae_rows)
    sim_avg_wae = sim_wae_df.groupby("team").wae.mean().reset_index(name="sim_avg_wae")

    # Seed + region from actual R64
    meta = pd.DataFrame([
        {"team": t.get("TEAM") or "Unknown",
         "seed": int(t.get("SEED") or 0),
         "region": t.get("region") or "Unknown"}
        for t in actual.get("64", [])
    ])

    actual_df = pd.DataFrame([
        {"team": t, "actual_wae": w} for t, w in actual_wae.items()
    ])

    df = meta.merge(actual_df, on="team", how="left")
    df = df.merge(sim_avg_wae, on="team", how="left")
    df["actual_wae"]  = df["actual_wae"].fillna(0).astype(int)
    df["sim_avg_wae"] = df["sim_avg_wae"].fillna(0).round(2)
    df["wae_delta"]   = (df["actual_wae"] - df["sim_avg_wae"]).round(2)
    df = df.sort_values(["region", "seed"]).reset_index(drop=True)
    return df

# ── Load data ─────────────────────────────────────────────────────────────────
try:
    actual = load_actual(ACTUAL_PATH)
except FileNotFoundError:
    st.error(f"Actual results file not found: `{ACTUAL_PATH}`")
    st.stop()

all_sims, sim_filenames, load_errors = load_sims(SIMS_DIR)
if load_errors:
    for e in load_errors:
        st.sidebar.warning(e)
if not all_sims:
    st.error(f"No simulations found in `{SIMS_DIR}`")
    st.stop()

n_sims = len(all_sims)

acc_df  = sim_accuracy_table(actual, all_sims, sim_filenames)
wae_df  = wae_comparison_table(actual, all_sims)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"### 📂 Sources")
    st.markdown(f"Actual: `{ACTUAL_PATH}`")
    st.markdown(f"Sims: `{SIMS_DIR}`")
    st.markdown(f"**{n_sims}** simulations loaded")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="title-block">
  <h1>🎯 Sim vs Reality</h1>
  <p>How well did the simulations predict the actual 2025 tournament outcomes?</p>
</div>
""", unsafe_allow_html=True)

# ── KPIs ──────────────────────────────────────────────────────────────────────
# Best sim = most correct teams summed across all rounds
best_sim_score = acc_df.groupby("sim").points.sum()
best_sim_idx   = int(best_sim_score.idxmax())
best_sim_pts   = int(best_sim_score.max())
avg_sim_pts    = acc_df.groupby("sim").points.sum().mean()

avg_s16 = acc_df[acc_df.rnd == 16].pct.mean()
avg_e8  = acc_df[acc_df.rnd == 8].pct.mean()
avg_ff  = acc_df[acc_df.rnd == 4].pct.mean()

cols = st.columns(4)
for col, val, lbl in [
    (cols[0], f"{n_sims:,}",                        "Simulations"),
    (cols[1], f"Sim #{best_sim_idx}",                "Best Sim Overall"),
    (cols[2], f"{best_sim_pts:,} / {MAX_SCORE:,}",  "Best Sim Score"),
    (cols[3], f"{avg_sim_pts:,.0f} / {MAX_SCORE:,}", "Avg Sim Score"),
]:
    col.markdown(f"""
    <div class="metric-card">
      <div class="val">{val}</div>
      <div class="lbl">{lbl}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_acc, tab_wae = st.tabs(["🎯 Accuracy by Round", "📈 Wins Above Expected"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 – ACCURACY BY ROUND
# ═══════════════════════════════════════════════════════════════════════════════
with tab_acc:

    # ── Distribution: avg/best/worst per round ────────────────────────────────
    st.subheader("Distribution of correct teams per round")

    dist = (
        acc_df.groupby(["rnd", "round"])
        .agg(avg_correct=("correct","mean"),
             best=("correct","max"),
             worst=("correct","min"),
             total=("total","first"))
        .reset_index()
        .sort_values("rnd", key=lambda s: s.map(lambda r: ROUND_ORDER.index(r)))
    )
    dist["avg_pct"]  = dist.avg_correct / dist.total * 100
    dist["label"]    = dist.apply(
        lambda r: f"{r['round']}<br>({int(r['total'])} teams)", axis=1
    )

    fig1 = go.Figure()
    fig1.add_bar(
        x=dist["label"], y=dist["best"],
        name="Best sim", marker_color="rgba(80,200,120,0.35)",
        text=dist.best.map(lambda v: f"{int(v)}"), textposition="outside",
    )
    fig1.add_bar(
        x=dist["label"], y=dist["avg_correct"],
        name="Avg sim", marker_color="#f0c040",
        text=dist.avg_correct.map(lambda v: f"{v:.1f}"), textposition="outside",
    )
    fig1.add_bar(
        x=dist["label"], y=dist["worst"],
        name="Worst sim", marker_color="rgba(220,80,80,0.45)",
        text=dist.worst.map(lambda v: f"{int(v)}"), textposition="outside",
    )
    fig1.add_scatter(
        x=dist["label"], y=dist["total"],
        name="Max possible", mode="markers+lines",
        marker=dict(symbol="line-ew", size=18, color="white", line=dict(width=2, color="white")),
        line=dict(dash="dot", color="rgba(255,255,255,0.3)"),
    )
    fig1.update_layout(
        barmode="overlay", height=420,
        legend=dict(orientation="h", y=1.08),
        yaxis_title="# Teams Correct",
        **PLOT_LAYOUT,
    )
    st.plotly_chart(fig1, width="stretch")

    # ── Avg accuracy % line ───────────────────────────────────────────────────
    st.subheader("Average accuracy % by round")
    fig2 = px.line(
        dist.sort_values("rnd", key=lambda s: s.map(lambda r: ROUND_ORDER.index(r))),
        x="round", y="avg_pct", markers=True,
        labels={"avg_pct": "Avg % Correct", "round": "Round"},
        category_orders={"round": [ROUND_META[r]["short"] for r in ROUND_ORDER]},
    )
    fig2.update_traces(
        line=dict(color="#f0c040", width=2.5), marker=dict(size=10, color="#f0c040")
    )
    fig2.add_hline(y=50, line_dash="dot", line_color="rgba(255,255,255,0.2)",
                   annotation_text="50%", annotation_position="right")
    fig2.update_layout(height=360, yaxis=dict(range=[0, 110]), **PLOT_LAYOUT)
    st.plotly_chart(fig2, width="stretch")

    # ── Per-sim breakdown table ───────────────────────────────────────────────
    st.subheader("Per-simulation breakdown")

    # Build filename map sim_idx -> filename
    sim_fname_map = acc_df[["sim","filename"]].drop_duplicates().set_index("sim")["filename"]
    pivot = acc_df.pivot_table(index="sim", columns="round", values="correct").reset_index()
    pivot["Total"]  = acc_df.groupby("sim").correct.sum().values
    pivot["Score"]  = acc_df.groupby("sim").points.sum().values
    pivot["File"]   = pivot["sim"].map(sim_fname_map)
    # Format correct counts as "X/N"
    for rk in SCORED_ROUNDS:
        short = ROUND_META[rk]["short"]
        total = ROUND_META[rk]["size"]
        if short in pivot.columns:
            pivot[short] = pivot[short].apply(lambda v: f"{int(v)}/{total}")

    col_order = ["sim", "File"] + [ROUND_META[r]["short"] for r in SCORED_ROUNDS if ROUND_META[r]["short"] in pivot.columns] + ["Total", "Score"]
    pivot = pivot[col_order].rename(columns={"sim": "Sim #"})
    pivot = pivot.sort_values("Score", ascending=False).reset_index(drop=True)

    st.dataframe(pivot, use_container_width=False, height=400)

    # ── Per-sim score distribution histogram ─────────────────────────────────
    st.subheader("Distribution of bracket scores across all sims")
    total_scores = acc_df.groupby("sim").points.sum().reset_index(name="score")
    fig3 = px.histogram(
        total_scores, x="score", nbins=20,
        labels={"score": f"Bracket score (max {MAX_SCORE:,})"},
        color_discrete_sequence=["#f0c040"],
    )
    fig3.add_vline(
        x=total_scores.score.mean(),
        line_dash="dash", line_color="white",
        annotation_text=f"avg {total_scores.score.mean():,.0f}",
        annotation_position="top right",
    )
    fig3.update_layout(height=320, showlegend=False, **PLOT_LAYOUT)
    st.plotly_chart(fig3, width="stretch")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 – WINS ABOVE EXPECTED
# ═══════════════════════════════════════════════════════════════════════════════
with tab_wae:

    st.markdown("""
    **WAE = round reached − seed expectation**  
    Seed expectations: 1→FF(5), 2→E8(4), 3-4→S16(3), 5-8→R32(2), 9-16→R64(1).  
    Positive = overperformed, negative = underperformed. No cap — exceeding FF counts.
    """)

    # ── Top-of-tab filters (apply to entire WAE tab) ──────────────────────────
    all_regions = sorted(wae_df.region.unique())
    f_col1, f_col2 = st.columns([2, 1])
    with f_col1:
        sel_regions = st.multiselect(
            "Filter by region", all_regions, default=all_regions, key="wae_region_filter"
        )
    with f_col2:
        # Build per-sim WAE for scatter sim filter
        per_sim_wae_rows = []
        for sim_idx, sim in enumerate(all_sims):
            fname = sim_filenames[sim_idx] if sim_idx < len(sim_filenames) else f"sim_{sim_idx}"
            for team, wae_val in compute_wae(sim).items():
                per_sim_wae_rows.append({"sim": sim_idx, "filename": fname, "team": team, "wae": wae_val})
        per_sim_wae_df = pd.DataFrame(per_sim_wae_rows)

        scatter_options = ["All sims (avg)"] + sorted(per_sim_wae_df.filename.unique().tolist())
        scatter_sim = st.selectbox("Scatter: compare against", scatter_options, key="scatter_sim_select")

    wae_filtered = wae_df[wae_df.region.isin(sel_regions)].copy()

    # ── Scatter: actual WAE vs sim WAE (avg or specific sim) ─────────────────
    st.subheader("Actual WAE vs Sim WAE")
    if scatter_sim == "All sims (avg)":
        scatter_x_col   = "sim_avg_wae"
        scatter_x_label = "Sim Avg WAE"
        scatter_data    = wae_filtered.copy()
    else:
        # Compute WAE for selected sim file
        sel_sim_wae = (
            per_sim_wae_df[per_sim_wae_df.filename == scatter_sim]
            .rename(columns={"wae": "sel_sim_wae"})
            [["team", "sel_sim_wae"]]
        )
        scatter_data  = wae_filtered.merge(sel_sim_wae, on="team", how="left").fillna({"sel_sim_wae": 0})
        scatter_data["wae_delta"] = (scatter_data["actual_wae"] - scatter_data["sel_sim_wae"]).round(2)
        scatter_x_col   = "sel_sim_wae"
        scatter_x_label = f"WAE: {scatter_sim}"

    fig4 = px.scatter(
        scatter_data,
        x=scatter_x_col, y="actual_wae",
        color="region", symbol="region",
        hover_data={"team": True, "seed": True, "wae_delta": True},
        labels={
            scatter_x_col: scatter_x_label,
            "actual_wae":  "Actual WAE",
            "wae_delta":   "Delta (actual − sim)",
        },
        text="team",
    )
    # Diagonal reference line (perfect prediction)
    wae_min = min(scatter_data[["actual_wae", scatter_x_col]].min())
    wae_max = max(scatter_data[["actual_wae", scatter_x_col]].max())
    fig4.add_shape(
        type="line", x0=wae_min, y0=wae_min, x1=wae_max, y1=wae_max,
        line=dict(dash="dot", color="rgba(255,255,255,0.3)"),
    )
    fig4.add_vline(x=0, line_dash="dot", line_color="rgba(255,255,255,0.15)")
    fig4.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.15)")
    fig4.update_traces(textposition="top center", marker_size=9)
    fig4.update_layout(height=540, **PLOT_LAYOUT)
    st.plotly_chart(fig4, width="stretch")

    st.caption("Teams above the diagonal: sim underestimated them. Below: sim overestimated them.")

    # ── Bar chart: delta (actual − sim avg) ───────────────────────────────────
    st.subheader("WAE Delta — where sims were most wrong (actual − sim avg)")
    wae_sorted = wae_filtered.sort_values("wae_delta", ascending=False)
    fig5 = px.bar(
        wae_sorted,
        x="team", y="wae_delta",
        color="wae_delta",
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0,
        labels={"wae_delta": "Delta (actual − sim avg)", "team": "Team"},
        hover_data={"seed": True, "region": True, "actual_wae": True, "sim_avg_wae": True},
    )
    fig5.add_hline(y=0, line_color="white", line_width=1)
    fig5.update_layout(
        height=420,
        xaxis_tickangle=-45,
        showlegend=False,
        coloraxis_showscale=False,
        **PLOT_LAYOUT,
    )
    st.plotly_chart(fig5, width="stretch")

    # ── Side-by-side bars: actual vs sim avg WAE per team ────────────────────
    st.subheader("Actual vs Sim Avg WAE by team")
    sel_region_wae = st.selectbox("Region detail", ["All"] + list(sel_regions), key="wae_bar_region")
    wae_plot = wae_filtered if sel_region_wae == "All" else wae_filtered[wae_filtered.region == sel_region_wae]
    wae_plot = wae_plot.sort_values(["region","seed"])

    fig6 = go.Figure()
    fig6.add_bar(
        x=wae_plot["team"], y=wae_plot["actual_wae"],
        name="Actual WAE", marker_color="#f0c040",
    )
    fig6.add_bar(
        x=wae_plot["team"], y=wae_plot["sim_avg_wae"],
        name="Sim Avg WAE", marker_color="rgba(100,160,255,0.75)",
    )
    fig6.add_hline(y=0, line_color="white", line_width=1)
    fig6.update_layout(
        barmode="group", height=440,
        xaxis_tickangle=-45,
        legend=dict(orientation="h", y=1.08),
        **PLOT_LAYOUT,
    )
    st.plotly_chart(fig6, width="stretch")

    # ── Full WAE table ────────────────────────────────────────────────────────
    st.subheader("Full WAE table")
    display_cols = ["team","seed","region","actual_wae","sim_avg_wae","wae_delta"]
    st.dataframe(
        wae_filtered[display_cols]
        .sort_values("wae_delta", ascending=False)
        .rename(columns={
            "team":        "Team",
            "seed":        "Seed",
            "region":      "Region",
            "actual_wae":  "Actual WAE",
            "sim_avg_wae": "Sim Avg WAE",
            "wae_delta":   "Delta (actual − sim)",
        })
        .reset_index(drop=True),
        use_container_width=False,
        height=500,
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f"<p style='text-align:center;color:rgba(255,255,255,0.2);font-size:0.8rem;'>"
    f"Actual: <code>{ACTUAL_PATH}</code> · Sims: <code>{SIMS_DIR}</code></p>",
    unsafe_allow_html=True,
)