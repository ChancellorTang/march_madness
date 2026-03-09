import streamlit as st
import json
import pandas as pd
import plotly.express as px
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
SIMS_DIR = Path("./Sims/2025/")

st.set_page_config(
    page_title="Tournament Sim Dashboard",
    page_icon="🏆",
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
    margin: 0;
    line-height: 1;
}
.title-block p { color: rgba(255,255,255,0.5); margin: 0.4rem 0 0; font-size: 0.9rem; }

.metric-card {
    background: linear-gradient(145deg, #1a1a2e, #16213e);
    border: 1px solid rgba(240,192,64,0.2);
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.metric-card .val {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.2rem;
    color: #f0c040;
    line-height: 1;
}
.metric-card .lbl {
    font-size: 0.75rem;
    color: rgba(255,255,255,0.4);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 0.3rem;
}
section[data-testid="stSidebar"] { background: #0d0d1a; }
section[data-testid="stSidebar"] * { color: #e0e0f0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
ROUND_ORDER = [64, 32, 16, 8, 4, 2, 1]
ROUND_META = {
    64: {"name": "Round of 64",  "short": "R64"},
    32: {"name": "Round of 32",  "short": "R32"},
    16: {"name": "Sweet 16",     "short": "S16"},
    8:  {"name": "Elite 8",      "short": "E8"},
    4:  {"name": "Final Four",   "short": "FF"},
    2:  {"name": "Championship", "short": "CG"},
    1:  {"name": "Champion",     "short": "WIN"},
}
ROUND_DEPTH = {64: 1, 32: 2, 16: 3, 8: 4, 4: 5, 2: 6, 1: 7}

PLOT_LAYOUT = dict(
    plot_bgcolor="#0d0d1a",
    paper_bgcolor="#0d0d1a",
    font_color="#e0e0f0",
    margin=dict(t=20, b=40, l=10, r=20),
)

# ── Load simulations ──────────────────────────────────────────────────────────
@st.cache_data
def load_sims(directory: Path):
    sims, errors = [], []
    files = sorted(directory.glob("*.json"))
    if not files:
        return [], [f"No JSON files found in {directory}"]
    for fp in files:
        try:
            with open(fp) as f:
                data = json.load(f)
            if isinstance(data, list):
                sims.extend(data)
            elif isinstance(data, dict):
                sims.append(data)
            else:
                errors.append(f"{fp.name}: unexpected format")
        except Exception as e:
            errors.append(f"{fp.name}: {e}")
    return sims, errors


@st.cache_data
def build_dataframe(sims: list) -> pd.DataFrame:
    rows = []
    for sim_idx, sim in enumerate(sims):
        for round_key, teams in sim.items():
            rk = int(round_key)
            for t in teams:
                rows.append({
                    "sim":    sim_idx,
                    "rnd":    rk,
                    "team":   t.get("TEAM", "Unknown"),
                    "seed":   int(t.get("SEED", 0)),
                    "region": t.get("region", "Unknown"),
                })
    return pd.DataFrame(rows)


all_sims, load_errors = load_sims(SIMS_DIR)

if load_errors:
    for e in load_errors:
        st.warning(e)
if not all_sims:
    st.error(f"Could not load any simulations from `{SIMS_DIR}`. Check the path and file format.")
    st.stop()

df = build_dataframe(all_sims)
n_sims   = df.sim.nunique()
n_teams  = df[df["rnd"] == 64].team.nunique()
all_regions = sorted(df.region.unique())

# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"### 📂 Loaded from\n`{SIMS_DIR}`")
    st.markdown(f"**{n_sims}** simulations · **{n_teams}** teams")
    st.markdown("---")
    st.markdown("### 🔍 Filters")
    sel_regions = st.multiselect("Regions", all_regions, default=all_regions)
    seed_range  = st.slider("Seed range", 1, 16, (1, 16))
    top_n       = st.slider("Top N teams in charts", 5, 32, 16)

df_f = df[df.region.isin(sel_regions) & df.seed.between(*seed_range)]

# ── Reach % helper ────────────────────────────────────────────────────────────
def reach_pct(groupby_cols, data=None):
    d = (data if data is not None else df_f)
    d = d[d["rnd"].isin(ROUND_ORDER)]
    out = (
        d.groupby(groupby_cols + ["rnd"])
        .sim.nunique()
        .div(n_sims).mul(100)
        .reset_index(name="pct")
    )
    out["round_short"] = out["rnd"].map(lambda r: ROUND_META[r]["short"])
    out["round_name"]  = out["rnd"].map(lambda r: ROUND_META[r]["name"])
    out["round_order"] = out["rnd"].map(lambda r: ROUND_ORDER.index(r))
    return out

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="title-block">
  <h1>🏆 Tournament Simulator</h1>
  <p>Breakdown of simulation outcomes by region and seed · 2025</p>
</div>
""", unsafe_allow_html=True)

# ── KPIs ──────────────────────────────────────────────────────────────────────
champions     = df[df["rnd"] == 1].groupby("team").sim.nunique()
top_champ     = champions.idxmax() if len(champions) else "—"
top_champ_pct = champions.max() / n_sims * 100 if len(champions) else 0

cols = st.columns(4)
for col, val, lbl in [
    (cols[0], f"{n_sims:,}",          "Simulations"),
    (cols[1], f"{n_teams}",            "Teams"),
    (cols[2], top_champ,               "Most Likely Champion"),
    (cols[3], f"{top_champ_pct:.1f}%", "Champion Win Rate"),
]:
    col.markdown(f"""
    <div class="metric-card">
      <div class="val">{val}</div>
      <div class="lbl">{lbl}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_region, tab_seed = st.tabs(["🌍 Region Breakdown", "🌱 Seed Performance"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 – REGION BREAKDOWN
# ═══════════════════════════════════════════════════════════════════════════════
with tab_region:
    def reach_pct(groupby_cols, data=None, as_pct=True):
        d = (data if data is not None else df_f)
        d = d[d["rnd"].isin(ROUND_ORDER)]
        # number of unique sims where the group is present at each round
        counts = (
            d.groupby(groupby_cols + ["rnd"])
            .sim.nunique()
            .reset_index(name="count")
        )
        counts["round_short"] = counts["rnd"].map(lambda r: ROUND_META[r]["short"])
        counts["round_name"]  = counts["rnd"].map(lambda r: ROUND_META[r]["name"])
        counts["round_order"] = counts["rnd"].map(lambda r: ROUND_ORDER.index(r))
        if as_pct:
            counts["pct"] = counts["count"].div(n_sims).mul(100)
        return counts

    # Top teams per region drill-down
    st.subheader("Top teams per region")
    sel_region_detail = st.selectbox("Select a region", all_regions)

    # get both counts and pct (counts = number of sims the team reached that round)
    team_reach = reach_pct(
        ["team", "seed", "region"],
        data=df_f[df_f.region == sel_region_detail],
        as_pct=True,
    )

    # pivot to show counts per team by round (counts are 0..n_sims)
    pivot_team = team_reach.pivot_table(
        index=["team", "seed"],
        columns="round_short",
        values="count",
        fill_value=0,
    )

    # ensure columns are in the expected round order
    ordered_shorts = [ROUND_META[r]["short"] for r in ROUND_ORDER if ROUND_META[r]["short"] in pivot_team.columns]
    pivot_team = pivot_team[ordered_shorts]

    # make labels nicer for display and sort teams by total appearances
    pivot_team["total"] = pivot_team.sum(axis=1)
    pivot_team = pivot_team.sort_values("total", ascending=False)
    pivot_team = pivot_team.drop(columns=["total"])

    st.markdown(f"Showing counts out of {n_sims} simulations — cells = number of sims the team reached that round.")

    # convert MultiIndex to a single string index "Team (seed)" so plotly shows names
    pivot_team.index = [f"{team} ({seed})" for team, seed in pivot_team.index]
    y_labels = pivot_team.index.tolist()

    fig5 = px.imshow(
        pivot_team,
        color_continuous_scale="YlOrRd",
        labels=dict(color="Sims"),
        text_auto=True,
        aspect="auto",
    )
    fig5.update_layout(
        height=max(300, 28 * len(pivot_team)),
        yaxis_title="Team (seed)",
        xaxis_title="Round",
        **PLOT_LAYOUT,
    )
    # hover shows the team label (since the y-axis now uses the string index)
    fig5.update_traces(hovertemplate="Team: %{y}<br>Round: %{x}<br>Sims: %{z}<extra></extra>")
    st.plotly_chart(fig5, width='stretch')

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 – SEED PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_seed:

    # 2a. Seed × Round matrix
    st.subheader("Seed × Round advancement matrix (% of simulations)")
    seed_reach = reach_pct(["seed"])
    pivot_sr = seed_reach.pivot_table(index="seed", columns="rnd", values="pct", fill_value=0)
    pivot_sr = pivot_sr[[c for c in ROUND_ORDER if c in pivot_sr.columns]]
    pivot_sr.columns = [ROUND_META[c]["short"] for c in pivot_sr.columns]
    pivot_sr = pivot_sr.loc[pivot_sr.index.isin(range(seed_range[0], seed_range[1] + 1))]
    fig6 = px.imshow(
        pivot_sr, color_continuous_scale="Blues",
        labels=dict(color="% Sims"), aspect="auto", text_auto=".0f",
    )
    fig6.update_layout(height=460, coloraxis_colorbar=dict(title="% Sims"), **PLOT_LAYOUT)
    st.plotly_chart(fig6, width='stretch')

    col_c, col_d = st.columns(2)

    # 2b. Average round depth per seed
    with col_c:
        st.subheader("Average round depth by seed")
        avg_depth = (
            df_f[df_f["rnd"].isin(ROUND_ORDER)]
            .groupby(["sim", "seed"])["rnd"].max()
            .map(ROUND_DEPTH).reset_index()
            .groupby("seed")["rnd"].mean()
            .reset_index(name="avg_depth")
        )
        avg_depth = avg_depth[avg_depth.seed.between(*seed_range)]
        fig7 = px.bar(
            avg_depth.sort_values("seed"),
            x="seed", y="avg_depth", color="avg_depth",
            color_continuous_scale="Viridis",
            text=avg_depth.sort_values("seed").avg_depth.map(lambda v: f"{v:.2f}"),
            labels={"avg_depth": "Avg Round Depth", "seed": "Seed"},
        )
        fig7.update_traces(textposition="outside")
        fig7.update_layout(
            height=380, showlegend=False,
            xaxis=dict(dtick=1),
            yaxis=dict(range=[0, avg_depth.avg_depth.max() * 1.2]),
            **PLOT_LAYOUT,
        )
        st.plotly_chart(fig7, width='stretch')

    # 2c. Champion % per seed
    with col_d:
        st.subheader("Champion probability by seed")
        champ_seed = (
            df_f[df_f["rnd"] == 1]
            .groupby("seed").sim.nunique()
            .div(n_sims).mul(100)
            .reset_index(name="pct")
        )
        champ_seed = champ_seed[champ_seed.seed.between(*seed_range)]
        fig8 = px.bar(
            champ_seed.sort_values("seed"),
            x="seed", y="pct", color="pct",
            color_continuous_scale="Reds",
            text=champ_seed.sort_values("seed").pct.map(lambda v: f"{v:.1f}%"),
            labels={"pct": "% Champion", "seed": "Seed"},
        )
        fig8.update_traces(textposition="outside")
        fig8.update_layout(
            height=380, showlegend=False,
            xaxis=dict(dtick=1),
            yaxis=dict(range=[0, champ_seed.pct.max() * 1.25]),
            **PLOT_LAYOUT,
        )
        st.plotly_chart(fig8, width='stretch')

    # 2d. Seed champion % split by region
    st.subheader("Champion probability — seed × region")
    seed_reg_champ = (
        df_f[df_f["rnd"] == 1]
        .groupby(["seed", "region"]).sim.nunique()
        .div(n_sims).mul(100)
        .reset_index(name="pct")
    )
    seed_reg_champ = seed_reg_champ[seed_reg_champ.seed.between(*seed_range)]
    fig9 = px.bar(
        seed_reg_champ.sort_values("seed"),
        x="seed", y="pct", color="region",
        barmode="group",
        labels={"pct": "% Champion", "seed": "Seed", "region": "Region"},
    )
    fig9.update_layout(height=380, xaxis=dict(dtick=1), **PLOT_LAYOUT)
    st.plotly_chart(fig9, width='stretch')

    # 2e. Upset tracker
    st.subheader("Upset tracker — high-seed deep runs")
    upset_thresh = st.slider("Show seeds ≥", 5, 16, 10)
    upsets = (
        df_f[df_f["rnd"].isin(ROUND_ORDER) & (df_f.seed >= upset_thresh)]
        .groupby(["team", "seed", "region", "rnd"]).sim.nunique()
        .div(n_sims).mul(100)
        .reset_index(name="pct")
    )
    upsets = upsets[upsets.pct > 0]
    best_upset = (
        upsets.sort_values(["rnd", "pct"], ascending=[False, False])
        .groupby("team").first().reset_index()
        .sort_values(["rnd", "pct"], ascending=[False, False])
        .head(top_n)
    )
    best_upset["label"] = best_upset.apply(
        lambda r: f"({r['seed']}) {r['team']} — {ROUND_META[r['rnd']]['name']}", axis=1
    )
    if best_upset.empty:
        st.info("No upsets found for the selected seed threshold.")
    else:
        fig10 = px.bar(
            best_upset, x="pct", y="label", orientation="h",
            color="seed", color_continuous_scale="Oranges_r",
            text=best_upset.pct.map(lambda v: f"{v:.1f}%"),
            labels={"pct": "% Sims Reached", "label": ""},
        )
        fig10.update_traces(textposition="outside")
        fig10.update_layout(
            height=max(350, 28 * len(best_upset)),
            yaxis=dict(categoryorder="total ascending"),
            margin=dict(t=10, r=70, b=40),
            **{k: v for k, v in PLOT_LAYOUT.items() if k != "margin"},
        )
        st.plotly_chart(fig10, width='stretch')

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f"<p style='text-align:center;color:rgba(255,255,255,0.2);font-size:0.8rem;'>"
    f"Loaded {n_sims} simulations from <code>{SIMS_DIR}</code></p>",
    unsafe_allow_html=True,
)