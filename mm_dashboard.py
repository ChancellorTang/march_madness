import streamlit as st
import json
import pandas as pd
import plotly.express as px
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
SIMS_DIR = Path("./Sims/2026/")

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
all_regions = sorted([r for r in df.region.unique() if r and r != "Unknown"])

# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"### 📂 Loaded from\n`{SIMS_DIR}`")
    st.markdown(f"**{n_sims}** simulations · **{n_teams}** teams")
    st.markdown("---")
    st.markdown("### 🔍 Filters")
    sel_region = st.selectbox("Select Region (Tab 1)", all_regions, index=0)
    seed_range  = st.slider("Seed range", 1, 16, (1, 16))

df_f = df[df.seed.between(*seed_range)]

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
tab_region, tab_finals = st.tabs(["🌍 By Region (R64-FF)", "🏆 Finals (CG-Winner)"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 – BY REGION (R64 to FINAL FOUR)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_region:
    st.subheader(f"Teams & Seeds by Round — {sel_region}")
    
    # Filter data for selected region and rounds up to FF (4)
    df_region = df_f[(df_f.region == sel_region) & (df_f.rnd >= 4)]
    
    # Build pivot table: teams/seeds as rows, rounds as columns, counts as values
    team_round_counts = (
        df_region.groupby(["team", "seed", "rnd"])
        .sim.nunique()
        .reset_index(name="count")
    )
    
    pivot_region = team_round_counts.pivot_table(
        index=["team", "seed"],
        columns="rnd",
        values="count",
        fill_value=0,
    )
    
    # Ensure columns are in round order and only include rounds >= 4 (FF and earlier)
    included_rounds = [r for r in ROUND_ORDER if r >= 4]
    pivot_region = pivot_region[[c for c in included_rounds if c in pivot_region.columns]]
    pivot_region.columns = [ROUND_META[c]["short"] for c in pivot_region.columns]
    
    # Sort by seed ascending
    pivot_region = pivot_region.sort_index(level='seed', ascending=True)
    
    # Reset index to get team/seed in columns
    pivot_region_df = pivot_region.reset_index()
    
    # Format for display
    display_df = pivot_region_df.set_index(['team', 'seed'])
    display_df.index = [f"{seed}. {team}" for team, seed in display_df.index]
    display_df = display_df.astype(int)
    
    if display_df.empty:
        st.info("No teams match the selected filter.")
    else:
        st.markdown(f"Showing counts out of {n_sims} simulations — cells = number of sims the team reached that round.")
        st.dataframe(display_df, use_container_width=True)
        
        # Heatmap visualization
        fig_region = px.imshow(
            display_df,
            color_continuous_scale="YlOrRd",
            labels=dict(color="Sims"),
            text_auto=True,
            aspect="auto",
        )
        fig_region.update_layout(
            height=max(300, 28 * len(display_df)),
            yaxis_title="Team (seed)",
            xaxis_title="Round",
            **PLOT_LAYOUT,
        )
        fig_region.update_traces(hovertemplate="Team: %{y}<br>Round: %{x}<br>Sims: %{z}<extra></extra>")
        st.plotly_chart(fig_region, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 – FINALS (CHAMPIONSHIP GAME & WINNER)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_finals:
    st.subheader("Teams & Seeds — Championship Game & Winner")
    
    # Filter data for CG (round 2) and Winner (round 1)
    df_finals = df_f[df_f.rnd.isin([2, 1])]
    
    # Build pivot table: teams/seeds as rows, rounds as columns, counts as values
    team_finals_counts = (
        df_finals.groupby(["team", "seed", "rnd"])
        .sim.nunique()
        .reset_index(name="count")
    )
    
    pivot_finals = team_finals_counts.pivot_table(
        index=["team", "seed"],
        columns="rnd",
        values="count",
        fill_value=0,
    )
    
    # Ensure proper column order (CG first, then Winner)
    pivot_finals = pivot_finals[[c for c in [2, 1] if c in pivot_finals.columns]]
    pivot_finals.columns = [ROUND_META[c]["short"] for c in [2, 1] if c in pivot_finals.columns]
    
    # Sort by seed ascending
    pivot_finals = pivot_finals.sort_index(level='seed', ascending=True)
    
    # Reset index to get team/seed in columns
    pivot_finals_df = pivot_finals.reset_index()
    
    # Format for display
    display_df_finals = pivot_finals_df.set_index(['team', 'seed'])
    display_df_finals.index = [f"{seed}. {team}" for team, seed in display_df_finals.index]
    display_df_finals = display_df_finals.astype(int)
    
    if display_df_finals.empty:
        st.info("No teams match the selected filter.")
    else:
        st.markdown(f"Showing counts out of {n_sims} simulations — cells = number of sims the team reached that round.")
        st.dataframe(display_df_finals, use_container_width=True)
        
        # Heatmap visualization
        fig_finals = px.imshow(
            display_df_finals,
            color_continuous_scale="Purples",
            labels=dict(color="Sims"),
            text_auto=True,
            aspect="auto",
        )
        fig_finals.update_layout(
            height=max(300, 28 * len(display_df_finals)),
            yaxis_title="Team (seed)",
            xaxis_title="Round",
            **PLOT_LAYOUT,
        )
        fig_finals.update_traces(hovertemplate="Team: %{y}<br>Round: %{x}<br>Sims: %{z}<extra></extra>")
        st.plotly_chart(fig_finals, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f"<p style='text-align:center;color:rgba(255,255,255,0.2);font-size:0.8rem;'>"
    f"Loaded {n_sims} simulations from <code>{SIMS_DIR}</code></p>",
    unsafe_allow_html=True,
)