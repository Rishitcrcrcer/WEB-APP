"""
HMM Market Regime Detection - Final Interactive Dashboard
IIT Indore - SPY + Cross-Asset Analysis (SPY, GLD, TLT, BTC)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM

# PAGE CONFIG
st.set_page_config(
    page_title="HMM Regime Dashboard - IIT Indore",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# THEME
PRIMARY_BG = "#0b0e14"
PANEL_BG   = "#11151d"
GRID       = "#242b38"
TEXT       = "#e6e9ef"
MUTED      = "#8a93a6"
ACCENT     = "#4fd1c5"

REGIME_COLORS      = {"Bull": "#3ddc84", "Chop": "#f5b942", "Bear": "#ef5b5b"}
REGIME_COLORS_SOFT = {"Bull": "rgba(61,220,132,0.12)", "Chop": "rgba(245,185,66,0.12)", "Bear": "rgba(239,91,91,0.12)"}
ASSET_COLORS       = {"SPY": "#4fd1c5", "GLD": "#f5b942", "TLT": "#a78bfa", "BTC": "#f97316"}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@400;500;600;700&display=swap');
.stApp {{ background-color: {PRIMARY_BG}; font-family: 'Inter', sans-serif; }}
section[data-testid="stSidebar"] {{ background-color: {PANEL_BG}; border-right: 1px solid {GRID}; }}
h1,h2,h3,h4,p,span,label,.stMarkdown {{ color: {TEXT} !important; }}
div[data-testid="stMetricValue"] {{ color: {ACCENT} !important; font-family: 'IBM Plex Mono', monospace; font-size: 1.5rem !important; }}
div[data-testid="stMetricLabel"] {{ color: {MUTED} !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 0.05em; }}
div[data-testid="stMetric"] {{ background-color: {PANEL_BG}; border: 1px solid {GRID}; border-radius: 10px; padding: 14px 18px; }}
.block-container {{ padding-top: 1.5rem; max-width: 1400px; }}
hr {{ border-color: {GRID}; }}
.stTabs [data-baseweb="tab-list"] {{ background-color: {PANEL_BG}; border-bottom: 1px solid {GRID}; gap: 0; }}
.stTabs [data-baseweb="tab"] {{ color: {MUTED}; font-weight: 500; padding: 0.75rem 1.5rem; border-radius: 0; }}
.stTabs [aria-selected="true"] {{ color: {ACCENT} !important; border-bottom: 2px solid {ACCENT} !important; background: transparent !important; }}
.section-tag {{ font-size: 0.7rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: {ACCENT}; border-left: 3px solid {ACCENT}; padding-left: 0.6rem; margin-bottom: 0.4rem; }}
</style>
""", unsafe_allow_html=True)

PLOTLY_BASE = dict(
    paper_bgcolor=PANEL_BG, plot_bgcolor=PANEL_BG,
    font=dict(color=TEXT, family="IBM Plex Mono, monospace", size=11),
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=GRID, borderwidth=1),
    margin=dict(l=10, r=10, t=50, b=10),
    hovermode="x unified",
)

# CONSTANTS
FEATURES = ["log_return", "volume_zscore", "momentum", "parkinson_vol", "rsi", "ma_cross"]
FEATURE_LABELS = {
    "log_return": "Log Return", "volume_zscore": "Volume Z-Score",
    "momentum": "Momentum", "parkinson_vol": "Parkinson Volatility",
    "rsi": "RSI", "ma_cross": "MA Crossover",
}
TRAIN_WINDOW   = 1260
STEP_SIZE      = 63
N_STATES       = 3
N_ITER         = 200
COV_TYPE       = "diag"
RANDOM_STATE   = 42
TARGET_VOL     = 0.15
VOL_WINDOW     = 20
ANNUALIZATION  = 252
TRANS_COST_BPS = 0.0002

CROSS_ASSET_FILES = {
    "SPY": "hmm_oos_results_SPY.csv",
    "GLD": "hmm_oos_results_GLD.csv",
    "TLT": "hmm_oos_results_TLT.csv",
    "BTC": "hmm_oos_results_BTC-USD.csv",
}

ASSET_LABELS = {"SPY": "S&P 500 ETF", "GLD": "Gold ETF", "TLT": "Treasury Bonds", "BTC": "Bitcoin"}

# FEATURE ENGINEERING
def parkinson_volatility(df, window=20):
    return np.sqrt((1/(4*np.log(2))) * (np.log(df["High"]/df["Low"])**2).rolling(window).mean()) * np.sqrt(252)

def rsi_calc(df, window=14):
    delta = df["Close"].diff()
    gain  = delta.where(delta > 0, 0).rolling(window).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(window).mean()
    return 100 - (100 / (1 + gain/loss))

def ma_crossover(df, fast=20, slow=50):
    return (df["Close"].rolling(fast).mean() - df["Close"].rolling(slow).mean()) / df["Close"].rolling(slow).mean() * 100

@st.cache_data(show_spinner=False, ttl=43200)
def fetch_raw_data(start="2005-01-01", end="2024-12-31"):
    spy = yf.download("SPY", start=start, end=end, progress=False, auto_adjust=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    return spy[~spy.index.duplicated(keep="first")].sort_index()

@st.cache_data(show_spinner=False)
def build_features(spy):
    df = spy.copy()
    df["log_return"]    = np.log(df["Close"] / df["Close"].shift(1))
    df["momentum"]      = df["log_return"].rolling(5).mean()
    df["volume_zscore"] = (df["Volume"] - df["Volume"].rolling(20).mean()) / df["Volume"].rolling(20).std()
    df["parkinson_vol"] = parkinson_volatility(df, 20)
    df["rsi"]           = rsi_calc(df, 14)
    df["ma_cross"]      = ma_crossover(df, 20, 50)
    return df.dropna(subset=FEATURES)

def label_regimes(model):
    s = np.argsort(model.means_[:, 0])
    return {int(s[0]): "Bear", int(s[1]): "Chop", int(s[-1]): "Bull"}

@st.cache_data(show_spinner=False)
def run_walk_forward(cache_key, master):
    n  = len(master)
    fa = master[FEATURES].values
    oos_results = []
    for step in range((n - TRAIN_WINDOW) // STEP_SIZE):
        te  = TRAIN_WINDOW + step * STEP_SIZE
        oe  = min(te + STEP_SIZE, n)
        if oe <= te: break
        td, od = fa[:te], fa[te:oe]
        lo, hi = np.percentile(td, 1, axis=0), np.percentile(td, 99, axis=0)
        sc = StandardScaler()
        ts = sc.fit_transform(np.clip(td, lo, hi))
        os_ = sc.transform(np.clip(od, lo, hi))
        try:
            m = GaussianHMM(n_components=N_STATES, covariance_type=COV_TYPE, n_iter=N_ITER, random_state=RANDOM_STATE, verbose=False)
            m.fit(ts)
            states = m.predict(os_)
        except Exception:
            continue
        rm = label_regimes(m)
        chunk = master.iloc[te:oe][FEATURES + ["Close"]].copy()
        chunk["state"]  = states
        chunk["regime"] = [rm[s] for s in states]
        oos_results.append(chunk)
    return pd.concat(oos_results).rename(columns={"Close": "close"})

def smooth_regimes(regimes, window=20):
    regimes = np.asarray(regimes)
    out = []
    for i in range(len(regimes)):
        s, e = max(0, i-window//2), min(len(regimes), i+window//2)
        vals, cnts = np.unique(regimes[s:e], return_counts=True)
        out.append(vals[np.argmax(cnts)])
    return out

def enforce_min_duration(regimes, min_days=15):
    regimes = list(regimes)
    result = regimes.copy()
    i = 0
    while i < len(regimes):
        j = i
        while j < len(regimes) and regimes[j] == regimes[i]: j += 1
        if j - i < min_days:
            pr = result[i-1] if i > 0 else (regimes[j] if j < len(regimes) else regimes[i])
            for k in range(i, j): result[k] = pr
        i = j
    return result

@st.cache_data(show_spinner=False)
def apply_dwell_filter(oos_df, min_days, smooth_window):
    df = oos_df.copy()
    tmp = smooth_regimes(df["regime"].values, window=smooth_window)
    tmp = enforce_min_duration(tmp, min_days=min_days)
    df["regime_smooth"] = enforce_min_duration(tmp, min_days=min_days)
    return df

@st.cache_data(show_spinner=False)
def run_backtest(oos_df, bull_cap, chop_cap, bear_exposure):
    df = oos_df.copy()
    # Fixed position sizing — no volatility targeting
    def size(row):
        r = row["regime_smooth"]
        if r == "Bull":  return bull_cap
        if r == "Chop":  return chop_cap
        return bear_exposure  # Bear
    df["raw_position"]       = df.apply(size, axis=1)
    df["position"]           = df["raw_position"].shift(1).fillna(0)
    df["strategy_gross_ret"] = df["position"] * df["log_return"]
    df["turnover"]           = df["position"].diff().abs().fillna(0)
    df["trans_cost"]         = TRANS_COST_BPS * df["turnover"]
    df["strategy_net_ret"]   = df["strategy_gross_ret"] - df["trans_cost"]
    df["bah_ret"]            = df["log_return"]
    df["strategy_cum"]       = (1 + df["strategy_net_ret"]).cumprod()
    df["bah_cum"]            = (1 + df["bah_ret"]).cumprod()
    return df

def dd_series(cum): return (cum - cum.cummax()) / cum.cummax() * 100

def compute_metrics(rets, cum):
    n = len(rets)
    if n == 0: return dict(ann_return=float("nan"), ann_vol=float("nan"), sharpe=float("nan"), mdd=float("nan"))
    ar = (cum.iloc[-1] ** (ANNUALIZATION/n)) - 1
    av = rets.std() * np.sqrt(ANNUALIZATION)
    return dict(ann_return=ar, ann_vol=av, sharpe=ar/av if av else float("nan"), mdd=dd_series(cum).min())

@st.cache_data(show_spinner=False)
def load_cross_asset_data():
    dfs = {}
    for asset, file in CROSS_ASSET_FILES.items():
        try:
            df = pd.read_csv(file, index_col=0, parse_dates=True)
            df["drawdown"]     = (df["strategy_cum"] / df["strategy_cum"].cummax()) - 1
            df["bah_drawdown"] = (df["bah_cum"] / df["bah_cum"].cummax()) - 1
            dfs[asset] = df
        except FileNotFoundError:
            pass
    return dfs

def add_regime_shading(fig, df, show_regimes, col="regime_smooth", row=None, col_=None):
    prev_date, prev_regime = df.index[0], df[col].iloc[0]
    for i in range(1, len(df)):
        cr, cd = df[col].iloc[i], df.index[i]
        if cr != prev_regime or i == len(df)-1:
            if prev_regime in show_regimes:
                kw = dict(x0=prev_date, x1=cd, fillcolor=REGIME_COLORS_SOFT[prev_regime], opacity=1, line_width=0, layer="below")
                if row and col_: fig.add_vrect(**kw, row=row, col=col_)
                else: fig.add_vrect(**kw)
            prev_date, prev_regime = cd, cr

def apply_layout(fig, **kwargs):
    fig.update_layout(**{**PLOTLY_BASE, **kwargs})
    return fig

# SIDEBAR
st.sidebar.markdown("## Controls")
st.sidebar.markdown("**Regime Smoothing**")
min_days      = st.sidebar.slider("Min regime duration (days)", 5, 60, 20, step=5)
smooth_window = st.sidebar.slider("Smoothing window (days)", 5, 80, 20, step=5)
st.sidebar.markdown("---")
st.sidebar.markdown("**Strategy Exposure Caps**")
bull_cap      = st.sidebar.slider("Bull cap", 0.0, 3.0, 2.0, step=0.1)
chop_cap      = st.sidebar.slider("Chop cap", 0.0, 3.0, 1.5, step=0.1)
bear_exposure = st.sidebar.slider("Bear exposure", 0.0, 1.0, 0.6, step=0.1)
st.sidebar.markdown("---")
show_regimes  = st.sidebar.multiselect("Regimes to shade", ["Bull","Chop","Bear"], default=["Bull","Chop","Bear"])
st.sidebar.markdown("---")
st.sidebar.caption("IIT Indore · SPY 2005-2024 · Walk-forward Gaussian HMM · 3 states")

# PIPELINE
st.markdown(f"""
<div style="margin-bottom:1rem;">
  <div style="font-size:0.7rem;font-weight:700;letter-spacing:0.15em;text-transform:uppercase;color:{ACCENT};margin-bottom:0.2rem;">IIT Indore · Market Intelligence</div>
  <h1 style="font-size:1.9rem;margin:0;line-height:1.2;">HMM Market Regime Dashboard</h1>
  <div style="color:{MUTED};font-size:0.85rem;margin-top:0.3rem;">Hidden Markov Model · SPY ETF · Walk-Forward Validated · Cross-Asset Analysis</div>
</div>
""", unsafe_allow_html=True)

with st.spinner("Loading market data..."):
    raw     = fetch_raw_data()
    feat_df = build_features(raw)

with st.spinner("Running walk-forward HMM (cached after first run)..."):
    oos_df = run_walk_forward("spy_2005_2024_v3", feat_df)

oos_df = apply_dwell_filter(oos_df, min_days=min_days, smooth_window=smooth_window)
oos_df = run_backtest(oos_df, bull_cap=bull_cap, chop_cap=chop_cap, bear_exposure=bear_exposure)

min_date, max_date = oos_df.index.min().date(), oos_df.index.max().date()
date_range = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
view = oos_df.loc[str(date_range[0]):str(date_range[1])].copy() if isinstance(date_range, tuple) and len(date_range)==2 else oos_df.copy()
if len(view) < 5: view = oos_df.copy()

clean   = view.dropna(subset=["strategy_net_ret","bah_ret"])
strat_m = compute_metrics(clean["strategy_net_ret"], clean["strategy_cum"].dropna())
bah_m   = compute_metrics(clean["bah_ret"], clean["bah_cum"].dropna())

# KPI ROW
c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("HMM Ann. Return",  f"{strat_m['ann_return']:.1%}", f"{strat_m['ann_return']-bah_m['ann_return']:+.1%} vs B&H")
c2.metric("HMM Sharpe",       f"{strat_m['sharpe']:.2f}",    f"{strat_m['sharpe']-bah_m['sharpe']:+.2f} vs B&H")
c3.metric("HMM Max Drawdown", f"{strat_m['mdd']:.1f}%",      f"{strat_m['mdd']-bah_m['mdd']:+.1f}% vs B&H")
c4.metric("B&H Ann. Return",  f"{bah_m['ann_return']:.1%}")
c5.metric("Current Regime",   view["regime_smooth"].iloc[-1])
st.markdown("---")

# MAIN TABS
tab1, tab2, tab3 = st.tabs([
    "📈  SPY Single-Asset Analysis",
    "🌐  Cross-Asset Analysis",
    "🔮  Next Day Prediction",
])

# TAB 1 - SPY SINGLE ASSET
with tab1:

    st.markdown('<div class="section-tag">Price History</div>', unsafe_allow_html=True)
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=view.index, y=view["close"], mode="lines", line=dict(color=TEXT, width=1.4), name="SPY Close"))
    add_regime_shading(fig1, view, show_regimes)
    for ds, lbl in {"2008-09-15":"2008 Crisis","2020-03-23":"COVID Crash","2022-01-03":"2022 Bear"}.items():
        d = pd.Timestamp(ds)
        if view.index.min() <= d <= view.index.max():
            fig1.add_vline(x=d, line=dict(color="#cc4444", dash="dot", width=1))
            fig1.add_annotation(x=d, y=1.02, yref="paper", text=lbl, showarrow=False, font=dict(color="#cc4444", size=10))
    apply_layout(fig1, height=440, yaxis_title="SPY Price (USD)",
                 title=dict(text="SPY Price History with HMM Detected Regimes", font=dict(color=TEXT, size=14)))
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown('<div class="section-tag">Cumulative Returns</div>', unsafe_allow_html=True)
    fig2 = go.Figure()
    add_regime_shading(fig2, view, show_regimes)
    fig2.add_trace(go.Scatter(x=view.index, y=view["strategy_cum"], mode="lines", line=dict(color=ACCENT, width=2), name="HMM Strategy"))
    fig2.add_trace(go.Scatter(x=view.index, y=view["bah_cum"], mode="lines", line=dict(color="#ef5b5b", width=2), name="Buy & Hold SPY"))
    apply_layout(fig2, height=440, yaxis_title="Growth of $1",
                 title=dict(text="Cumulative Returns: HMM Strategy vs Buy & Hold", font=dict(color=TEXT, size=14)),
                 legend=dict(orientation="h", y=1.08, bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-tag">Drawdown</div>', unsafe_allow_html=True)
    fig3 = go.Figure()
    add_regime_shading(fig3, view, show_regimes)
    fig3.add_trace(go.Scatter(x=view.index, y=dd_series(view["strategy_cum"]), mode="lines", fill="tozeroy",
                               line=dict(color=ACCENT, width=1.2), fillcolor="rgba(79,209,197,0.2)", name="HMM Strategy"))
    fig3.add_trace(go.Scatter(x=view.index, y=dd_series(view["bah_cum"]), mode="lines", fill="tozeroy",
                               line=dict(color="#ef5b5b", width=1.2), fillcolor="rgba(239,91,91,0.15)", name="Buy & Hold SPY"))
    apply_layout(fig3, height=380, yaxis_title="Drawdown (%)",
                 title=dict(text="Drawdown Comparison", font=dict(color=TEXT, size=14)),
                 legend=dict(orientation="h", y=1.08, bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<div class="section-tag">Feature Analysis</div>', unsafe_allow_html=True)
    col_l, col_r = st.columns([1,3])
    with col_l:
        selected_feature = st.selectbox("Feature", FEATURES, format_func=lambda f: FEATURE_LABELS[f])
    with col_r:
        fig4 = go.Figure()
        for regime in ["Bull","Chop","Bear"]:
            data = view.loc[view["regime_smooth"]==regime, selected_feature].dropna()
            if data.empty: continue
            data = data.clip(data.quantile(0.02), data.quantile(0.98))
            fig4.add_trace(go.Histogram(x=data, name=regime, opacity=0.55, histnorm="probability density",
                                         marker_color=REGIME_COLORS[regime], nbinsx=40))
        apply_layout(fig4, height=340, barmode="overlay", xaxis_title="Value", yaxis_title="Density",
                     title=dict(text=f"Feature Distribution per Regime - {FEATURE_LABELS[selected_feature]}", font=dict(color=TEXT, size=13)),
                     legend=dict(orientation="h", y=1.1, bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig4, use_container_width=True)

    fig5 = go.Figure()
    add_regime_shading(fig5, view, show_regimes)
    fig5.add_trace(go.Scatter(x=view.index, y=view[selected_feature], mode="lines",
                               line=dict(color=TEXT, width=1), name=FEATURE_LABELS[selected_feature]))
    fig5.add_hline(y=0, line=dict(color=MUTED, dash="dot", width=1))
    apply_layout(fig5, height=340, yaxis_title=FEATURE_LABELS[selected_feature], showlegend=False,
                 title=dict(text=f"Feature Over Time - {FEATURE_LABELS[selected_feature]}", font=dict(color=TEXT, size=13)))
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown('<div class="section-tag">Transition Matrix</div>', unsafe_allow_html=True)
    labels_ord = ["Bull","Chop","Bear"]
    regimes_arr = view["regime_smooth"].values
    trans = np.zeros((3,3))
    for i in range(len(regimes_arr)-1):
        try: trans[labels_ord.index(regimes_arr[i])][labels_ord.index(regimes_arr[i+1])] += 1
        except: pass
    rs = trans.sum(axis=1, keepdims=True); rs[rs==0] = 1
    tp = trans / rs
    fig6 = go.Figure(data=go.Heatmap(
        z=tp, x=labels_ord, y=labels_ord,
        colorscale=[[0,"#ef5b5b"],[0.5,"#f5b942"],[1,"#3ddc84"]],
        zmin=0, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in tp],
        texttemplate="%{text}", textfont=dict(size=18, color="#0b0e14"),
        colorbar=dict(title=dict(text="P", font=dict(color=MUTED)), tickfont=dict(color=MUTED)),
    ))
    fig6.update_yaxes(autorange="reversed")
    apply_layout(fig6, height=400, xaxis_title="To Regime", yaxis_title="From Regime",
                 title=dict(text="Regime Transition Probability Matrix", font=dict(color=TEXT, size=14)))
    st.plotly_chart(fig6, use_container_width=True)
    st.caption("Diagonal = regime persistence. High diagonal means stable regimes, low means frequent switching.")
    st.markdown("---")
    st.caption(f"Showing {len(view):,} OOS trading days - {view.index.min().date()} to {view.index.max().date()} - "
               f"Min dwell: {min_days}d - Walk-forward: {TRAIN_WINDOW}d window / {STEP_SIZE}d step")

# TAB 2 - CROSS-ASSET
with tab2:
    ca_dfs = load_cross_asset_data()

    if not ca_dfs:
        st.warning("Cross-asset CSV files not found. Upload hmm_oos_results_SPY.csv, GLD.csv, TLT.csv, BTC-USD.csv to the same folder as app.py")
    else:
        assets = list(ca_dfs.keys())

        st1, st2, st3, st4, st5 = st.tabs([
            "Cumulative Returns",
            "Regime Timeline",
            "Regime Alignment",
            "Drawdown Comparison",
            "Performance Summary",
        ])

        # Option 1 - Cumulative Returns
        with st1:
            st.markdown('<div class="section-tag">Cross-Asset Cumulative Returns</div>', unsafe_allow_html=True)
            fig_cr = go.Figure()
            for asset, df in ca_dfs.items():
                color = ASSET_COLORS[asset]
                fig_cr.add_trace(go.Scatter(x=df.index, y=(df["strategy_cum"]-1)*100, name=f"{asset} HMM",
                    line=dict(color=color, width=2),
                    hovertemplate=f"<b>{asset} HMM</b><br>%{{x|%b %d %Y}}<br>Return: %{{y:.1f}}%<extra></extra>"))
                fig_cr.add_trace(go.Scatter(x=df.index, y=(df["bah_cum"]-1)*100, name=f"{asset} B&H",
                    line=dict(color=color, width=1, dash="dash"), opacity=0.5,
                    hovertemplate=f"<b>{asset} B&H</b><br>%{{x|%b %d %Y}}<br>Return: %{{y:.1f}}%<extra></extra>"))
            apply_layout(fig_cr, height=500, yaxis_title="Cumulative Return (%)",
                         title=dict(text="Cross-Asset: HMM Strategy vs Buy & Hold Returns", font=dict(color=TEXT, size=14)),
                         legend=dict(orientation="h", y=-0.15, bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig_cr, use_container_width=True)

        # Option 2 - Regime Timeline
        with st2:
            st.markdown('<div class="section-tag">Cross-Asset Regime Timeline</div>', unsafe_allow_html=True)
            n_a = len(assets)
            fig_rt = make_subplots(rows=n_a, cols=1, shared_xaxes=True,
                                    subplot_titles=assets, vertical_spacing=0.04)
            for i, (asset, df) in enumerate(ca_dfs.items(), start=1):
                rc = "regime" if "regime" in df.columns else [c for c in df.columns if "regime" in c][0]
                for regime, val, color in [("Bull",1,REGIME_COLORS["Bull"]),("Chop",0,REGIME_COLORS["Chop"]),("Bear",-1,REGIME_COLORS["Bear"])]:
                    mask = df[rc] == regime
                    fig_rt.add_trace(go.Scatter(
                        x=df.index[mask], y=[val]*mask.sum(), mode="markers",
                        marker=dict(color=color, size=3, symbol="square"),
                        name=regime, legendgroup=regime, showlegend=(i==1),
                        hovertemplate=f"<b>{asset} - {regime}</b><br>%{{x|%b %d %Y}}<extra></extra>",
                    ), row=i, col=1)
                fig_rt.update_yaxes(tickvals=[-1,0,1], ticktext=["Bear","Chop","Bull"],
                                     tickfont=dict(size=9, color=MUTED), row=i, col=1)
                fig_rt.update_xaxes(gridcolor=GRID, linecolor=GRID, tickfont=dict(color=MUTED), row=i, col=1)
            fig_rt.update_layout(**{**PLOTLY_BASE, "height": 80*n_a+120,
                                    "title": dict(text="Regime Timeline Across Assets", font=dict(color=TEXT, size=14))})
            for ann in fig_rt.layout.annotations: ann.font.color = MUTED
            st.plotly_chart(fig_rt, use_container_width=True)

        # Option 3 - Regime Alignment
        with st3:
            st.markdown('<div class="section-tag">Regime Alignment Heatmap</div>', unsafe_allow_html=True)
            regime_series = {}
            for asset, df in ca_dfs.items():
                rc = "regime" if "regime" in df.columns else [c for c in df.columns if "regime" in c][0]
                regime_series[asset] = df[[rc]].rename(columns={rc: asset})
            combined = pd.concat(regime_series.values(), axis=1).dropna()
            st.caption(f"Common trading days across all assets: {len(combined):,}")

            am = pd.DataFrame(index=assets, columns=assets, dtype=float)
            for a1 in assets:
                for a2 in assets:
                    am.loc[a1,a2] = round((combined[a1]==combined[a2]).sum()/len(combined)*100, 1)

            fig_am = go.Figure(data=go.Heatmap(
                z=am.values.astype(float), x=assets, y=assets,
                colorscale=[[0,"#0b0e14"],[0.3,"#1e3a5f"],[0.7,"#1d4ed8"],[1,"#3ddc84"]],
                text=[[f"{v:.1f}%" for v in row] for row in am.values.astype(float)],
                texttemplate="%{text}", textfont=dict(size=16, family="IBM Plex Mono", color=TEXT),
                zmin=0, zmax=100,
                colorbar=dict(title=dict(text="% Same Regime", font=dict(color=MUTED)),
                              tickfont=dict(color=MUTED), ticksuffix="%"),
                hovertemplate="%{y} vs %{x}<br>Same Regime: %{text}<extra></extra>",
            ))
            apply_layout(fig_am, height=380,
                         title=dict(text="% of Days Assets Share Same Regime", font=dict(color=TEXT, size=14)),
                         xaxis=dict(tickfont=dict(color=TEXT, size=13)),
                         yaxis=dict(tickfont=dict(color=TEXT, size=13)))
            st.plotly_chart(fig_am, use_container_width=True)

            if "SPY" in combined.columns:
                st.markdown('<div class="section-tag">Conditional: Given SPY Regime, What Are Others In?</div>', unsafe_allow_html=True)
                other_assets = [a for a in assets if a != "SPY"]
                regimes_list = ["Bull","Chop","Bear"]
                fig_cond = make_subplots(rows=1, cols=3,
                    subplot_titles=[f"When SPY is {r}" for r in regimes_list],
                    horizontal_spacing=0.08)
                for ci, spy_regime in enumerate(regimes_list, start=1):
                    mask    = combined["SPY"] == spy_regime
                    subset  = combined[mask]
                    n_days  = len(subset)
                    z_vals, y_labels = [], []
                    for other in other_assets:
                        row_v = [(subset[other]==r).sum()/n_days*100 if n_days>0 else 0 for r in regimes_list]
                        z_vals.append(row_v)
                        y_labels.append(other)
                    fig_cond.add_trace(go.Heatmap(
                        z=z_vals, x=regimes_list, y=y_labels,
                        colorscale=[[0,"#0b0e14"],[0.5,"#1e3a5f"],[1,REGIME_COLORS[spy_regime]]],
                        text=[[f"{v:.0f}%" for v in row] for row in z_vals],
                        texttemplate="%{text}", textfont=dict(size=13, family="IBM Plex Mono", color=TEXT),
                        showscale=False, zmin=0, zmax=100,
                        hovertemplate="%{y} is %{x}: %{text}<extra></extra>",
                    ), row=1, col=ci)
                fig_cond.update_layout(**{**PLOTLY_BASE, "height": 320,
                                          "title": dict(text="Conditional Regime Distribution (Anchor: SPY)", font=dict(color=TEXT, size=13))})
                for ann in fig_cond.layout.annotations: ann.font.color = MUTED
                st.plotly_chart(fig_cond, use_container_width=True)

        # Option 4 - Drawdown Comparison
        with st4:
            st.markdown('<div class="section-tag">Cross-Asset Drawdown Comparison</div>', unsafe_allow_html=True)
            n_a = len(ca_dfs)
            subplot_titles = [f"{a} - {ASSET_LABELS.get(a, a)}" for a in ca_dfs]
            fig_dd = make_subplots(rows=n_a, cols=1, shared_xaxes=True,
                                    subplot_titles=subplot_titles, vertical_spacing=0.06)
            for i, (asset, df) in enumerate(ca_dfs.items(), start=1):
                color = ASSET_COLORS[asset]
                r, g, b = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)
                fig_dd.add_trace(go.Scatter(
                    x=df.index, y=df["drawdown"]*100, name=f"{asset} HMM",
                    line=dict(color=color, width=1.5), fill="tozeroy",
                    fillcolor=f"rgba({r},{g},{b},0.15)", legendgroup=asset,
                    hovertemplate=f"<b>{asset} Strategy</b><br>%{{x|%b %d %Y}}<br>DD: %{{y:.1f}}%<extra></extra>",
                ), row=i, col=1)
                fig_dd.add_trace(go.Scatter(
                    x=df.index, y=df["bah_drawdown"]*100, name=f"{asset} B&H",
                    line=dict(color="#64748b", width=1, dash="dash"), legendgroup=f"{asset}_bah",
                    hovertemplate=f"<b>{asset} B&H</b><br>%{{x|%b %d %Y}}<br>DD: %{{y:.1f}}%<extra></extra>",
                ), row=i, col=1)
                max_dd_date = df["drawdown"].idxmin()
                max_dd_val  = df["drawdown"].min() * 100
                fig_dd.add_annotation(x=max_dd_date, y=max_dd_val, text=f"HMM: {max_dd_val:.1f}%",
                                       showarrow=True, arrowhead=2, arrowcolor=color,
                                       font=dict(color=color, size=10, family="IBM Plex Mono"),
                                       bgcolor=PANEL_BG, bordercolor=color, borderwidth=1,
                                       ax=40, ay=-30, row=i, col=1)
                fig_dd.update_yaxes(ticksuffix="%", gridcolor=GRID, linecolor=GRID,
                                     tickfont=dict(color=MUTED, family="IBM Plex Mono"), row=i, col=1)
                fig_dd.update_xaxes(gridcolor=GRID, linecolor=GRID,
                                     tickfont=dict(color=MUTED, family="IBM Plex Mono"), row=i, col=1)
            fig_dd.update_layout(**{**PLOTLY_BASE, "height": 900,
                                    "title": dict(text="Cross-Asset Drawdown: HMM Strategy vs Buy & Hold", font=dict(color=TEXT, size=14)),
                                    "legend": dict(orientation="h", y=1.02, x=0.5, xanchor="center", bgcolor="rgba(0,0,0,0)")})
            for ann in fig_dd.layout.annotations: ann.font.color = MUTED
            st.plotly_chart(fig_dd, use_container_width=True)

        # Option 5 - Performance Summary
        with st5:
            st.markdown('<div class="section-tag">Performance Summary - All Assets</div>', unsafe_allow_html=True)
            rows = []
            for asset, df in ca_dfs.items():
                cl = df.dropna(subset=["strategy_net_ret","bah_ret"])
                n  = len(cl)
                if n == 0: continue
                def ann(cum): return (cum.iloc[-1] ** (252/n)) - 1
                def sh(rets, cum): v=rets.std()*np.sqrt(252); return ann(cum)/v if v else float("nan")
                def mdd(cum): return ((cum/cum.cummax())-1).min()*100
                rows.append({
                    "Asset": asset,
                    "HMM Ann. Return": f"{ann(cl['strategy_cum']):.1%}",
                    "B&H Ann. Return": f"{ann(cl['bah_cum']):.1%}",
                    "HMM Sharpe": f"{sh(cl['strategy_net_ret'],cl['strategy_cum']):.2f}",
                    "B&H Sharpe": f"{sh(cl['bah_ret'],cl['bah_cum']):.2f}",
                    "HMM Max DD": f"{mdd(cl['strategy_cum']):.1f}%",
                    "B&H Max DD": f"{mdd(cl['bah_cum']):.1f}%",
                })
            st.dataframe(pd.DataFrame(rows).set_index("Asset"), use_container_width=True)

            fig_ps = go.Figure()
            for row in rows:
                asset = row["Asset"]
                color = ASSET_COLORS[asset]
                hmm_s = float(row["HMM Sharpe"])
                bah_s = float(row["B&H Sharpe"])
                fig_ps.add_trace(go.Bar(name=f"{asset} HMM", x=[asset], y=[hmm_s],
                                         marker_color=color, opacity=0.9,
                                         text=f"{hmm_s:.2f}", textposition="outside",
                                         textfont=dict(color=color, family="IBM Plex Mono")))
                fig_ps.add_trace(go.Bar(name=f"{asset} B&H", x=[asset], y=[bah_s],
                                         marker_color=color, opacity=0.35,
                                         text=f"{bah_s:.2f}", textposition="outside",
                                         textfont=dict(color=MUTED, family="IBM Plex Mono")))
            apply_layout(fig_ps, height=360, barmode="group", yaxis_title="Sharpe Ratio",
                         title=dict(text="Sharpe Ratio: HMM vs Buy & Hold Across Assets", font=dict(color=TEXT, size=14)),
                         legend=dict(orientation="h", y=-0.15, bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig_ps, use_container_width=True)

# TAB 3 - NEXT DAY PREDICTION
with tab3:
    st.markdown('<div class="section-tag">Next Day Regime Prediction</div>', unsafe_allow_html=True)

    regimes_ord = ["Bull","Chop","Bear"]
    trans_p = pd.crosstab(view["regime_smooth"], view["regime_smooth"].shift(-1), normalize="index")
    trans_p = trans_p.reindex(index=regimes_ord, columns=regimes_ord, fill_value=0)
    last_regime = view["regime_smooth"].iloc[-1]
    next_probs  = trans_p.loc[last_regime].to_dict() if last_regime in trans_p.index else {r: 1/3 for r in regimes_ord}
    predicted   = max(next_probs, key=next_probs.get)
    pred_color  = REGIME_COLORS[predicted]

    col_pred, col_bar = st.columns([1,2])
    with col_pred:
        st.markdown(f"""
        <div style="background:{PANEL_BG};border:1px solid {pred_color};border-radius:14px;
                    padding:1.5rem;text-align:center;box-shadow:0 0 24px {pred_color}22;">
            <div style="font-size:0.7rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;
                        color:{MUTED};margin-bottom:0.5rem;">Next Day Prediction</div>
            <div style="font-size:2.5rem;font-weight:700;color:{pred_color};font-family:'IBM Plex Mono',monospace;">
                {predicted}
            </div>
            <div style="color:{MUTED};font-size:0.8rem;margin-top:0.3rem;">As of {view.index[-1].strftime('%b %d, %Y')}</div>
            <div style="color:{MUTED};font-size:0.75rem;margin-top:0.5rem;">
                Current: <span style="color:{REGIME_COLORS[last_regime]};font-weight:600;">{last_regime}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_bar:
        fig_pred = go.Figure()
        for regime, prob in sorted(next_probs.items(), key=lambda x: -x[1]):
            fig_pred.add_trace(go.Bar(
                x=[prob*100], y=[regime], orientation="h",
                marker_color=REGIME_COLORS[regime], opacity=0.85,
                text=f"{prob:.1%}", textposition="outside",
                textfont=dict(color=REGIME_COLORS[regime], family="IBM Plex Mono", size=14),
                hovertemplate=f"<b>{regime}</b>: {prob:.1%}<extra></extra>",
            ))
        apply_layout(fig_pred, height=220, xaxis_title="Probability (%)", showlegend=False,
                     xaxis=dict(range=[0,110], ticksuffix="%"),
                     title=dict(text="Next Day Regime Probabilities", font=dict(color=TEXT, size=13)))
        st.plotly_chart(fig_pred, use_container_width=True)

    st.markdown('<div class="section-tag">Transition Probability Matrix</div>', unsafe_allow_html=True)
    fig_tr = go.Figure(data=go.Heatmap(
        z=trans_p.values, x=regimes_ord, y=regimes_ord,
        colorscale=[[0,"#ef5b5b"],[0.5,"#f5b942"],[1,"#3ddc84"]],
        zmin=0, zmax=1,
        text=[[f"{v:.1%}" for v in row] for row in trans_p.values],
        texttemplate="%{text}", textfont=dict(size=16, color="#0b0e14"),
        colorbar=dict(title=dict(text="Prob", font=dict(color=MUTED)), tickfont=dict(color=MUTED), tickformat=".0%"),
        hovertemplate="From: %{y}<br>To: %{x}<br>P: %{text}<extra></extra>",
    ))
    fig_tr.update_yaxes(autorange="reversed")
    apply_layout(fig_tr, height=380, xaxis_title="Next Day Regime", yaxis_title="Current Regime",
                 title=dict(text="Regime Transition Matrix (from OOS data)", font=dict(color=TEXT, size=14)))
    st.plotly_chart(fig_tr, use_container_width=True)
    st.caption("Row = current regime. Column = next day regime. Each row sums to 100%.")
    st.markdown("---")
    st.caption(f"Prediction based on empirical transition probabilities from {len(view):,} OOS trading days - IIT Indore - HMM Market Regime Classification")
