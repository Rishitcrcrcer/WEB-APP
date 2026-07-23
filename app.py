"""
HMM Market Regime Detection — Interactive Dashboard
Streamlit + Plotly app that reproduces the full research pipeline
(feature engineering -> walk-forward HMM -> regime smoothing -> backtest)
and lets evaluators explore it interactively.
"""


import warnings
warnings.filterwarnings("ignore")

import os

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM

# ----------------------------------------------------------------------
# Page config + theme
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="HMM Regime Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY_BG = "#0b0e14"
PANEL_BG = "#11151d"
GRID = "#242b38"
TEXT = "#e6e9ef"
MUTED = "#8a93a6"
ACCENT = "#4fd1c5"

REGIME_COLORS = {"Bull": "#3ddc84", "Chop": "#f5b942", "Bear": "#ef5b5b"}
REGIME_COLORS_SOFT = {"Bull": "rgba(61,220,132,0.15)", "Chop": "rgba(245,185,66,0.15)", "Bear": "rgba(239,91,91,0.15)"}

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: {PRIMARY_BG}; }}
    section[data-testid="stSidebar"] {{ background-color: {PANEL_BG}; border-right: 1px solid {GRID}; }}
    h1, h2, h3, h4, p, span, label, .stMarkdown {{ color: {TEXT} !important; }}
    div[data-testid="stMetricValue"] {{ color: {ACCENT} !important; font-family: 'IBM Plex Mono', monospace; }}
    div[data-testid="stMetricLabel"] {{ color: {MUTED} !important; }}
    div[data-testid="stMetric"] {{
        background-color: {PANEL_BG}; border: 1px solid {GRID}; border-radius: 8px;
        padding: 12px 16px;
    }}
    .block-container {{ padding-top: 2rem; }}
    hr {{ border-color: {GRID}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

PLOTLY_TEMPLATE = dict(
    paper_bgcolor=PANEL_BG,
    plot_bgcolor=PANEL_BG,
    font=dict(color=TEXT, family="IBM Plex Mono, monospace", size=12),
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    margin=dict(l=10, r=10, t=50, b=10),
)

FEATURES = ["log_return", "volume_zscore", "momentum", "parkinson_vol", "rsi", "ma_cross"]
FEATURE_LABELS = {
    "log_return": "Log Return",
    "volume_zscore": "Volume Z-Score",
    "momentum": "Momentum",
    "parkinson_vol": "Parkinson Volatility",
    "rsi": "RSI",
    "ma_cross": "MA Crossover",
}

TRAIN_WINDOW = 1260
STEP_SIZE = 63
N_STATES = 3
N_ITER = 200
COVARIANCE_TYPE = "diag"
RANDOM_STATE = 42

TARGET_VOL = 0.30
ROLLING_VOL_WINDOW = 20
ANNUALIZATION = 252
TRANS_COST_BPS = 0.0002


# ----------------------------------------------------------------------
# Feature engineering
# ----------------------------------------------------------------------
def parkinson_volatility(df, window=20):
    return np.sqrt(
        (1 / (4 * np.log(2))) * (np.log(df["High"] / df["Low"]) ** 2).rolling(window).mean()
    ) * np.sqrt(252)


def rsi(df, window=14):
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def ma_crossover(df, fast=20, slow=50):
    ma_fast = df["Close"].rolling(fast).mean()
    ma_slow = df["Close"].rolling(slow).mean()
    return (ma_fast - ma_slow) / ma_slow * 100


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
LOCAL_SPY_CSV = os.path.join(DATA_DIR, "spy_2005_2024.csv")


def _load_local_spy_csv(path: str) -> pd.DataFrame:
    """Loads a locally saved SPY OHLCV file. Handles two common export formats:

    1. Plain single-header CSV:
       Date,Open,High,Low,Close,Adj Close,Volume
       2005-01-03,...

    2. yfinance's raw multi-header export (what you get from df.to_csv() on a
       yf.download() result), which has 3 header rows before the data starts:
       Price,Close,High,Low,Open,Volume
       Ticker,SPY,SPY,SPY,SPY,SPY
       Date,,,,,
       2005-01-03,81.17...,...
    """
    with open(path, "r", encoding="utf-8-sig") as f:  # utf-8-sig strips a BOM if present
        first_lines = [f.readline() for _ in range(3)]

    # Loose detection: look for "ticker" as a whole token on line 2, not a strict
    # prefix match, so quoting/BOM/whitespace differences don't break detection.
    line2_tokens = [t.strip().strip('"').strip("'").lower() for t in first_lines[1].split(",")]
    line3_tokens = [t.strip().strip('"').strip("'").lower() for t in first_lines[2].split(",")]
    is_yfinance_multiheader = (line2_tokens[:1] == ["ticker"]) and (line3_tokens[:1] == ["date"])

    if is_yfinance_multiheader:
        # Row 0 = real column names, rows 1-2 = Ticker/Date label rows to skip,
        # first column = the actual date index.
        spy = pd.read_csv(path, skiprows=[1, 2], index_col=0, parse_dates=True, encoding="utf-8-sig")
        spy.index.name = "Date"
    else:
        spy = pd.read_csv(path, index_col=0, parse_dates=True, encoding="utf-8-sig")

    # Belt-and-suspenders: whatever branch ran above, force every OHLCV column to
    # be actually numeric. If any stray non-numeric header/label rows slipped
    # through detection, this coerces them to NaN so they get dropped below,
    # instead of silently poisoning the whole column into string dtype.
    numeric_cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in spy.columns]
    for col in numeric_cols:
        spy[col] = pd.to_numeric(spy[col], errors="coerce")

    spy.index = pd.to_datetime(spy.index, errors="coerce")
    spy = spy[spy.index.notna()]
    spy = spy.dropna(subset=[c for c in ["Open", "High", "Low", "Close"] if c in spy.columns], how="any")

    # Normalize column names/casing in case of stray whitespace etc.
    spy.columns = [str(c).strip() for c in spy.columns]
    return spy


@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)
def fetch_raw_data(start="2005-01-01", end="2024-12-31"):
    # Prefer a bundled local CSV so the app doesn't depend on Yahoo Finance
    # being reachable/un-rate-limited at runtime. Just drop your downloaded
    # file at data/spy_2005_2024.csv - both plain and yfinance-export formats work.
    if os.path.exists(LOCAL_SPY_CSV):
        try:
            spy = _load_local_spy_csv(LOCAL_SPY_CSV)
            required_cols = {"Open", "High", "Low", "Close", "Volume"}
            if required_cols.issubset(spy.columns):
                spy = spy[~spy.index.duplicated(keep="first")].sort_index()
                spy = spy.dropna(how="all")
                if not spy.empty:
                    return spy, None
        except Exception:
            # Fall through to live fetch if the CSV is somehow unreadable/corrupt
            pass

    try:
        spy = yf.download("SPY", start=start, end=end, progress=False, auto_adjust=False)
    except Exception as e:
        return pd.DataFrame(), str(e)
    if spy is None or spy.empty:
        return pd.DataFrame(), (
            "No usable local data file found at data/spy_2005_2024.csv, and the live "
            "yfinance fallback also returned no rows (download may be rate-limited "
            "or blocked on this network). Download SPY data somewhere with normal "
            "internet access and commit the resulting CSV to data/."
        )
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    spy = spy[~spy.index.duplicated(keep="first")].sort_index()
    return spy, None



@st.cache_data(show_spinner=False)
def build_features(spy: pd.DataFrame) -> pd.DataFrame:
    df = spy.copy()
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["momentum"] = df["log_return"].rolling(5).mean()
    df["volume_zscore"] = (df["Volume"] - df["Volume"].rolling(20).mean()) / df["Volume"].rolling(20).std()
    df["parkinson_vol"] = parkinson_volatility(df, window=20)
    df["rsi"] = rsi(df, window=14)
    df["ma_cross"] = ma_crossover(df, fast=20, slow=50)
    df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURES)
    return df


def winsorize_fit(data, lower_pct=1.0, upper_pct=99.0):
    return np.percentile(data, lower_pct, axis=0), np.percentile(data, upper_pct, axis=0)


def winsorize_transform(data, lo, hi):
    return np.clip(data, lo, hi)


def label_regimes(model: GaussianHMM) -> dict:
    mean_returns = model.means_[:, 0]
    sorted_states = np.argsort(mean_returns)
    return {int(sorted_states[0]): "Bear", int(sorted_states[1]): "Chop", int(sorted_states[-1]): "Bull"}


@st.cache_data(show_spinner=False)
def run_walk_forward(master_key: str, master: pd.DataFrame) -> pd.DataFrame:
    """master_key just forces cache invalidation when underlying data changes."""
    n = len(master)
    feature_array = master[FEATURES].values
    n_steps = (n - TRAIN_WINDOW) // STEP_SIZE
    oos_results = []

    for step in range(n_steps):
        train_end = TRAIN_WINDOW + step * STEP_SIZE
        oos_start = train_end
        oos_end = min(oos_start + STEP_SIZE, n)
        if oos_end <= oos_start:
            break

        train_data = feature_array[:train_end]
        oos_data = feature_array[oos_start:oos_end]

        lo, hi = winsorize_fit(train_data)
        train_wins = winsorize_transform(train_data, lo, hi)
        oos_wins = winsorize_transform(oos_data, lo, hi)

        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_wins)
        oos_scaled = scaler.transform(oos_wins)

        try:
            model = GaussianHMM(
                n_components=N_STATES,
                covariance_type=COVARIANCE_TYPE,
                n_iter=N_ITER,
                random_state=RANDOM_STATE,
                verbose=False,
            )
            model.fit(train_scaled)
            oos_states = model.predict(oos_scaled)
        except Exception:
            continue

        regime_map = label_regimes(model)
        oos_regimes = [regime_map[s] for s in oos_states]

        chunk = master.iloc[oos_start:oos_end][FEATURES + ["Close"]].copy()
        chunk["state"] = oos_states
        chunk["regime"] = oos_regimes
        oos_results.append(chunk)

    if oos_results:
        oos_df = pd.concat(oos_results)
        oos_df = oos_df.rename(columns={"Close": "close"})
        return oos_df
    else:
        return pd.DataFrame()  # caller checks for empty and stops with a clear message


# ----------------------------------------------------------------------
# Regime smoothing (min dwell-time filter)
# ----------------------------------------------------------------------
def smooth_regimes(regimes, window=20):
    regimes = np.asarray(regimes)
    smoothed = []
    for i in range(len(regimes)):
        start = max(0, i - window // 2)
        end = min(len(regimes), i + window // 2)
        chunk = regimes[start:end]
        values, counts = np.unique(chunk, return_counts=True)
        smoothed.append(values[np.argmax(counts)])
    return smoothed


def enforce_min_duration(regimes, min_days=15):
    regimes = list(regimes)
    result = regimes.copy()
    i = 0
    while i < len(regimes):
        j = i
        while j < len(regimes) and regimes[j] == regimes[i]:
            j += 1
        duration = j - i
        if duration < min_days:
            prev_regime = result[i - 1] if i > 0 else (regimes[j] if j < len(regimes) else regimes[i])
            for k in range(i, j):
                result[k] = prev_regime
        i = j
    return result


@st.cache_data(show_spinner=False)
def apply_dwell_filter(oos_df: pd.DataFrame, min_days: int, smooth_window: int) -> pd.DataFrame:
    df = oos_df.copy()
    temp = smooth_regimes(df["regime"].values, window=smooth_window)
    temp = enforce_min_duration(temp, min_days=min_days)
    df["regime_smooth"] = enforce_min_duration(temp, min_days=min_days)
    return df


# ----------------------------------------------------------------------
# Position sizing + backtest
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def run_backtest(oos_df: pd.DataFrame, bull_cap: float, chop_cap: float, bear_exposure: float) -> pd.DataFrame:
    df = oos_df.copy()
    df["realized_vol"] = df["log_return"].rolling(ROLLING_VOL_WINDOW, min_periods=5).std() * np.sqrt(ANNUALIZATION)
    df["realized_vol"] = df["realized_vol"].replace(0.0, np.nan).ffill().bfill()
    df["base_exposure"] = TARGET_VOL / df["realized_vol"]

    caps = {"Bull": bull_cap, "Chop": chop_cap}

    def size_position(row):
        base = row["base_exposure"]
        regime = row["regime_smooth"]
        if pd.isna(base):
            return 0.0
        if regime in caps:
            return float(np.clip(base, 0.0, caps[regime]))
        return bear_exposure

    df["raw_position"] = df.apply(size_position, axis=1)
    df["position"] = df["raw_position"].shift(1).fillna(0.0)

    df["strategy_gross_ret"] = df["position"] * df["log_return"]
    df["turnover"] = df["position"].diff().abs().fillna(0.0)
    df["trans_cost"] = TRANS_COST_BPS * df["turnover"]
    df["strategy_net_ret"] = df["strategy_gross_ret"] - df["trans_cost"]
    df["bah_ret"] = df["log_return"]

    df["strategy_cum"] = (1 + df["strategy_net_ret"]).cumprod()
    df["bah_cum"] = (1 + df["bah_ret"]).cumprod()
    return df


def max_drawdown_series(cum_returns: pd.Series) -> pd.Series:
    running_max = cum_returns.cummax()
    return (cum_returns - running_max) / running_max * 100


def compute_metrics(daily_returns, cum_wealth):
    n_days = len(daily_returns)
    if n_days == 0:
        return dict(ann_return=np.nan, ann_vol=np.nan, sharpe=np.nan, mdd=np.nan, calmar=np.nan)
    ann_return = (cum_wealth.iloc[-1] ** (ANNUALIZATION / n_days)) - 1.0
    ann_vol = daily_returns.std() * np.sqrt(ANNUALIZATION)
    sharpe = ann_return / ann_vol if ann_vol else np.nan
    dd = max_drawdown_series(cum_wealth)
    mdd = dd.min()
    calmar = ann_return / abs(mdd / 100) if mdd else np.nan
    return dict(ann_return=ann_return, ann_vol=ann_vol, sharpe=sharpe, mdd=mdd, calmar=calmar)


# ----------------------------------------------------------------------
# Sidebar controls
# ----------------------------------------------------------------------
st.sidebar.markdown("## ⚙️ Controls")

st.sidebar.markdown("**Regime smoothing (dwell-time filter)**")
min_days = st.sidebar.slider("Minimum regime duration (days)", 5, 60, 20, step=5,
                              help="Regimes shorter than this get merged into the surrounding regime, "
                                   "so the model can't flip state every few days.")
smooth_window = st.sidebar.slider("Smoothing window (days)", 5, 80, 20, step=5)

st.sidebar.markdown("---")
st.sidebar.markdown("**Strategy exposure caps**")
bull_cap = st.sidebar.slider("Bull cap", 0.0, 5.0, 3.5, step=0.1)
chop_cap = st.sidebar.slider("Chop cap", 0.0, 5.0, 2.2, step=0.1)
bear_exposure = st.sidebar.slider("Bear exposure", -1.0, 1.0, 0.0, step=0.1)

st.sidebar.markdown("---")
show_regimes = st.sidebar.multiselect("Regimes to shade", ["Bull", "Chop", "Bear"], default=["Bull", "Chop", "Bear"])

st.sidebar.markdown("---")
st.sidebar.caption("Data: SPY & VIX via yfinance · 2005–2024 · Walk-forward Gaussian HMM (3 states)")

# ----------------------------------------------------------------------
# Pipeline execution
# ----------------------------------------------------------------------
st.title("📈 HMM Market Regime Dashboard")
st.caption("Walk-forward Hidden Markov Model regime detection on SPY (2005–2024), with dwell-time smoothing and a regime-aware trading overlay.")

with st.spinner("Loading market data..."):
    raw, fetch_err = fetch_raw_data()

if fetch_err or raw.empty:
    st.error(
        f"Could not load SPY price data from Yahoo Finance: {fetch_err or 'no rows returned'}. "
        "This usually means the download was rate-limited or the network/egress settings are "
        "blocking access to finance.yahoo.com. Fix network access (or try again later) and reload."
    )
    st.stop()

feat_df = build_features(raw)

if feat_df.empty or len(feat_df) <= TRAIN_WINDOW:
    st.error(
        f"Only {len(feat_df)} rows of usable feature data are available, but the walk-forward "
        f"model needs at least {TRAIN_WINDOW} (TRAIN_WINDOW) rows before it can produce any "
        "out-of-sample predictions. Widen the date range in fetch_raw_data() or reduce TRAIN_WINDOW."
    )
    st.stop()

with st.spinner("Running walk-forward HMM (first load only, then cached)..."):
    oos_df = run_walk_forward("spy_2005_2024_v1", feat_df)

if oos_df.empty:
    st.error("The walk-forward HMM produced no out-of-sample results. Check the data and date ranges above.")
    st.stop()

oos_df = apply_dwell_filter(oos_df, min_days=min_days, smooth_window=smooth_window)
oos_df = run_backtest(oos_df, bull_cap=bull_cap, chop_cap=chop_cap, bear_exposure=bear_exposure)

# Date range selector, based on actual OOS index
min_date, max_date = oos_df.index.min().date(), oos_df.index.max().date()
date_range = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_d, end_d = date_range
    view = oos_df.loc[str(start_d):str(end_d)].copy()
else:
    view = oos_df.copy()

if len(view) < 5:
    st.warning("Selected date range is too short — showing full range instead.")
    view = oos_df.copy()

# ----------------------------------------------------------------------
# Top metrics row
# ----------------------------------------------------------------------
clean = view.dropna(subset=["strategy_net_ret", "bah_ret"])
strat_m = compute_metrics(clean["strategy_net_ret"], clean["strategy_cum"].dropna())
bah_m = compute_metrics(clean["bah_ret"], clean["bah_cum"].dropna())

c1, c2, c3, c4 = st.columns(4)
c1.metric("HMM Ann. Return", f"{strat_m['ann_return']:.1%}", f"{(strat_m['ann_return']-bah_m['ann_return']):+.1%} vs B&H")
c2.metric("HMM Sharpe", f"{strat_m['sharpe']:.2f}", f"{(strat_m['sharpe']-bah_m['sharpe']):+.2f} vs B&H")
c3.metric("HMM Max Drawdown", f"{strat_m['mdd']:.1f}%", f"{(strat_m['mdd']-bah_m['mdd']):+.1f}% vs B&H")
c4.metric("Buy & Hold Return", f"{bah_m['ann_return']:.1%}")

st.markdown("---")


def add_regime_shading(fig, df, row=None, col=None, alpha_key="regime_smooth"):
    prev_date = df.index[0]
    prev_regime = df[alpha_key].iloc[0]
    for i in range(1, len(df)):
        cur_regime = df[alpha_key].iloc[i]
        cur_date = df.index[i]
        if cur_regime != prev_regime or i == len(df) - 1:
            if prev_regime in show_regimes:
                fig.add_vrect(
                    x0=prev_date, x1=cur_date,
                    fillcolor=REGIME_COLORS_SOFT[prev_regime], opacity=1, line_width=0,
                    layer="below", row=row, col=col,
                )
            prev_date = cur_date
            prev_regime = cur_regime
    return fig


# ----------------------------------------------------------------------
# Chart 1: SPY Price History with Regime Colors
# ----------------------------------------------------------------------
st.subheader("SPY Price History with Regime Colors")
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=view.index, y=view["close"], mode="lines",
                           line=dict(color=TEXT, width=1.4), name="SPY Close"))
add_regime_shading(fig1, view)

events = {"2008-09-15": "2008 Crisis", "2020-03-23": "COVID Crash", "2022-01-03": "2022 Bear"}
for date_str, label in events.items():
    d = pd.Timestamp(date_str)
    if view.index.min() <= d <= view.index.max():
        fig1.add_vline(x=d, line=dict(color="#cc4444", dash="dot", width=1))
        fig1.add_annotation(x=d, y=1.02, yref="paper", text=label, showarrow=False,
                             font=dict(color="#cc4444", size=10))

fig1.update_layout(**PLOTLY_TEMPLATE, height=460, yaxis_title="SPY Price (USD)")
fig1.update_layout(legend=dict(orientation="h", y=1.08))
st.plotly_chart(fig1, width='stretch')

# ----------------------------------------------------------------------
# Chart 2: Cumulative Returns
# ----------------------------------------------------------------------
st.subheader("Cumulative Returns: HMM Strategy vs Buy & Hold")

fig2, ax2 = plt.subplots(figsize=(16, 7))

# plot both curves
ax2.plot(view.index, view["strategy_cum"],
         color="#e74c3c", linewidth=2, label="Aggressive HMM Strategy", zorder=3)
ax2.plot(view.index, view["bah_cum"],
         color="#2c3e50", linewidth=2, label="Buy & Hold SPY", zorder=3)

# regime background (uses the same date-filtered `view` as every other chart,
# so it always matches whatever range/sliders are currently selected)
regime_colors_mpl = {"Bull": "#90EE90", "Chop": "#FFD700", "Bear": "#FF6B6B"}
prev_date = view.index[0]
prev_regime = view["regime_smooth"].iloc[0]
for i in range(1, len(view)):
    current_regime = view["regime_smooth"].iloc[i]
    current_date = view.index[i]
    if current_regime != prev_regime or i == len(view) - 1:
        if prev_regime in show_regimes:
            ax2.axvspan(prev_date, current_date, alpha=0.15,
                        color=regime_colors_mpl[prev_regime], zorder=1)
        prev_date = current_date
        prev_regime = current_regime

# final value annotations
final_strategy = view["strategy_cum"].dropna().iloc[-1]
final_bah = view["bah_cum"].dropna().iloc[-1]

ax2.annotate(f"Aggressive: {final_strategy:.2f}x\n({(final_strategy-1)*100:.0f}%)",
             xy=(view.index[-1], final_strategy),
             xytext=(-100, 10), textcoords="offset points",
             fontsize=11, fontweight="bold", color="#e74c3c",
             arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.5))

ax2.annotate(f"Buy & Hold: {final_bah:.2f}x\n({(final_bah-1)*100:.0f}%)",
             xy=(view.index[-1], final_bah),
             xytext=(-100, -40), textcoords="offset points",
             fontsize=11, fontweight="bold", color="#2c3e50",
             arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=1.5))

# key events
event_labels = {"2020-03-23": "COVID\nCrash", "2022-01-03": "2022\nBear"}
for date_str, label in event_labels.items():
    d = pd.Timestamp(date_str)
    if view.index.min() <= d <= view.index.max():
        ax2.axvline(x=d, color="gray", linestyle="--", linewidth=1, alpha=0.7)
        ax2.text(d, view["strategy_cum"].max() * 0.95, label,
                  fontsize=9, color="gray", ha="center", style="italic")

# performance box (reuses strat_m, already computed above for the metrics row)
textstr = (
    f"Aggressive Strategy\n"
    f"Total Return: {(final_strategy - 1) * 100:.2f}%\n"
    f"Sharpe Ratio: {strat_m['sharpe']:.2f}\n"
    f"Max Drawdown: {strat_m['mdd']:.2f}%\n"
    f"Calmar Ratio: {strat_m['calmar']:.2f}"
)
props = dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="#e74c3c")
ax2.text(0.02, 0.97, textstr, transform=ax2.transAxes, fontsize=10,
          verticalalignment="top", bbox=props, color="#e74c3c")

# formatting
ax2.set_title("Aggressive HMM Strategy vs Buy & Hold SPY", fontsize=15, fontweight="bold", pad=15)
ax2.set_xlabel("Date", fontsize=12)
ax2.set_ylabel("Cumulative Return (1.0 = Starting Value)", fontsize=12)
ax2.legend(fontsize=12, loc="upper left", framealpha=0.9)
ax2.grid(True, alpha=0.3, linestyle="--")
ax2.set_facecolor("#f8f9fa")
fig2.patch.set_facecolor("white")
fig2.tight_layout()

st.pyplot(fig2)
plt.close(fig2)  # free the figure each rerun so memory doesn't build up across sessions


# ----------------------------------------------------------------------
# Chart 3: Drawdown Comparison
# ----------------------------------------------------------------------
st.subheader("Drawdown Comparison")
strat_dd = max_drawdown_series(view["strategy_cum"])
bah_dd = max_drawdown_series(view["bah_cum"])
fig3 = go.Figure()
add_regime_shading(fig3, view)
fig3.add_trace(go.Scatter(x=view.index, y=strat_dd, mode="lines", fill="tozeroy",
                           line=dict(color=ACCENT, width=1.2), fillcolor="rgba(79,209,197,0.25)", name="HMM Strategy"))
fig3.add_trace(go.Scatter(x=view.index, y=bah_dd, mode="lines", fill="tozeroy",
                           line=dict(color="#ef5b5b", width=1.2), fillcolor="rgba(239,91,91,0.2)", name="Buy & Hold SPY"))
fig3.update_layout(**PLOTLY_TEMPLATE, height=420, yaxis_title="Drawdown (%)")
fig3.update_layout(legend=dict(orientation="h", y=1.08))
st.plotly_chart(fig3, width='stretch')

# ----------------------------------------------------------------------
# Chart 4 & 5: Feature Distribution + Feature Over Time (feature picker)
# ----------------------------------------------------------------------
col_left, col_right = st.columns([1, 3])
with col_left:
    st.markdown("#### Feature explorer")
    selected_feature = st.selectbox("Feature", FEATURES, format_func=lambda f: FEATURE_LABELS[f])

with col_right:
    st.subheader(f"Feature Distribution per Regime — {FEATURE_LABELS[selected_feature]}")
    fig4 = go.Figure()
    for regime in ["Bull", "Chop", "Bear"]:
        data = view.loc[view["regime_smooth"] == regime, selected_feature].dropna()
        if data.empty:
            continue
        lo, hi = data.quantile(0.02), data.quantile(0.98)
        data = data.clip(lo, hi)
        fig4.add_trace(go.Histogram(x=data, name=regime, opacity=0.55, histnorm="probability density",
                                     marker_color=REGIME_COLORS[regime], nbinsx=40))
    fig4.update_layout(**PLOTLY_TEMPLATE, height=380, barmode="overlay",
                        xaxis_title="Value", yaxis_title="Density")
    fig4.update_layout(legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig4, width='stretch')

st.subheader(f"Feature Over Time — {FEATURE_LABELS[selected_feature]}")
fig5 = go.Figure()
add_regime_shading(fig5, view)
fig5.add_trace(go.Scatter(x=view.index, y=view[selected_feature], mode="lines",
                           line=dict(color=TEXT, width=1), name=FEATURE_LABELS[selected_feature]))
fig5.add_hline(y=0, line=dict(color=MUTED, dash="dot", width=1))
fig5.update_layout(**PLOTLY_TEMPLATE, height=380, yaxis_title=FEATURE_LABELS[selected_feature],
                    showlegend=False)
st.plotly_chart(fig5, width='stretch')

# ----------------------------------------------------------------------
# Chart 6: Transition Heatmap
# ----------------------------------------------------------------------
st.subheader("Regime Transition Probability Matrix")
regimes = view["regime_smooth"].values
labels = ["Bull", "Chop", "Bear"]
trans = np.zeros((3, 3))
for i in range(len(regimes) - 1):
    trans[labels.index(regimes[i])][labels.index(regimes[i + 1])] += 1
row_sums = trans.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1
trans_prob = trans / row_sums

fig6 = go.Figure(data=go.Heatmap(
    z=trans_prob, x=labels, y=labels,
    colorscale=[[0, "#ef5b5b"], [0.5, "#f5b942"], [1, "#3ddc84"]],
    zmin=0, zmax=1,
    text=[[f"{v:.2f}" for v in row] for row in trans_prob],
    texttemplate="%{text}", textfont=dict(size=16, color="#0b0e14"),
    colorbar=dict(title="P"),
))
fig6.update_layout(**PLOTLY_TEMPLATE, height=420, xaxis_title="To Regime", yaxis_title="From Regime")
fig6.update_yaxes(autorange="reversed")
st.plotly_chart(fig6, width='stretch')
st.caption("Diagonal = regime persistence. High diagonal values mean stable regimes; low values mean frequent switching.")

st.markdown("---")
st.caption(
    f"Showing {len(view):,} out-of-sample trading days from {view.index.min().date()} to {view.index.max().date()} · "
    f"Regime smoothing: {min_days}-day minimum dwell · Walk-forward window {TRAIN_WINDOW}d / step {STEP_SIZE}d"
)

