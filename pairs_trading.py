"""
Pairs trading backtest.

Two assets that track each other will sometimes drift apart. If the gap
between them is mean-reverting, you can trade it: short the expensive one,
buy the cheap one, wait for them to converge.

The strategy is simple. Most of the care in this script goes into not
fooling myself, because a pairs backtest is unusually easy to fake:

  - The hedge ratio is fitted on a training window and then frozen. The
    backtest only trades the period after it. Fitting the hedge ratio on
    the whole sample and then "backtesting" over that same sample is the
    classic way to manufacture a great Sharpe out of nothing, because you
    are using future prices to decide today's position.

  - The z-score uses a rolling mean and standard deviation, so it only
    ever sees data up to the current day.

  - Data is downloaded from Yahoo. If the download fails the script stops
    rather than silently substituting simulated prices. Synthetic mode
    exists for testing the plumbing but you have to ask for it with
    --synthetic, and it prints a warning on every result.

  - Trades are counted as actual round trips, not as days with a
    non-zero P&L.

Run it:
    python pairs_trading.py                  real data, default pair
    python pairs_trading.py --t1 KO --t2 PEP
    python pairs_trading.py --synthetic      fake data, clearly labelled
"""

import argparse
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


TRADING_DAYS = 252


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------

def load_prices(t1, t2, start, end):
    """Download adjusted closes from Yahoo. Fails loudly if it can't."""
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("yfinance is not installed. Run: pip install yfinance\n"
                 "Or use --synthetic to test with simulated data.")

    data = yf.download([t1, t2], start=start, end=end,
                       auto_adjust=True, progress=False)["Close"].dropna()

    if len(data) < 500:
        sys.exit(f"Only got {len(data)} days of data for {t1}/{t2}. "
                 "Not enough to split into training and test periods.\n"
                 "Check the tickers and your connection.")

    print(f"Downloaded {len(data)} trading days "
          f"({data.index[0].date()} to {data.index[-1].date()})")
    return data[t1], data[t2]


def synthetic_prices(t1, t2, start, end, seed=42):
    """Simulated prices for testing the code, NOT for producing results.

    These two series share a common random walk plus an explicitly
    mean-reverting gap. In other words they are built to be cointegrated,
    so any strategy that trades the gap will look excellent. That is a
    statement about this function, not about the strategy.
    """
    dates = pd.bdate_range(start, end)
    n = len(dates)
    rng = np.random.default_rng(seed)

    common = np.cumsum(rng.normal(0.0003, 0.011, n))
    gap = np.zeros(n)
    for i in range(1, n):
        gap[i] = 0.95 * gap[i - 1] + rng.normal(0, 0.02)

    p1 = pd.Series(50 * np.exp(common + gap / 2), index=dates, name=t1)
    p2 = pd.Series(45 * np.exp(common - gap / 2), index=dates, name=t2)
    return p1, p2


# ----------------------------------------------------------------------
# Fitting the relationship, on training data only
# ----------------------------------------------------------------------

def fit_hedge_ratio(p1_train, p2_train):
    """How many units of asset 2 offset one unit of asset 1.

    Regress log price 1 on log price 2 and take the slope. Log prices
    because the relationship between two similar assets is usually
    proportional rather than additive: a 1% move in one goes with roughly
    a beta% move in the other, regardless of price level.

    np.polyfit with degree 1 is just least squares. No need for a
    regression library to fit a straight line.
    """
    beta, alpha = np.polyfit(np.log(p2_train), np.log(p1_train), 1)
    return beta, alpha


def spread_half_life(spread_train):
    """How long the gap takes to close, in trading days.

    Fit the discrete Ornstein-Uhlenbeck update:

        s(t) - s(t-1) = a + lambda * s(t-1) + noise

    If lambda is negative the spread is pulled back toward its mean, and
    it decays exponentially at rate |lambda|. Same maths as radioactive
    decay, so the half-life is

        t_half = -ln(2) / lambda

    This is the number I actually care about when deciding whether a pair
    is tradeable. A p-value from a cointegration test tells you the gap
    probably closes. The half-life tells you whether it closes fast
    enough to be worth the capital, and whether a 60-day z-score window
    is even the right length.

    Returns infinity if lambda >= 0, meaning the spread isn't reverting.
    """
    s = spread_train.dropna()
    lagged = s.shift(1).dropna()
    delta = s.diff().dropna()
    lagged, delta = lagged.align(delta, join="inner")

    lam, _ = np.polyfit(lagged.values, delta.values, 1)
    if lam >= 0:
        return np.inf
    return -np.log(2) / lam


def adf_pvalue(series):
    """Augmented Dickey-Fuller test for stationarity. Optional extra.

    Null hypothesis is that the series has a unit root, i.e. wanders
    without reverting. A low p-value is evidence against that. Reported
    as a secondary check because the half-life above is more informative
    and much easier to interpret.
    """
    try:
        from statsmodels.tsa.stattools import adfuller
        return adfuller(series.dropna())[1]
    except ImportError:
        return np.nan


# ----------------------------------------------------------------------
# Signal and backtest
# ----------------------------------------------------------------------

def compute_zscore(spread, window):
    """Standardise the spread against its own recent history.

    Rolling, not full-sample. On day t this only uses days t-window to t,
    so there is no peeking at the future. Using the full-sample mean and
    standard deviation here would be a subtle version of the same
    lookahead problem the training split is designed to avoid.
    """
    mean = spread.rolling(window).mean()
    std = spread.rolling(window).std()
    return (spread - mean) / std


def generate_positions(z, entry=2.0, exit_z=0.5):
    """Position from the z-score.

    Spread unusually high  -> short it, expecting it to fall
    Spread unusually low   -> long it, expecting it to rise
    Spread back near zero  -> close out

    Entry and exit thresholds are deliberately different. Entering at 2
    and exiting at 0.5 rather than at 2 in both directions stops the
    position flickering on and off every time the z-score wobbles across
    a single threshold, which would rack up transaction costs for nothing.
    """
    pos = np.zeros(len(z))
    current = 0

    for i, val in enumerate(z.values):
        if np.isnan(val):
            pos[i] = 0
            continue
        if current == 0:
            if val > entry:
                current = -1
            elif val < -entry:
                current = 1
        elif current == 1 and val > -exit_z:
            current = 0
        elif current == -1 and val < exit_z:
            current = 0
        pos[i] = current

    return pd.Series(pos, index=z.index)


def backtest(p1, p2, position, beta, cost_per_leg=0.0005):
    """Daily returns of the strategy.

    Long the spread means long asset 1 and short beta of asset 2, so the
    return is r1 - beta*r2. Position is lagged by one day: we see today's
    z-score at today's close and can only trade from tomorrow.

    Costs are charged on position changes and doubled, because opening a
    spread trade means transacting in both legs.
    """
    r1 = p1.pct_change()
    r2 = p2.pct_change()

    gross = position.shift(1) * (r1 - beta * r2)
    turnover = position.diff().abs()
    costs = turnover * cost_per_leg * 2

    net = (gross - costs).fillna(0)
    equity = (1 + net).cumprod()
    return net, equity


def extract_trades(position, net_returns):
    """Split the return stream into individual round trips.

    A trade runs from the day the position becomes non-zero to the day it
    returns to zero. This matters because the original version of this
    project counted every day with a non-zero P&L as a separate "trade",
    which turned a daily win rate into something that looked like a
    per-trade win rate and inflated the trade count enormously.
    """
    trades = []
    open_idx = None

    for i in range(1, len(position)):
        prev, curr = position.iloc[i - 1], position.iloc[i]
        if prev == 0 and curr != 0:
            open_idx = i
        elif prev != 0 and curr == 0 and open_idx is not None:
            trade_ret = (1 + net_returns.iloc[open_idx:i + 1]).prod() - 1
            trades.append(trade_ret)
            open_idx = None

    return np.array(trades)


def performance(net, equity, trades):
    """Standard performance statistics.

    Sharpe has no risk-free rate subtracted. The strategy is long one
    asset and short another, so it is close to cash-neutral and the
    financing largely cancels. Worth being aware this is a simplification.
    """
    daily = net[net != 0]
    if len(daily) < 2:
        return None

    years = len(equity) / TRADING_DAYS
    total = equity.iloc[-1] - 1
    cagr = equity.iloc[-1] ** (1 / years) - 1
    sharpe = net.mean() / net.std() * np.sqrt(TRADING_DAYS) if net.std() > 0 else 0

    downside = net[net < 0].std()
    sortino = net.mean() / downside * np.sqrt(TRADING_DAYS) if downside > 0 else np.nan

    dd = (equity - equity.cummax()) / equity.cummax()

    return {
        "total_return": total,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": dd.min(),
        "drawdown": dd,
        "n_trades": len(trades),
        "win_rate": (trades > 0).mean() if len(trades) else np.nan,
        "avg_trade": trades.mean() if len(trades) else np.nan,
        "best_trade": trades.max() if len(trades) else np.nan,
        "worst_trade": trades.min() if len(trades) else np.nan,
    }


# ----------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------

def make_plots(p1, p2, spread, z, position, equity, perf, t1, t2,
               split_date, entry, exit_z, filename, synthetic):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5))
    title = f"Pairs trading: {t1} / {t2}"
    if synthetic:
        title += "   [SIMULATED DATA, NOT A REAL RESULT]"
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # Normalised prices, with the training period shaded so it's obvious
    # which part of the sample the hedge ratio came from.
    ax = axes[0, 0]
    ax.plot(p1.index, p1 / p1.iloc[0], label=t1, lw=1.1)
    ax.plot(p2.index, p2 / p2.iloc[0], label=t2, lw=1.1)
    ax.axvspan(p1.index[0], split_date, alpha=0.12, color="grey")
    ax.axvline(split_date, color="black", lw=1, ls="--")
    ax.text(p1.index[0], ax.get_ylim()[1] * 0.97, "  fitted here",
            fontsize=8, va="top")
    ax.set_title("Normalised prices")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(z.index, z, lw=0.8, color="tab:purple")
    for lvl, c in [(entry, "tab:red"), (-entry, "tab:red"),
                   (exit_z, "tab:green"), (-exit_z, "tab:green")]:
        ax.axhline(lvl, color=c, lw=0.9, ls="--" if abs(lvl) == entry else ":")
    ax.axhline(0, color="black", lw=0.6)
    ax.axvline(split_date, color="black", lw=1, ls="--")
    ax.set_ylim(-5, 5)
    ax.set_title(f"Spread z-score (entry +/-{entry}, exit +/-{exit_z})")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(equity.index, equity, color="tab:green", lw=1.4)
    ax.axhline(1, color="black", lw=0.6, ls=":")
    ax.set_title("Equity curve, out-of-sample period only")
    ax.set_ylabel("Growth of 1")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.fill_between(perf["drawdown"].index, perf["drawdown"] * 100, 0,
                    color="tab:red", alpha=0.5)
    ax.set_title(f"Drawdown, worst {perf['max_dd']:.1%}")
    ax.set_ylabel("%")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(filename, dpi=140)
    plt.close(fig)
    print(f"Saved {filename}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def run(t1, t2, start, end, train_frac=0.4, window=60,
        entry=2.0, exit_z=0.5, cost=0.0005, synthetic=False):

    if synthetic:
        print("\n" + "!" * 62)
        print("  SYNTHETIC DATA. These prices were generated to be")
        print("  cointegrated, so good results here mean nothing.")
        print("  Do not quote these numbers anywhere.")
        print("!" * 62)
        p1, p2 = synthetic_prices(t1, t2, start, end)
    else:
        p1, p2 = load_prices(t1, t2, start, end)

    # Split before doing anything else, so there is no way to
    # accidentally use test data in the fitting step.
    split = int(len(p1) * train_frac)
    split_date = p1.index[split]
    p1_train, p2_train = p1.iloc[:split], p2.iloc[:split]

    beta, alpha = fit_hedge_ratio(p1_train, p2_train)
    spread = np.log(p1) - beta * np.log(p2) - alpha
    half_life = spread_half_life(spread.iloc[:split])
    adf_p = adf_pvalue(spread.iloc[:split])

    print(f"\nPair: {t1} / {t2}")
    print("-" * 56)
    print(f"Training period   {p1.index[0].date()} to {split_date.date()}  "
          f"({split} days)")
    print(f"Trading period    {split_date.date()} to {p1.index[-1].date()}  "
          f"({len(p1) - split} days)")
    print(f"Hedge ratio beta  {beta:.4f}   (fitted on training only)")
    print(f"Spread half-life  {half_life:.1f} days")
    print(f"ADF p-value       {adf_p:.4f}"
          if not np.isnan(adf_p) else "ADF p-value       n/a")
    print("-" * 56)

    if half_life > window:
        print(f"NOTE: half-life ({half_life:.0f}d) exceeds the z-score window "
              f"({window}d).\n      The spread reverts more slowly than the "
              "window can see, so\n      signals will be unreliable.")

    z = compute_zscore(spread, window)

    # Trade only after the training split.
    position = generate_positions(z, entry, exit_z)
    position.iloc[:split] = 0

    net, equity_full = backtest(p1, p2, position, beta, cost)
    net_oos = net.iloc[split:]
    equity = (1 + net_oos).cumprod()

    trades = extract_trades(position.iloc[split:], net_oos)
    perf = performance(net_oos, equity, trades)

    if perf is None:
        print("No trades taken. Try a lower entry threshold or a different pair.")
        return None

    print(f"Total return      {perf['total_return']:+.2%}")
    print(f"CAGR              {perf['cagr']:+.2%}")
    print(f"Sharpe            {perf['sharpe']:.2f}")
    print(f"Sortino           {perf['sortino']:.2f}")
    print(f"Max drawdown      {perf['max_dd']:.2%}")
    print(f"Round trips       {perf['n_trades']}")
    print(f"Win rate          {perf['win_rate']:.1%}  (per trade, not per day)")
    print(f"Average trade     {perf['avg_trade']:+.2%}")
    print(f"Best / worst      {perf['best_trade']:+.2%} / {perf['worst_trade']:+.2%}")
    print("-" * 56)

    if perf["n_trades"] < 20:
        print(f"CAUTION: only {perf['n_trades']} trades. With a sample this "
              "small the Sharpe\n         ratio has a very wide error bar. "
              "Treat it as indicative.")

    fname = f"pairs_{t1}_{t2}.png"
    make_plots(p1, p2, spread, z, position, equity, perf, t1, t2,
               split_date, entry, exit_z, fname, synthetic)

    if synthetic:
        print("\nReminder: the above used simulated data and is not a result.")

    return perf


def main():
    ap = argparse.ArgumentParser(description="Pairs trading backtest")
    ap.add_argument("--t1", default="GLD")
    ap.add_argument("--t2", default="SLV")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default="2024-01-01")
    ap.add_argument("--train-frac", type=float, default=0.4,
                    help="Fraction of data used to fit the hedge ratio")
    ap.add_argument("--window", type=int, default=60,
                    help="Rolling window for the z-score, in days")
    ap.add_argument("--entry", type=float, default=2.0)
    ap.add_argument("--exit", dest="exit_z", type=float, default=0.5)
    ap.add_argument("--cost", type=float, default=0.0005,
                    help="Transaction cost per leg, 0.0005 = 5bp")
    ap.add_argument("--synthetic", action="store_true",
                    help="Use simulated data for testing. Not a real result.")
    args = ap.parse_args()

    run(args.t1, args.t2, args.start, args.end, args.train_frac,
        args.window, args.entry, args.exit_z, args.cost, args.synthetic)


if __name__ == "__main__":
    main()
