# Pairs Trading Backtest

Two assets that normally track each other occasionally drift apart. If the gap
between them reverts, you can trade it: short the expensive leg, buy the cheap
one, wait for them to converge.

The strategy is about twenty lines. Most of this repo is about not fooling
myself, because a pairs backtest is unusually easy to fake and the failure mode
produces a beautiful equity curve.

## Results

> **To fill in after running.** Paste the terminal output here, and commit the
> generated `pairs_GLD_SLV.png`. Do not put synthetic-mode numbers in this
> section.

```
python pairs_trading.py --t1 GLD --t2 SLV
```

```
(output goes here)
```

## The three ways this goes wrong, and what I did about each

**Fitting the hedge ratio on data you then trade.** The hedge ratio beta says
how many units of asset 2 offset one unit of asset 1. If you estimate it by
regressing over your whole sample and then backtest across that same sample,
you are placing trades in 2016 using a number computed partly from 2023 prices.
The spread is guaranteed to look well-behaved and mean-reverting because you
constructed it that way with hindsight. It will produce a great Sharpe ratio
and it means nothing.

Here the data is split first. Beta is fitted on the first 40% and frozen, and
the backtest only trades the remaining 60%. The training window is shaded on
the price chart so it's obvious which part is which.

**Standardising against the full sample.** Same problem one level down. The
z-score needs a mean and standard deviation for the spread, and using
full-sample values leaks the future into every signal. This uses a rolling
60-day window, so on any given day it only sees the previous 60 days.

**Silently substituting simulated data.** An earlier version of this project
downloaded prices from Yahoo and quietly fell back to synthetic data if the
download failed. The synthetic generator built two series from a shared random
walk plus an explicitly mean-reverting gap, so the fake data was cointegrated
by construction and any strategy trading the gap looked superb. Worse, the
README claimed the results came from real market data, and there was no way to
tell from the repo which had actually run.

Now the download failing stops the script. Synthetic mode still exists for
testing the code, but you have to pass `--synthetic`, and when it runs it
prints a warning before the results, stamps the chart title, and prints another
warning after. It cannot end up in this README by accident.

## Half-life, and why I prefer it to a p-value

The usual test for whether a pair is tradeable is Engle-Granger cointegration,
which gives you a p-value. I report the ADF p-value as a secondary check, but
the number I actually look at is the half-life of the spread.

Fit the discrete Ornstein-Uhlenbeck update:

```
s(t) - s(t-1) = a + λ·s(t-1) + noise
```

If λ is negative, the spread gets pulled back toward its mean and decays
exponentially. Which makes the half-life the same calculation as radioactive
decay:

```
t½ = -ln(2) / λ
```

A p-value tells you the gap probably closes eventually. The half-life tells you
roughly how long you'll be sitting in the trade, whether that's worth the
capital, and whether a 60-day z-score window is even the right length. If the
half-life comes out longer than the window, the script warns you: the spread is
reverting more slowly than your signal can see, so the z-scores aren't
measuring what you think they are.

## Counting trades properly

The previous version computed `trades = (returns != 0).sum()`, counting every
day with a non-zero P&L as a trade. That inflated the trade count by roughly the
average holding period and turned a daily win rate into something labelled as a
per-trade win rate.

`extract_trades()` now walks the position series and pairs each entry with its
exit, so a round trip is one trade. The win rate, average trade and best/worst
figures are all per round trip.

## Other details worth knowing

**Costs.** Charged on every position change and doubled, because opening a
spread trade means transacting in both legs. Default is 5bp per leg. These are
liquid ETFs and large caps so that's roughly reasonable, but it excludes market
impact and assumes you can always borrow the short leg.

**Sharpe.** No risk-free rate subtracted. The position is long one asset and
short another, so it's close to cash-neutral and the financing largely cancels.
A simplification, but a defensible one.

**Entry at 2, exit at 0.5.** Deliberately asymmetric. If you entered and exited
at the same threshold the position would flicker on and off every time the
z-score wobbled across it, generating costs for nothing.

## The bias I have not fixed

I picked GLD/SLV because gold and silver are the textbook cointegrated pair, and
I picked it knowing how they behaved over the sample. That's selection bias, and
no amount of careful train/test splitting inside a single pair removes it.

A properly honest version would define a universe of candidate pairs in advance,
screen them on training data alone, and trade whatever came out, accepting the
losers along with the winners. What's here tests whether the mechanics work on
a pair I already suspected would work. That's a weaker claim, and it's the claim
I'd make about these results.

Related: with a small number of round trips, the Sharpe ratio has a wide error
bar. The script warns if fewer than 20 trades were taken. Two years of a
strategy that trades a handful of times a year is simply not enough evidence to
distinguish skill from luck.

## Running it

```bash
pip install numpy pandas matplotlib yfinance
pip install statsmodels          # optional, only for the ADF p-value

python pairs_trading.py                        # GLD/SLV, 2015-2024
python pairs_trading.py --t1 KO --t2 PEP
python pairs_trading.py --entry 1.5 --window 30
python pairs_trading.py --synthetic            # test mode, not a result
```

Useful flags: `--train-frac` for the size of the fitting window, `--window` for
the z-score lookback, `--entry` and `--exit` for thresholds, `--cost` for
transaction cost per leg.
