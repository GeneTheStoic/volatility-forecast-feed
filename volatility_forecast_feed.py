"""
volatility_forecast_feed.py - publish an option-implied variance history.

This script publishes the one input the terminal cannot compute for itself:
the option-implied variance the market quoted on each date. It does not fit
or forecast anything. The forecast is produced inside MQL5, from the broker's
own daily bars, because realized variance is not the same quantity on the two
sides of the pipeline. Measured from 2019 onward, the overnight gap carries
59 percent of GLD's daily variance and 14 percent of gold futures'. A
six-and-a-half hour session leaves seventeen hours for news to arrive; a
twenty-three hour session leaves one. A forecast fitted on the fund's bars
would be a forecast for the fund, not for the instrument on the chart.

The implied series is a CBOE volatility index rather than a recovered option
chain. GVZ is built from GLD options, VIX from SPX options, OVX from USO
options and VXN from NDX options. The indices are free, carry years of
history, and keep this script small.

    python volatility_forecast_feed.py --ticker SPY --out vol_spy.json

Reads only. Never trades.
"""
import argparse
import datetime as dt
import json
import math
import sys

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    sys.exit("yfinance is required:  pip install yfinance pandas numpy")


class FeedUnusable(Exception):
    """Raised when the inputs cannot support an honest feed."""


#--- each optionable fund, the CBOE index built from its options, and the
#--- local symbol a reader is most likely to pair it with. The local name is
#--- a hint printed in the feed, never something this script acts on, because
#--- broker symbol names vary and only the reader knows theirs
PAIRS = {
    "GLD": ("^GVZ", "CBOE Gold ETF Volatility Index, built from GLD options",
            "XAUUSD", "GLD ETF proxy for gold, not spot gold or broker XAUUSD"),
    "SPY": ("^VIX", "CBOE Volatility Index, built from SPX options",
            "US500", "SPY ETF proxy for the S&P 500, not a broker index CFD"),
    "USO": ("^OVX", "CBOE Crude Oil ETF Volatility Index, built from USO options",
            "USOIL", "USO holds oil futures; it is not spot crude or broker USOIL"),
    "QQQ": ("^VXN", "CBOE Nasdaq 100 Volatility Index, built from NDX options",
            "US100", "QQQ ETF proxy for the Nasdaq 100, not a broker index CFD"),
}

#--- The published history starts where the research sample starts, so the
#--- engine can fit the reader's bars over the same years the tests covered
#--- rather than over a short recent window. How much of it is used depends
#--- on how much daily history the broker holds
HISTORY_START = "2015-01-01"

#--- The reference diagnostics are measured twice, over the whole history and
#--- over the most recent three years, because a premium measured over three
#--- years and one measured over eleven are not the same claim. Where they
#--- disagree, that is the finding rather than an error
RECENT_DAYS = 756

#--- a volatility index is quoted in annualized percent. The engine wants
#--- annualized variance, because that is what it takes the logarithm of
def index_to_variance(level_pct):
    """
    Convert an annualized volatility index level into annualized variance.

    GVZ at 17.5 means 17.5 percent a year, so the variance is 0.175 squared.
    Assumes the index is quoted in percent rather than as a decimal.
    """
    v = float(level_pct) / 100.0
    return v * v


#--- Garman-Klass, used only for the reference diagnostics this script prints.
#--- The engine computes its own realized side and does not read these
GK_CONST = 2.0 * math.log(2.0) - 1.0


def daily_variance(o, h, l, c, prev_c):
    """
    Overnight jump squared plus the intraday Garman-Klass path.

    Assumes all five prices are positive and consecutive.
    """
    gap = np.log(o / prev_c)
    hl = np.log(h / l)
    co = np.log(c / o)
    gk = 0.5 * hl * hl - GK_CONST * co * co
    return gap * gap, np.clip(gk, 0.0, None)


#--- a volatility index quoted today describes the next thirty calendar days,
#--- which is about twenty-one sessions. Comparing it against the same day's
#--- realized variance would be comparing a forecast against yesterday
FORWARD_SESSIONS = 21


def forward_win_rate(iv_series, rv_series, horizon=FORWARD_SESSIONS):
    """
    How often the implied quote exceeded what the underlying then realized.

    For each date carrying an implied quote, the realized variance over the
    next `horizon` sessions is averaged and annualized, and the two are
    compared. Dates without a full forward window are dropped rather than
    padded, because a partial window understates realized variance and would
    flatter the implied side.

    Returns (fraction, count), or (None, count) when fewer than 100 dates
    overlap, so the caller can report how many there were.

    Assumes both inputs are date-indexed, that iv_series holds raw index
    levels in annualized percent, which are converted to variance here, and
    that rv_series holds daily variance.
    """
    rv = rv_series.sort_index()
    fwd = rv.rolling(horizon).mean().shift(-(horizon - 1)) * 252.0
    above = 0
    n = 0
    for stamp, level in iv_series.items():
        if stamp not in fwd.index:
            continue
        realized = fwd.loc[stamp]
        if realized is None or not np.isfinite(realized) or realized <= 0.0:
            continue
        n += 1
        if index_to_variance(level) > realized:
            above += 1
    if n < 100:
        return None, n
    return float(above) / float(n), n


def download(ticker, days, debug=False, start=None):
    """
    Daily bars for one ticker, oldest first, with a generous lookback.

    Assumes yfinance returns a frame indexed by date with OHLC columns.
    """
    if start is None:
        start = (dt.date.today() - dt.timedelta(days=int(days * 1.8) + 60)).isoformat()
    try:
        df = yf.download(ticker, start=start, progress=False, auto_adjust=False)
    except Exception as reason:
        raise FeedUnusable("download failed for %s: %s" % (ticker, reason))
    if df is None or df.empty:
        raise FeedUnusable("no rows returned for %s" % ticker)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(how="all")
    if debug:
        print("  %-6s %d rows, %s to %s"
              % (ticker, len(df), df.index[0].date(), df.index[-1].date()))
    return df


def build(ticker, debug=False):
    """
    Assemble the implied history and the reference diagnostics for one fund.

    The history is the published object. The diagnostics exist so a reader can
    see what the fund's own bars say and compare that against what the panel
    reports on their chart, which will differ, and should.

    Assumes the ticker is one of the four in PAIRS.
    """
    if ticker not in PAIRS:
        raise FeedUnusable("%s is not one of %s" % (ticker, ", ".join(sorted(PAIRS))))
    index_ticker, index_note, local_hint, under_note = PAIRS[ticker]

    idx = download(index_ticker, RECENT_DAYS, debug, start=HISTORY_START)
    if "Close" not in idx.columns:
        raise FeedUnusable("%s carried no close" % index_ticker)
    iv_full = idx["Close"].dropna()
    iv_full = iv_full[iv_full > 0.0]
    #--- the engine gets the whole history; the recent tail is kept for the
    #--- second measurement of the premium
    iv = iv_full.tail(RECENT_DAYS)
    if len(iv) < 150:
        raise FeedUnusable("%s gave only %d usable closes" % (index_ticker, len(iv)))

    #--- The published history goes out as two parallel comma-separated lists
    #--- rather than one field of pairs, because that is the shape a small
    #--- MQL5 reader handles directly: StringSplit on a comma, then
    #--- StringToTime on a YYYY.MM.DD date. The JSON stays flat either way
    dates = ",".join(stamp.strftime("%Y.%m.%d") for stamp in iv_full.index)
    values = ",".join("%.8f" % index_to_variance(level) for level in iv_full.values)

    #--- reference only: what the fund's own bars say. Downloaded over the long
    #--- diagnostic window, then measured twice, because the answer moves
    und = download(ticker, RECENT_DAYS, debug, start=HISTORY_START)
    need = ("Open", "High", "Low", "Close")
    if any(col not in und.columns for col in need):
        raise FeedUnusable("%s carried incomplete bars" % ticker)
    und = und[list(need)].dropna()
    o = und["Open"].to_numpy(dtype=float)
    h = und["High"].to_numpy(dtype=float)
    l = und["Low"].to_numpy(dtype=float)
    c = und["Close"].to_numpy(dtype=float)
    on, gk = daily_variance(o[1:], h[1:], l[1:], c[1:], c[:-1])
    tot = on + gk
    #--- the dates of the variance series, which the forward comparison needs
    rv = pd.Series(tot, index=und.index[1:])
    rv = rv[rv > 0.0]
    on = pd.Series(on, index=und.index[1:])[rv.index]
    if len(rv) < 150:
        raise FeedUnusable("%s produced only %d usable variance days" % (ticker, len(rv)))

    #--- the fund figures quoted alongside the history describe the recent
    #--- three years, the window the recent premium is measured over
    rv_recent = rv.tail(RECENT_DAYS)
    on_recent = on.tail(RECENT_DAYS)
    overnight_share = float(on_recent.sum() / rv_recent.sum())
    rv_ann = float(rv_recent.mean() * 252.0)
    iv_ann = float(np.mean([index_to_variance(x) for x in iv]))

    #--- The ratio of means above is not the statistic the literature reports,
    #--- and the two can disagree sharply. Selling volatility wins on most days
    #--- and loses enormously on a few, so a handful of spikes lifts the mean of
    #--- realized variance without touching how often implied sat above it. The
    #--- published figure, near 80 percent, is the fraction of dates whose
    #--- implied quote exceeded what the underlying went on to realize. That
    #--- comparison also has to look FORWARD, because a 30-day implied number
    #--- is a statement about the next month, not about the day it was quoted.
    #--- It is measured twice on purpose. A premium is not a constant of the
    #--- instrument, and the two windows are allowed to disagree
    win_recent, n_recent = forward_win_rate(iv, rv)
    win_full, n_full = forward_win_rate(iv_full, rv)

    latest_level = float(iv.iloc[-1])
    feed = {
        "symbol": ticker,
        #--- the reader prints this beside the implied figure, so it names the
        #--- date the quote belongs to rather than the moment of the download
        "asof": iv_full.index[-1].strftime("%Y.%m.%d"),
        "underlying_note": under_note,
        "index": index_ticker.lstrip("^"),
        "index_note": index_note,
        "local_symbol_hint": local_hint,
        "quantity": "annualized variance, risk neutral",
        "iv_days": int(len(iv_full)),
        "iv_first": iv_full.index[0].strftime("%Y-%m-%d"),
        "iv_last": iv_full.index[-1].strftime("%Y-%m-%d"),
        "recent_from": iv.index[0].strftime("%Y-%m-%d"),
        "iv_latest_var": round(index_to_variance(latest_level), 8),
        "iv_latest_vol_pct": round(latest_level, 4),
        "iv_mean_var": round(iv_ann, 8),
        #--- the three figures below describe the FUND, not the reader's
        #--- instrument. They are published so the difference is visible
        #--- rather than surprising when the panel shows something else
        "ref_fund_rv_ann_var": round(rv_ann, 8),
        "ref_fund_rv_vol_pct": round(100.0 * math.sqrt(rv_ann), 4),
        "ref_fund_overnight_share": round(overnight_share, 4),
        #--- two different questions, kept apart on purpose. The ratio asks by
        #--- how much implied exceeded realized on average, in VARIANCE, over
        #--- the recent window; its square root is the matching volatility
        #--- ratio. The win rate asks how often implied exceeded forward
        #--- realized at all, which is the figure the literature quotes
        "ref_premium_var_ratio": round(iv_ann / rv_ann, 4) if rv_ann > 0 else None,
        "ref_premium_win_recent": round(win_recent, 4) if win_recent is not None else None,
        "ref_premium_days_recent": n_recent,
        "ref_premium_win_full": round(win_full, 4) if win_full is not None else None,
        "ref_premium_days_full": n_full,
        "ref_premium_full_from": iv_full.index[0].strftime("%Y-%m-%d"),
        "reliable": "the implied history and its dates",
        "less_reliable": "the fund reference figures, which describe the fund",
        "updated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "%s close, converted to annualized variance" % index_ticker.lstrip("^"),
        "dates": dates,
        "values": values,
    }
    return feed


def main():
    """
    Build one feed, print it without the long history, and write it to --out.

    Assumes one ticker per run. A scheduled job loops over the four tickers
    and calls this once for each, so one failure cannot block the others.
    """
    parser = argparse.ArgumentParser(
        description="Publish an option-implied variance history as flat JSON")
    parser.add_argument("--ticker", default="GLD",
                        help="optionable ETF: GLD, SPY, USO or QQQ")
    parser.add_argument("--out", help="also write the JSON to this file")
    parser.add_argument("--debug", action="store_true", help="show what was downloaded")
    args = parser.parse_args()
    ticker = args.ticker.strip().upper()

    if args.debug:
        print("building the implied variance history:")
    try:
        feed = build(ticker, args.debug)
    except FeedUnusable as reason:
        #--- a thin or failed download is a normal event. Leaving the previous
        #--- feed in place is correct, and exiting with status zero keeps a
        #--- scheduled job green instead of alarming on a quiet morning
        print("feed unusable today: %s" % reason)
        print("the previous feed is left untouched")
        return 0

    #--- the history is long, so the printed object shows everything else and
    #--- reports the history by its shape rather than dumping tens of kilobytes
    shown = dict(feed)
    shown["dates"] = "<%d dates, %s to %s>" % (
        feed["iv_days"], feed["iv_first"], feed["iv_last"])
    shown["values"] = "<%d annualized variances>" % feed["iv_days"]
    print(json.dumps(shown, indent=2))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(feed, indent=2) + "\n")
        print("\nwritten to %s" % args.out)

    print("\n%s against %s, %d sessions, %s to %s"
          % (feed["symbol"], feed["index"], feed["iv_days"],
             feed["iv_first"], feed["iv_last"]))
    print("implied now %.2f%% a year; the fund itself realized %.2f%% over the recent window"
          % (feed["iv_latest_vol_pct"], feed["ref_fund_rv_vol_pct"]))
    print("fund overnight share %.1f%% over the recent window"
          % (100.0 * feed["ref_fund_overnight_share"]))
    print()
    print("variance premium, measured twice")
    if feed["ref_premium_var_ratio"] is not None:
        print("  ratio of mean variances, recent window  %.3f" % feed["ref_premium_var_ratio"])
    if feed["ref_premium_win_recent"] is not None:
        print("  implied above forward realized, %s to %s   %.1f%% of %d dates"
              % (feed["recent_from"], feed["iv_last"],
                 100.0 * feed["ref_premium_win_recent"], feed["ref_premium_days_recent"]))
    if feed["ref_premium_win_full"] is not None:
        print("  implied above forward realized, %s to %s   %.1f%% of %d dates"
              % (feed["ref_premium_full_from"], feed["iv_last"],
                 100.0 * feed["ref_premium_win_full"], feed["ref_premium_days_full"]))
    if (feed["ref_premium_win_recent"] is not None
            and feed["ref_premium_win_full"] is not None):
        move = 100.0 * (feed["ref_premium_win_recent"] - feed["ref_premium_win_full"])
        if abs(move) >= 5.0:
            print("  the two windows disagree by %+.1f points, so the premium on this"
                  % move)
            print("  instrument is not a constant and should not be quoted as one")
    print()
    print("the panel will show a different realized figure, measured on your broker's bars")
    return 0


if __name__ == "__main__":
    sys.exit(main())
