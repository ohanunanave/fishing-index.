#!/usr/bin/env python3
"""
Calibrate the fishing index from YOUR logbook (כיול המדד מיומן הדיג שלך).

Key idea: the index is multiplicative,
    E[catches] = hours * exp(b0) * Π_k  s_k ** w_k
so taking logs gives a Poisson log-linear model:
    log E[catches] = log(hours) + b0 + Σ_k w_k * log(s_k)
The GLM coefficients on log(s_k) ARE the factor weights. With your own trips
you estimate them directly instead of trusting anyone's guesses (including mine).

Logbook CSV columns (one row per session; record blank sessions too!):
    date, start, end, beach, species, catches, bites, notes
    2026-11-29, 05:30, 08:00, nitzanim, gumbar, 2, 5, jig 20g silver
  * species = what you targeted (key from SPECIES)
  * bites optional (strikes incl. missed) – often a better signal than landed fish
  * zero-catch sessions are essential – without them nothing can be learned

Usage:
  python calibrate.py logbook.csv            # fetches historical conditions
  python calibrate.py --demo                 # synthetic logbook, no network
Rule of thumb: start trusting the output after ~30-40 sessions; before that the
ridge penalty keeps weights close to the defaults.
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

import fishing_index as fi

FACTORS = list(fi.WEIGHTS)


def session_features(log: pd.DataFrame, demo=False) -> pd.DataFrame:
    """Average log sub-scores over each session's hours."""
    rows = []
    for bkey, g in log.groupby("beach"):
        b = fi.BEACHES[bkey]
        d0 = (g["t0"].min() - pd.Timedelta(days=3)).strftime("%Y-%m-%d")
        d1 = g["t1"].max().strftime("%Y-%m-%d")
        if demo:
            raw = _demo_history(g["t0"].min() - pd.Timedelta(days=3), g["t1"].max())
        else:
            raw = fi.fetch_open_meteo(b["lat"], b["lon"], days=0, start_date=d0, end_date=d1)
        feat = fi.add_features(raw, b)
        for i, r in g.iterrows():
            sub = fi.score_species(feat.loc[r["t0"]:r["t1"]], fi.SPECIES[r["species"]], beach_key=bkey)
            if sub.empty:
                continue
            x = {f"log_{k}": np.log(np.clip(sub[f"s_{k}"].astype(float), 1e-3, 1)).mean() for k in FACTORS}
            x.update(idx=i, unsafe_share=sub["unsafe"].mean())
            rows.append(x)
    X = pd.DataFrame(rows).set_index("idx")
    return log.join(X, how="inner")


def fit(data: pd.DataFrame, target="catches", alpha=1.0):
    """Ridge-penalised Poisson GLM, shrinking weights toward the current defaults
    (implemented by fitting deviations from the prior with an offset)."""
    import statsmodels.api as sm
    warnings.filterwarnings("ignore")

    cols = [f"log_{k}" for k in FACTORS]
    X = data[cols].values
    prior = np.array([fi.WEIGHTS[k] for k in FACTORS])
    offset = np.log(data["hours"].values) + X @ prior          # prior model as offset
    Xc = sm.add_constant(X, has_constant="add")
    model = sm.GLM(data[target].values, Xc, family=sm.families.Poisson(), offset=offset)
    # penalise only the deviations (not the intercept)
    pen = np.r_[0.0, np.full(len(cols), alpha)]
    res = model.fit_regularized(alpha=pen / len(data), L1_wt=0.0, maxiter=500)
    dev = res.params[1:]
    new_w = np.clip(prior + dev, 0, None)
    out = pd.DataFrame({"factor": FACTORS, "prior_weight": prior, "fitted_weight": new_w.round(2),
                        "change": (new_w - prior).round(2)})
    # unpenalised fit for rough significance (only meaningful with enough sessions)
    try:
        plain = sm.GLM(data[target].values, Xc, family=sm.families.Poisson(),
                       offset=np.log(data["hours"].values)).fit()
        out["mle_weight"] = plain.params[1:].round(2)
        out["p_value"] = plain.pvalues[1:].round(3)
    except Exception:
        pass
    return out


def read_log(path: str) -> pd.DataFrame:
    log = pd.read_csv(path, skipinitialspace=True)
    log["t0"] = pd.to_datetime(log["date"] + " " + log["start"]).dt.tz_localize(fi.TZ)
    log["t1"] = pd.to_datetime(log["date"] + " " + log["end"]).dt.tz_localize(fi.TZ)
    log.loc[log["t1"] <= log["t0"], "t1"] += pd.Timedelta(days=1)   # past midnight
    log["hours"] = (log["t1"] - log["t0"]).dt.total_seconds() / 3600
    return log


# ---------------------------------------------------------------- demo helpers
def _demo_history(t0, t1, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(t0.floor("h"), t1.ceil("h"), freq="h")
    n = len(idx)
    t = np.arange(n) / 24
    storms = sum(np.exp(-((t - c) ** 2) / 0.8) for c in rng.uniform(0, t[-1], max(2, n // 24 // 7)))
    df = pd.DataFrame(index=idx)
    df["pressure_msl"] = 1016 - 10 * storms + rng.normal(0, .5, n)
    df["wind_speed_10m"] = 10 + 30 * storms + rng.normal(0, 3, n)
    df["wind_gusts_10m"] = df["wind_speed_10m"] * 1.5
    df["wind_direction_10m"] = np.where(storms > .2, 260, 100)
    df["cloud_cover"] = 30.0
    df["precipitation"] = np.clip(2 * storms - .5, 0, None)
    df["temperature_2m"] = 18.0
    df["wave_height"] = np.clip(.4 + 2 * np.convolve(storms, np.ones(12) / 12, "same"), .1, None)
    df["wave_period"] = 6.0
    df["wave_direction"] = 280.0
    m = idx.month.values
    df["sea_surface_temperature"] = 22 + 5 * np.cos(2 * np.pi * (m - 8) / 12)
    df["sea_level_height_msl"] = .15 * np.sin(2 * np.pi * t / (12.42 / 24))
    df["ocean_current_velocity"] = .6 + storms
    return df


def demo_log(n=80, seed=11) -> pd.DataFrame:
    """Synthetic sessions where truth = strong light & waves effect, no pressure effect."""
    rng = np.random.default_rng(seed)
    days = pd.date_range("2026-01-01", "2026-11-15", freq="D")
    pick = rng.choice(days, n, replace=False)
    starts = rng.choice(["05:00", "06:00", "10:00", "13:00", "16:00", "19:00", "22:00"], n)
    log = pd.DataFrame({"date": pd.DatetimeIndex(pick).strftime("%Y-%m-%d"), "start": starts,
                        "beach": "nitzanim", "species": "gumbar"})
    log["end"] = (pd.to_datetime(log["start"], format="%H:%M") + pd.Timedelta(hours=2)).dt.strftime("%H:%M")
    log = read_log_df(log)
    feats = session_features(log.assign(catches=0), demo=True)
    true_w = dict(fi.WEIGHTS, light=2.0, waves=1.5, pressure_trend=0.0)
    lam = feats["hours"] * 1.2 * np.exp(sum(true_w[k] * feats[f"log_{k}"] for k in FACTORS))
    feats["catches"] = rng.poisson(lam)
    return feats


def read_log_df(log):
    log = log.copy()
    log["t0"] = pd.to_datetime(log["date"] + " " + log["start"]).dt.tz_localize(fi.TZ)
    log["t1"] = pd.to_datetime(log["date"] + " " + log["end"]).dt.tz_localize(fi.TZ)
    log.loc[log["t1"] <= log["t0"], "t1"] += pd.Timedelta(days=1)
    log["hours"] = (log["t1"] - log["t0"]).dt.total_seconds() / 3600
    return log


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logbook", nargs="?")
    ap.add_argument("--target", default="catches", choices=["catches", "bites"])
    ap.add_argument("--alpha", type=float, default=1.0, help="ridge strength (higher = stay near defaults)")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args(argv)
    if a.demo:
        data = demo_log()
        print("דמו: יומן סינתטי שבו האמת היא אור=2.0, גלים=1.5, לחץ=0\n")
    else:
        if not a.logbook:
            ap.error("logbook.csv required (or --demo)")
        data = session_features(read_log(a.logbook))
    print(f"מפגשים: {len(data)} | סה\"כ {a.target}: {int(data[a.target].sum())} | שעות: {data['hours'].sum():.0f}\n")
    print(fit(data, a.target, a.alpha).to_string(index=False))
    print("\nהעתיקו את fitted_weight ל-WEIGHTS בקובץ fishing_index.py (אחרי 30-40 מפגשים לפחות).")


if __name__ == "__main__":
    main()
