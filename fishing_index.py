#!/usr/bin/env python3
"""
Shore-spinning fishing index (מדד דיג לז'רז'ור מהחוף) – southern Israeli coast.

Hourly score 0-100 per beach x target species for the next days, built from
free Open-Meteo data (weather + marine), plus astronomy computed locally.

Design:
  * Each factor -> a 0..1 sub-score (transparent, editable functions).
  * Weighted geometric mean of sub-scores (one bad factor drags the hour down).
  * Hard safety gate (big waves / strong wind) -> score 0 and flagged.
  * Weights live in WEIGHTS / SPECIES; calibrate.py re-estimates them from
    your own logbook (Poisson GLM), which is the part that actually makes
    an index good.

Usage:
  python fishing_index.py                    # live data, all beaches, all species
  python fishing_index.py --beach nitzanim --species gumbar --days 5
  python fishing_index.py --demo             # synthetic data, no internet needed
Outputs: console summary of best windows + fishing_index.csv + fishing_index.html

Data: Open-Meteo (free for non-commercial use; attribution required).
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

TZ = "Asia/Jerusalem"

# ----------------------------------------------------------------------------
# Locations. Points are ~0.5-1 km offshore so the marine model returns sea cells.
# coast_normal_deg = direction the beach faces (towards the sea); used to split
# wind into onshore / offshore. Southern Israeli coast faces roughly WNW (~290°).
# Coordinates are approximate – adjust to your exact spot.
# ----------------------------------------------------------------------------
BEACHES = {
    # habitat features 0..1 (rough, editable): rocks = breakwaters / kurkar reefs,
    # lights = marina / promenade lighting at night, stream = river/stream mouth nearby
    "nitzanim": dict(name="ניצנים", lat=31.727, lon=34.585, coast_normal_deg=292,
                     habitat=dict(rocks=0.4, lights=0.0, stream=0.0)),
    "ashkelon_delila": dict(name="אשקלון – דלילה", lat=31.683, lon=34.548, coast_normal_deg=295,
                            habitat=dict(rocks=1.0, lights=0.7, stream=0.0)),
    "ashdod_lido": dict(name="אשדוד – לידו/מרינה", lat=31.802, lon=34.628, coast_normal_deg=290,
                        habitat=dict(rocks=1.0, lights=0.8, stream=0.2)),
    "zikim": dict(name="זיקים", lat=31.612, lon=34.505, coast_normal_deg=300,
                  habitat=dict(rocks=0.2, lights=0.0, stream=0.4)),
}

# ----------------------------------------------------------------------------
# Species profiles = PRIORS. Values are reasonable starting points from angler
# practice + general biology, NOT measured local truth. Calibrate them.
#   months:     presence/activity prior per month 1..12 (0..1)
#   sst:        (low_ok, best_low, best_high, high_ok) °C trapezoid
#   light:      weights for night / twilight / day
#   waves:      (min_ok, best_low, best_high, max_ok) significant wave height, m
#   murk_pref:  -1 prefers clear water, 0 neutral, +1 likes some murk/whitewater
# ----------------------------------------------------------------------------
@dataclass
class Species:
    key: str
    name: str
    months: list
    sst: tuple
    light: dict
    waves: tuple
    murk_pref: float
    notes: str = ""
    habitat: dict = field(default_factory=dict)  # feature -> importance 0..1
    legal: str = ""                               # rules shown to the user


SPECIES = {
    "gumbar": Species(
        "gumbar", "גומבר (Bluefish)",
        months=[0.7, 0.6, 0.5, 0.5, 0.5, 0.4, 0.4, 0.5, 0.7, 0.9, 1.0, 0.9],
        sst=(15, 18, 24, 28),
        light=dict(night=0.55, twilight=1.0, day=0.45),
        waves=(0.15, 0.4, 1.1, 1.6), murk_pref=0.5,
        notes="schools chasing bait; dawn/dusk blitzes; tolerates rough water"),
    "intias": Species(
        "intias", "אינטיאס (Leerfish, Lichia amia)",
        months=[0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.7, 0.8, 1.0, 1.0, 0.8],
        sst=(16, 20, 27, 29),  # FishBase preferred range 16.2-29 °C
        light=dict(night=0.5, twilight=1.0, day=0.5),
        waves=(0.2, 0.5, 1.2, 1.7), murk_pref=0.6,
        notes="close to shore, murky surf and river/stream mouths, follows mullet",
        habitat=dict(stream=0.4)),
    "levrak": Species(
        "levrak", "לברק (European seabass)",
        months=[1.0, 1.0, 0.8, 0.5, 0.3, 0.2, 0.2, 0.2, 0.3, 0.6, 0.9, 1.0],
        sst=(12, 15, 20, 24),
        light=dict(night=0.8, twilight=1.0, day=0.35),
        waves=(0.3, 0.6, 1.3, 1.8), murk_pref=1.0,
        notes="winter fish; white-water and murky surf after storms; low light"),
    "palamida": Species(
        "palamida", "פלמידה (Atlantic bonito)",
        months=[0.4, 0.3, 0.3, 0.4, 0.4, 0.4, 0.5, 0.6, 0.8, 1.0, 1.0, 0.7],
        sst=(16, 18, 24, 27),
        light=dict(night=0.1, twilight=1.0, day=0.6),
        waves=(0.0, 0.2, 0.7, 1.2), murk_pref=-1.0,
        notes="pelagic, clear water, calm-ish sea, long casts with metal jigs"),
    "trachon": Species(
        "trachon", "טרחון (Blue runner, Caranx crysos)",
        # Sources disagree on season: Med FAD studies -> Sep–Jan; local angler guide (Baba Jig)
        # -> late winter + spring. Keep a flat-high prior with both peaks and let SST decide.
        months=[0.75, 0.85, 0.95, 0.95, 0.8, 0.6, 0.6, 0.7, 0.85, 0.95, 0.95, 0.85],
        sst=(15, 18, 27, 30),  # FishBase preferred 16.9-27.9, mean 24.9 °C
        light=dict(night=0.25, twilight=1.0, day=0.9),
        waves=(0.0, 0.2, 0.9, 1.4), murk_pref=-0.3,
        notes="diurnal schooling jack; near shore, breakwaters, holes and currents; fast shiny lures – "
              "metal jigs, small poppers, short minnows; medium steady retrieve with sudden jerks",
        habitat=dict(rocks=0.4),
        legal=""),
    "lokus": Species(
        "lokus", "לוקוס (Dusky grouper)",
        months=[0.3, 0.3, 0.4, 0.5, 0.7, 0.9, 1.0, 1.0, 0.9, 0.7, 0.5, 0.4],
        sst=(16, 19, 27, 30),
        light=dict(night=0.45, twilight=1.0, day=0.7),
        waves=(0.0, 0.15, 0.8, 1.3), murk_pref=-0.3,
        notes="ambush predator of rocks/breakwaters; mostly inactive, more active when warm and "
              "around spawning (summer); work jigs/soft plastics close to the rocks",
        habitat=dict(rocks=0.9),
        legal="גודל מינימלי 40 ס\"מ. דקר אלכסנדרוני (לוקוס חופש) מוגן ואסור לדוג. "
              "יש איסור עונתי נפרד לדקרים – בדוק במודעת רשות הטבע והגנים"),
    "dennis": Species(
        "dennis", "דניס (Gilthead seabream)",
        months=[0.8, 0.7, 0.7, 0.7, 0.6, 0.5, 0.5, 0.5, 0.6, 0.8, 0.9, 0.9],
        sst=(13, 16, 25, 28),
        light=dict(night=0.4, twilight=1.0, day=0.8),
        waves=(0.1, 0.3, 0.9, 1.4), murk_pref=0.2,
        notes="feeds on shellfish/crabs near rocks and mussel-covered structure; mostly a bait fish – "
              "on lures use small crab/soft-plastic imitations slowly near the bottom",
        habitat=dict(rocks=0.6),
        legal="גודל מינימלי 15 ס\"מ"),
}

# Global factor weights (exponents in the geometric mean). Evidence-based ranking:
# light & sea state & season strongest; pressure/moon weak; solunar 0 by default.
WEIGHTS = dict(
    season=1.0,
    sst=0.8,
    light=1.2,
    waves=1.0,
    water_clarity=0.7,
    post_storm=0.5,
    pressure_trend=0.25,
    wind=0.5,
    current=0.2,
    moon_night=0.15,
    habitat=1.0,
)

SAFETY = dict(max_wave_m=2.0, max_wind_kmh=40.0, max_gust_kmh=60.0)


# ----------------------------------------------------------------------------
# Astronomy (no external libs). Accurate to ~1° which is plenty here.
# ----------------------------------------------------------------------------
def sun_elevation_deg(ts_utc: pd.DatetimeIndex, lat: float, lon: float) -> np.ndarray:
    """NOAA-style solar elevation."""
    jd = ts_utc.to_julian_date().values
    n = jd - 2451545.0
    L = np.deg2rad((280.460 + 0.9856474 * n) % 360)
    g = np.deg2rad((357.528 + 0.9856003 * n) % 360)
    lam = L + np.deg2rad(1.915) * np.sin(g) + np.deg2rad(0.020) * np.sin(2 * g)
    eps = np.deg2rad(23.439 - 0.0000004 * n)
    ra = np.arctan2(np.cos(eps) * np.sin(lam), np.cos(lam))
    dec = np.arcsin(np.sin(eps) * np.sin(lam))
    gmst_h = (18.697374558 + 24.06570982441908 * n) % 24
    lst = np.deg2rad((gmst_h * 15 + lon) % 360)
    ha = lst - ra
    latr = np.deg2rad(lat)
    elev = np.arcsin(np.sin(latr) * np.sin(dec) + np.cos(latr) * np.cos(dec) * np.cos(ha))
    return np.rad2deg(elev)


SYNODIC = 29.530588853
NEW_MOON_REF = pd.Timestamp("2000-01-06 18:14", tz="UTC")


def moon_phase(ts_utc: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """Returns (age_days 0..29.5, illuminated fraction 0..1)."""
    age = ((ts_utc - NEW_MOON_REF).total_seconds().values / 86400.0) % SYNODIC
    illum = (1 - np.cos(2 * np.pi * age / SYNODIC)) / 2
    return age, illum


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
WEATHER_VARS = ["pressure_msl", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
                "cloud_cover", "precipitation", "temperature_2m"]
MARINE_VARS = ["wave_height", "wave_period", "wave_direction", "sea_surface_temperature",
               "sea_level_height_msl", "ocean_current_velocity"]


def _get_json(url: str, params: dict, tries: int = 5):
    """GET with retry/backoff – Open-Meteo returns 429 when shared CI IPs are busy."""
    import time
    import requests

    wait = 10
    for attempt in range(tries):
        try:
            r = requests.get(url, params=params, timeout=60)
        except (requests.ConnectionError, requests.Timeout):
            if attempt == tries - 1:
                raise
            time.sleep(wait)
            wait = min(wait * 2, 120)
            continue
        if r.status_code not in (429, 500, 502, 503, 504):
            r.raise_for_status()
            return r.json()
        if attempt == tries - 1:
            r.raise_for_status()
        ra = r.headers.get("Retry-After")
        time.sleep(int(ra) if ra and ra.isdigit() else wait)
        wait = min(wait * 2, 120)
    raise RuntimeError("unreachable")


def fetch_open_meteo(lat: float, lon: float, days: int, past_days: int = 3,
                     start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
    import requests

    common = dict(latitude=lat, longitude=lon, timezone=TZ, wind_speed_unit="kmh")
    if start_date:
        common.update(start_date=start_date, end_date=end_date)
        wurl = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    else:
        common.update(forecast_days=days, past_days=past_days)
        wurl = "https://api.open-meteo.com/v1/forecast"
    w = _get_json(wurl, {**common, "hourly": ",".join(WEATHER_VARS)})
    m = _get_json("https://marine-api.open-meteo.com/v1/marine", {**common, "hourly": ",".join(MARINE_VARS)})
    dw = pd.DataFrame(w["hourly"])
    dm = pd.DataFrame(m["hourly"])
    df = dw.merge(dm, on="time", how="outer")
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(TZ, ambiguous="NaT", nonexistent="shift_forward")
    return df.dropna(subset=["time"]).set_index("time").sort_index()


def fetch_open_meteo_multi(points: list[tuple[float, float]], days: int, past_days: int = 3) -> list[pd.DataFrame]:
    """One weather + one marine request for all locations (Open-Meteo accepts
    comma-separated coordinates and returns a list) – avoids rate limits."""
    lats = ",".join(str(p[0]) for p in points)
    lons = ",".join(str(p[1]) for p in points)
    common = dict(latitude=lats, longitude=lons, timezone=TZ, wind_speed_unit="kmh",
                  forecast_days=days, past_days=past_days)
    w = _get_json("https://api.open-meteo.com/v1/forecast", {**common, "hourly": ",".join(WEATHER_VARS)})
    m = _get_json("https://marine-api.open-meteo.com/v1/marine", {**common, "hourly": ",".join(MARINE_VARS)})
    w = w if isinstance(w, list) else [w]
    m = m if isinstance(m, list) else [m]
    out = []
    for wi, mi in zip(w, m):
        df = pd.DataFrame(wi["hourly"]).merge(pd.DataFrame(mi["hourly"]), on="time", how="outer")
        df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(TZ, ambiguous="NaT", nonexistent="shift_forward")
        out.append(df.dropna(subset=["time"]).set_index("time").sort_index())
    return out


def demo_data(days: int, past_days: int = 3, seed: int = 7) -> pd.DataFrame:
    """Synthetic but realistic-looking data: a winter storm passing mid-week."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp.now(tz=TZ).floor("D") - pd.Timedelta(days=past_days)
    idx = pd.date_range(start, periods=24 * (days + past_days), freq="h")
    t = np.arange(len(idx)) / 24.0
    storm = np.exp(-((t - (past_days + 2.0)) ** 2) / 0.8)          # storm peak ~2 days ahead
    df = pd.DataFrame(index=idx)
    df["pressure_msl"] = 1016 - 12 * storm + rng.normal(0, 0.4, len(t))
    df["wind_speed_10m"] = 10 + 35 * storm + 4 * np.sin(2 * np.pi * t) + rng.normal(0, 2, len(t))
    df["wind_gusts_10m"] = df["wind_speed_10m"] * 1.5
    df["wind_direction_10m"] = np.where(storm > 0.2, 260, 100) + rng.normal(0, 15, len(t))
    df["cloud_cover"] = np.clip(20 + 80 * storm + rng.normal(0, 10, len(t)), 0, 100)
    df["precipitation"] = np.clip(3 * storm - 0.5, 0, None)
    df["temperature_2m"] = 19 + 4 * np.sin(2 * np.pi * (t - 0.4))
    df["wave_height"] = np.clip(0.4 + 2.4 * np.convolve(storm, np.ones(12) / 12, "same")
                                + rng.normal(0, 0.05, len(t)), 0.1, None)
    df["wave_period"] = 5 + 3 * storm
    df["wave_direction"] = 280.0
    df["sea_surface_temperature"] = 22.0 - 0.3 * t / (days + past_days)
    df["sea_level_height_msl"] = 0.15 * np.sin(2 * np.pi * t / (12.42 / 24)) + 0.2 * storm
    df["ocean_current_velocity"] = 0.6 + 1.5 * storm
    return df


# ----------------------------------------------------------------------------
# Feature engineering
# ----------------------------------------------------------------------------
def add_features(df: pd.DataFrame, beach: dict) -> pd.DataFrame:
    df = df.copy()
    ts_utc = df.index.tz_convert("UTC")
    df["sun_elev"] = sun_elevation_deg(ts_utc, beach["lat"], beach["lon"])
    df["moon_age"], df["moon_illum"] = moon_phase(ts_utc)
    df["pressure_trend_3h"] = df["pressure_msl"].diff(3)
    df["pressure_trend_12h"] = df["pressure_msl"].diff(12)
    df["wave_max_prev48h"] = df["wave_height"].rolling(48, min_periods=6).max()
    df["wave_trend_12h"] = df["wave_height"].diff(12)
    # onshore component of wind (km/h): positive = blowing from sea to land
    rel = np.deg2rad(df["wind_direction_10m"] - beach["coast_normal_deg"])
    df["onshore_wind"] = df["wind_speed_10m"] * np.cos(rel)
    df["sea_level_rate"] = df["sea_level_height_msl"].diff()
    # crude turbidity proxy 0..1: recent big waves + onshore wind stir the sand
    stir = (df["wave_height"].rolling(24, min_periods=3).mean() - 0.5).clip(lower=0) / 1.2
    df["murk_proxy"] = (stir + 0.1 * (df["precipitation"].rolling(24, min_periods=1).sum() > 5)).clip(0, 1)
    return df


# ----------------------------------------------------------------------------
# Sub-scores (0..1)
# ----------------------------------------------------------------------------
def trapezoid(x, a, b, c, d, floor=0.05):
    x = np.asarray(x, dtype=float)
    y = np.where(x <= a, 0, np.where(x < b, (x - a) / max(b - a, 1e-9),
                 np.where(x <= c, 1, np.where(x < d, (d - x) / max(d - c, 1e-9), 0))))
    y = np.where(np.isnan(x), 0.5, y)
    return np.clip(y, floor, 1.0)


def light_score(sun_elev, sp: Species):
    """Twilight window = sun between about -10° and +12° (first/last light,
    the classic predator feeding window). Membership is a trapezoid on solar
    elevation so it works with hourly data."""
    e = np.asarray(sun_elev, dtype=float)
    tw = trapezoid(e, -12, -6, 6, 14, floor=0.0)
    night = (e <= -6).astype(float)
    day = (e >= 6).astype(float)
    s = np.maximum.reduce([sp.light["twilight"] * tw,
                           sp.light["night"] * night,
                           sp.light["day"] * day])
    return np.clip(s, 0.05, 1.0)


def clarity_score(murk, sp: Species):
    """murk_pref +1 -> likes some murk (~0.3-0.6); -1 -> wants clear water.
    Penalty scaled by how strong the preference is."""
    m = np.asarray(murk, dtype=float)
    target = 0.45 if sp.murk_pref > 0 else 0.0
    fit = np.exp(-((m - target) / 0.4) ** 2)
    return np.clip(1 - abs(sp.murk_pref) * 0.6 * (1 - fit), 0.2, 1.0)


def post_storm_score(df):
    """Hypothesis common in surf fishing: 'the day after the storm' – waves
    dropping from a recent high. Bonus when last-48h max was big and waves now fall."""
    recent_big = ((df["wave_max_prev48h"] - 1.2) / 1.0).clip(0, 1)
    falling = (-df["wave_trend_12h"] / 0.6).clip(0, 1)
    return (0.8 + 0.2 * recent_big * falling).fillna(0.8).values


def pressure_score(df):
    # slow fall (−1..−4 hPa/12h) slightly positive; sharp swings slightly negative
    p = df["pressure_trend_12h"].fillna(0).values
    s = 0.85 + 0.15 * np.exp(-((p + 2.5) / 2.0) ** 2) - 0.25 * (np.abs(p) > 7)
    return np.clip(s, 0.3, 1.0)


def wind_score(df):
    # light-moderate wind good (surface chop), strong onshore bad for casting/line control
    ws = df["wind_speed_10m"].values
    on = df["onshore_wind"].values
    s = trapezoid(ws, -1, 4, 20, 35, floor=0.2)
    s = s - 0.25 * np.clip((on - 20) / 15, 0, 1)
    return np.clip(s, 0.15, 1.0)


def current_score(df):
    v = df["ocean_current_velocity"].fillna(0.5).values      # km/h in Open-Meteo
    return np.clip(trapezoid(v, 0.0, 0.4, 2.5, 4.5, floor=0.5), 0.5, 1.0)


def moon_night_score(df):
    # Only matters at night: some light helps visual predators; very dark slightly worse.
    night = df["sun_elev"].values < -8
    m = df["moon_illum"].values
    return np.where(night, 0.8 + 0.2 * m, 1.0)


def habitat_score(sp: Species, beach_key: str | None) -> float:
    """How well the beach's features match what the fish needs (rocks, lights, stream)."""
    if not beach_key or not sp.habitat:
        return 1.0
    feat = BEACHES[beach_key].get("habitat", {})
    s = 1.0
    for k, imp in sp.habitat.items():
        s *= 1 - imp * (1 - feat.get(k, 0.0))
    return float(np.clip(s, 0.1, 1.0))


def score_species(df: pd.DataFrame, sp: Species, weights=WEIGHTS, beach_key: str | None = None) -> pd.DataFrame:
    month = df.index.month.values
    sub = pd.DataFrame(index=df.index)
    sub["habitat"] = habitat_score(sp, beach_key)
    sub["season"] = np.array(sp.months)[month - 1]
    sub["sst"] = trapezoid(df["sea_surface_temperature"], *sp.sst, floor=0.1)
    sub["light"] = light_score(df["sun_elev"], sp)
    sub["waves"] = trapezoid(df["wave_height"], *sp.waves, floor=0.05)
    sub["water_clarity"] = clarity_score(df["murk_proxy"], sp)
    sub["post_storm"] = post_storm_score(df) if sp.murk_pref >= 0 else 0.9
    sub["pressure_trend"] = pressure_score(df)
    sub["wind"] = wind_score(df)
    sub["current"] = current_score(df)
    sub["moon_night"] = moon_night_score(df)

    # weighted product: each factor pulls the score down by s**w (ideal hour = 100)
    logscore = sum(weights[k] * np.log(np.clip(sub[k].astype(float), 1e-3, 1)) for k in weights)
    score = 100 * np.exp(logscore)

    unsafe = ((df["wave_height"] > SAFETY["max_wave_m"]) |
              (df["wind_speed_10m"] > SAFETY["max_wind_kmh"]) |
              (df.get("wind_gusts_10m", 0) > SAFETY["max_gust_kmh"]))
    score = np.where(unsafe, 0, score)
    out = sub.add_prefix("s_")
    out["score"] = np.round(score, 1)
    out["unsafe"] = unsafe.values
    # the weakest factor = the main reason an hour is not better
    contrib = pd.DataFrame({k: weights[k] * np.log(np.clip(sub[k].astype(float), 1e-3, 1)) for k in weights})
    out["limiting"] = contrib.idxmin(axis=1)
    return out


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
FACTOR_HE = dict(season="עונה", sst="טמפ' מים", light="אור/שעה", waves="גלים",
                 water_clarity="צלילות", post_storm="אחרי סערה", pressure_trend="לחץ",
                 wind="רוח", current="זרם", moon_night="ירח", habitat="מבנה החוף")


def best_windows(res: pd.DataFrame, min_len=2, top=3) -> list[dict]:
    """Group consecutive hours with score >= 75th percentile of the period."""
    s = res["score"]
    thr = max(np.nanpercentile(s[s > 0], 85) if (s > 0).any() else 0, 1)
    good = s >= thr
    wins, start = [], None
    for t, g in good.items():
        if g and start is None:
            start = t
        if (not g) and start is not None:
            wins.append((start, prev))
            start = None
        prev = t
    if start is not None:
        wins.append((start, prev))
    out = []
    for a, b in wins:
        seg = res.loc[a:b]
        if len(seg) >= min_len:
            out.append(dict(start=a, end=b + pd.Timedelta(hours=1), mean=seg["score"].mean(),
                            peak=seg["score"].max(), limiting=seg["limiting"].mode().iat[0]))
    return sorted(out, key=lambda d: -d["mean"])[:top]


def run(beaches, species_keys, days, demo=False) -> pd.DataFrame:
    rows = []
    now = pd.Timestamp.now(tz=TZ).floor("h")
    if not demo:
        raws = dict(zip(beaches, fetch_open_meteo_multi(
            [(BEACHES[k]["lat"], BEACHES[k]["lon"]) for k in beaches], days)))
    for bkey in beaches:
        b = BEACHES[bkey]
        raw = demo_data(days) if demo else raws[bkey]
        feat = add_features(raw, b)
        feat = feat[feat.index >= now]
        for skey in species_keys:
            res = score_species(feat, SPECIES[skey], beach_key=bkey)
            res.insert(0, "species", skey)
            res.insert(0, "beach", bkey)
            res = res.join(feat[["wave_height", "sea_surface_temperature", "wind_speed_10m",
                                 "onshore_wind", "pressure_msl", "sun_elev", "moon_illum"]])
            rows.append(res)
    out = pd.concat(rows)
    out.index.name = "time"
    return out


DAYS_HE = ["שני", "שלישי", "רביעי", "חמישי", "שישי", "שבת", "ראשון"]


def summary_text(allres: pd.DataFrame, md=False) -> str:
    lines = []
    for (bkey, skey), res in allres.groupby(["beach", "species"], sort=False):
        lines.append(("\n### " if md else "\n=== ") + f"{BEACHES[bkey]['name']} | {SPECIES[skey].name}" + ("" if md else " ==="))
        unsafe_h = int(res["unsafe"].sum())
        if unsafe_h:
            lines.append(f"- ⚠ {unsafe_h} שעות לא בטוחות (גלים/רוח)")
        for w in best_windows(res):
            lines.append(f"- יום {DAYS_HE[w['start'].weekday()]} {w['start']:%d/%m} {w['start']:%H:%M}–{w['end']:%H:%M}"
                         f" | ממוצע {w['mean']:.0f}, שיא {w['peak']:.0f}"
                         f" | מגביל: {FACTOR_HE.get(w['limiting'], w['limiting'])}")
    return "\n".join(lines)


def print_summary(allres: pd.DataFrame):
    print(summary_text(allres))


def write_html(allres: pd.DataFrame, path: str):
    """Simple heatmap: rows = beach×species, columns = hours."""
    piv = allres.reset_index().pivot_table(index=["beach", "species"], columns="time", values="score")
    def color(v):
        if np.isnan(v):
            return "#eee"
        if v == 0:
            return "#555"
        h = int(120 * min(v, 100) / 100)  # red->green
        return f"hsl({h},70%,{80 - v * 0.35:.0f}%)"
    cols = piv.columns
    head = "".join(f"<th>{c:%d/%m<br>%H}</th>" if c.hour % 3 == 0 else "<th></th>" for c in cols)
    body = ""
    for (b, s), r in piv.iterrows():
        cells = "".join(f'<td title="{c:%a %d/%m %H:00} → {v:.0f}" style="background:{color(v)}"></td>'
                        for c, v in r.items())
        body += f"<tr><th class=l>{BEACHES[b]['name']}<br><small>{SPECIES[s].name}</small></th>{cells}</tr>"
    html = f"""<!doctype html><html dir="rtl" lang="he"><meta charset="utf-8">
<title>מדד דיג</title><style>
body{{font-family:system-ui,Arial;margin:12px}} table{{border-collapse:collapse;direction:ltr}}
td{{width:9px;height:26px;padding:0}} th{{font-size:9px;font-weight:normal;color:#666}}
th.l{{text-align:right;font-size:12px;color:#000;padding-right:6px;white-space:nowrap;direction:rtl}}
</style><h3>מדד דיג לפי שעה (0–100). אפור כהה = לא בטוח</h3>
<div style="overflow-x:auto"><table><tr><th></th>{head}</tr>{body}</table></div>
<p style="font-size:12px;color:#666">נתונים: Open-Meteo. המדד הוא הערכה ראשונית – כיילו מול יומן הדיג (calibrate.py).</p></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--beach", nargs="*", default=list(BEACHES), choices=list(BEACHES))
    ap.add_argument("--species", nargs="*", default=list(SPECIES), choices=list(SPECIES))
    ap.add_argument("--days", type=int, default=5)
    ap.add_argument("--demo", action="store_true", help="synthetic data, no network")
    ap.add_argument("--csv", default="fishing_index.csv")
    ap.add_argument("--html", default="fishing_index.html")
    ap.add_argument("--md", default=None, help="write a markdown summary to this path")
    a = ap.parse_args(argv)
    res = run(a.beach, a.species, a.days, demo=a.demo)
    print_summary(res)
    res.to_csv(a.csv, encoding="utf-8-sig")
    write_html(res, a.html)
    if a.md:
        with open(a.md, "w", encoding="utf-8") as f:
            f.write(f"## מדד דיג – עודכן {pd.Timestamp.now(tz=TZ):%d/%m/%Y %H:%M}\n" + summary_text(res, md=True) + "\n")
    print(f"\nנשמר: {a.csv}, {a.html}")


if __name__ == "__main__":
    main()
