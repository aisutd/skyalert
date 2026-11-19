"""Train the flood-prediction model: for each county and day, the chance that a flood or flash flood gets reported.

One row = one county on one day (lower 48 states). Label = 1 if NOAA Storm Events has a Flood or Flash Flood
starting in that county that day. Inputs = rain around that day and in the days and weeks before it (CPC daily
rain, averaged over the 0.75 degree box around the county centre), how unusual that rain is for the county,
the time of year, and where the county is.

When used live, the "rain around that day" comes from a weather forecast, which is what makes this a prediction.
Here it is trained on the rain that really fell, so the scores below are a best case: a real forecast adds its own error.

Run from the flood-prediction folder, after fetch_data.py:
    python train_forecast.py             # build the table, train, score on the test years
    python train_forecast.py --rebuild   # ignore the cached table in data/county_days.npz

Years: train 2015-2021, validation 2022 (picks the alert cut-off), test 2023-2024 (the numbers to report).
"""
import argparse
import glob
import json
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve

TABLE, MODEL, RESULTS, FIGURE = "data/county_days.npz", "data/flood_forecast.pkl", "data/forecast_results.json", "data/forecast_results.png"
TRAIN_END, VAL_YEAR = 2021, 2022  # train up to 2021, validate on 2022, test on everything after
SKIP_STATES = {2, 15, 60, 66, 69, 72, 78}  # Alaska, Hawaii and territories: outside the rain grid
FEATURES = ["rain_2day", "rain_2day_peak", "rain_day", "rain_next_morning", "rain_prev_1d", "rain_prev_3d", "rain_prev_7d",
            "rain_prev_14d", "rain_prev_30d", "rain_2day_vs_county_extreme", "rain_prev_30d_vs_county_normal",
            "season_sin", "season_cos", "lat", "lon", "log_area_km2", "county_flood_days_per_year"]


def load_counties():
    c = pd.read_csv("data/counties.txt", sep="\t", dtype={"GEOID": int})
    c.columns = c.columns.str.strip()
    c = c[~(c.GEOID // 1000).isin(SKIP_STATES)].reset_index(drop=True)
    return c.GEOID.values, c.INTPTLAT.values, c.INTPTLONG.values, c.ALAND.values / 1e6


def load_rain(lat, lon):
    """Daily rain per county [days, counties]: mean and peak of the 3 x 3 grid cells (0.75 degrees) around its centre."""
    means, peaks, dates = [], [], []
    for path in sorted(glob.glob("data/cpc_rain/precip_*.nc")):
        with netCDF4.Dataset(path) as n:
            if not means:
                iy = np.abs(n["lat"][:].data[None] - lat[:, None]).argmin(1)
                ix = np.abs(n["lon"][:].data[None] - (lon[:, None] % 360)).argmin(1)  # the grid counts longitude 0-360
                ny, nx = n["precip"].shape[1:]
            p = n["precip"][:].filled(np.nan)  # sea and other countries are NaN
            dates.append(pd.to_datetime("1800-01-01") + pd.to_timedelta(n["time"][:].data, unit="h"))
        box = np.stack([p[:, np.clip(iy + dy, 0, ny - 1), np.clip(ix + dx, 0, nx - 1)] for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
        with np.errstate(all="ignore"):
            means.append(np.nanmean(box, 0)); peaks.append(np.nanmax(box, 0))
    return np.concatenate(means).astype("float32"), np.concatenate(peaks).astype("float32"), pd.DatetimeIndex(np.concatenate(dates)).normalize()


def load_reports(dates, geoid):
    """label [days, counties]: 1 where a flood starts, 0 otherwise, -1 on the following days of a flood that is still going on."""
    day0, col = dates[0], {g: i for i, g in enumerate(geoid)}
    label = np.zeros((len(dates), len(geoid)), "int8")
    starts, skipped = [], 0
    for path in sorted(glob.glob("data/storm_events/details_*.csv.gz")):
        d = pd.read_csv(path, usecols=["BEGIN_YEARMONTH", "BEGIN_DAY", "END_YEARMONTH", "END_DAY", "STATE_FIPS", "CZ_TYPE", "CZ_FIPS", "EVENT_TYPE"])
        d = d[d.EVENT_TYPE.isin(["Flood", "Flash Flood"])]
        begin = pd.to_datetime((d.BEGIN_YEARMONTH * 100 + d.BEGIN_DAY).astype(str), format="%Y%m%d")
        end = pd.to_datetime((d.END_YEARMONTH * 100 + d.END_DAY).astype(str), format="%Y%m%d")
        for b, e, fips, kind in zip((begin - day0).dt.days, (end - day0).dt.days, d.STATE_FIPS * 1000 + d.CZ_FIPS, d.CZ_TYPE):
            if kind != "C" or fips not in col or not 0 <= b < len(dates):
                skipped += 1
                continue
            label[b + 1:min(e, b + 30) + 1, col[fips]] = -1  # still flooded: not a new flood, and not a "no flood" day either
            starts.append((b, col[fips]))
    for b, c in starts:
        label[b, c] = 1
    print(f"flood reports used: {len(starts)}   skipped (outside the lower 48 or the years loaded): {skipped}", flush=True)
    return label


def rolling_sum(x, n):
    """Sum of the n days before each day (not counting the day itself)."""
    c = np.concatenate([np.zeros((1, x.shape[1]), "float64"), np.cumsum(x, 0, dtype="float64")])
    out = np.full(x.shape, np.nan, "float32")
    out[n:] = c[n:-1] - c[:-n - 1]
    return out


def build():
    geoid, lat, lon, area = load_counties()
    rain, peak, dates = load_rain(lat, lon)
    ok = np.isfinite(rain).mean(0) > 0.99  # counties whose box is on the rain grid
    geoid, lat, lon, area, rain, peak = geoid[ok], lat[ok], lon[ok], area[ok], np.nan_to_num(rain[:, ok]), np.nan_to_num(peak[:, ok])
    print(f"{len(geoid)} counties, {len(dates)} days ({dates[0].date()} to {dates[-1].date()})", flush=True)
    label = load_reports(dates, geoid)
    train_days = dates.year <= TRAIN_END

    # A rain "day" in this dataset ends at 12:00 UTC, i.e. early morning in the US. So local day D is covered by rain day D
    # (the night before and early morning) plus rain day D+1 (the rest of the day).
    nxt = np.concatenate([rain[1:], np.full((1, rain.shape[1]), np.nan, "float32")])
    two = rain + nxt
    two_peak = np.maximum(peak, np.concatenate([peak[1:], peak[-1:]]))
    prev = {n: rolling_sum(rain, n) for n in (1, 3, 7, 14, 30)}
    extreme = np.maximum(np.nanpercentile(two[train_days], 99, 0), 1.0)  # each county's own "very wet 2 days", from training years only
    normal30 = np.maximum(np.nanmean(prev[30][train_days], 0), 1.0)
    rate = (label[train_days] == 1).sum(0) / (train_days.sum() / 365.25)  # how often this county reports floods (training years only)
    season = 2 * np.pi * np.asarray(dates.day_of_year) / 365.25

    T, C = rain.shape
    wide = lambda v: np.broadcast_to(v[None, :], (T, C))  # one value per county, repeated for every day
    tall = lambda v: np.broadcast_to(v[:, None], (T, C))  # one value per day, repeated for every county
    cols = [two, two_peak, rain, nxt, prev[1], prev[3], prev[7], prev[14], prev[30], two / extreme, prev[30] / normal30,
            tall(np.sin(season)), tall(np.cos(season)), wide(lat), wide(lon), wide(np.log(area)), wide(rate)]
    keep = (label >= 0) & np.isfinite(two) & np.isfinite(prev[30])  # drop ongoing-flood days and the edges where a window is incomplete
    X = np.stack([c[keep] for c in cols], 1).astype("float32")
    np.savez_compressed(TABLE, X=X, y=label[keep], year=tall(dates.year.values)[keep].astype("int16"),
             day=tall(np.arange(T))[keep].astype("int32"), county=wide(np.arange(C))[keep].astype("int16"),
             dates=dates.values.astype("datetime64[D]").astype(str), geoid=geoid, lat=lat, lon=lon, extreme=extreme, normal30=normal30, rate=rate)
    print(f"table: {len(X):,} county-days, {int((label[keep] == 1).sum()):,} with a flood  -> {TABLE}", flush=True)


def at_cutoff(y, score, cut):
    alert = score >= cut
    tp = int((alert & (y == 1)).sum()); fp = int((alert & (y == 0)).sum()); fn = int((~alert & (y == 1)).sum())
    return {"precision": round(tp / max(tp + fp, 1), 4),  # of the alerts, how many had a flood report
            "recall": round(tp / max(tp + fn, 1), 4),  # of the flood reports, how many had an alert
            "csi": round(tp / max(tp + fp + fn, 1), 4),  # hits / (hits + misses + false alarms), the usual forecast score
            "alerts": tp + fp, "floods": tp + fn}


def best_cutoff(y, score):
    p, r, t = precision_recall_curve(y, score)
    csi = 1 / np.maximum(1 / np.maximum(p, 1e-9) + 1 / np.maximum(r, 1e-9) - 1, 1e-9)
    return float(t[np.argmax(csi[:-1])])  # the cut-off with the best hits-vs-misses-vs-false-alarms balance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="rebuild data/county_days.npz from the downloads")
    a = ap.parse_args()
    if a.rebuild or not os.path.exists(TABLE):
        build()
    d = np.load(TABLE)
    X, y, year = d["X"], d["y"], d["year"]
    tr, va, te = year <= TRAIN_END, year == VAL_YEAR, year > VAL_YEAR
    print(f"train {tr.sum():,} rows ({y[tr].mean():.3%} floods)   val {va.sum():,}   test {te.sum():,} ({y[te].mean():.3%} floods)", flush=True)

    model = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, max_leaf_nodes=63, min_samples_leaf=200, l2_regularization=1.0,
                                           early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, scoring="average_precision", random_state=0)
    model.fit(X[tr], y[tr])
    print(f"trained: {model.n_iter_} rounds", flush=True)
    with open(MODEL, "wb") as f:
        pickle.dump({"model": model, "features": FEATURES}, f)

    rivals = {"model": lambda rows: model.predict_proba(X[rows])[:, 1],
              "rule: 2-day rain": lambda rows: X[rows, 0],  # the plain "heavy rain means flood" rule
              "rule: 2-day rain vs county extreme": lambda rows: X[rows, 9]}
    out, curves = {"train_rows": int(tr.sum()), "test_rows": int(te.sum()), "test_years": sorted(set(year[te].tolist())),
                   "test_flood_share": round(float(y[te].mean()), 5), "rounds": int(model.n_iter_), "features": FEATURES}, {}
    for name, score in rivals.items():
        cut, s = best_cutoff(y[va], score(va)), score(te)
        out[name] = {"average_precision": round(float(average_precision_score(y[te], s)), 4), "cutoff_from_val": round(cut, 4), **at_cutoff(y[te], s, cut)}
        curves[name] = precision_recall_curve(y[te], s)[:2]
    json.dump(out, open(RESULTS, "w"), indent=2)

    # figure: precision-recall curves, then the busiest test day as a map (model's chance vs what was reported)
    p_te, day_te, county_te = rivals["model"](te), d["day"][te], d["county"][te]
    busiest = np.bincount(day_te[y[te] == 1]).argmax()
    on = day_te == busiest
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    for name, (p, r) in curves.items():
        ax[0].plot(r, p, label=f"{name} (AP {out[name]['average_precision']:.2f})")
    ax[0].set_xlabel("recall: share of flood reports that had an alert"); ax[0].set_ylabel("precision: share of alerts with a flood report")
    ax[0].set_title(f"Test years {out['test_years'][0]}-{out['test_years'][-1]}"); ax[0].legend()
    lon, lat = d["lon"][county_te[on]], d["lat"][county_te[on]]
    m = ax[1].scatter(lon, lat, c=p_te[on], s=6, cmap="viridis", vmin=0, vmax=max(0.3, float(p_te[on].max())))
    plt.colorbar(m, ax=ax[1], label="model's chance of a flood report"); ax[1].set_title(f"Model, {d['dates'][busiest]}")
    hit = y[te][on] == 1
    ax[2].scatter(lon[~hit], lat[~hit], c="#dddddd", s=6); ax[2].scatter(lon[hit], lat[hit], c="#ff7f0e", s=10)
    ax[2].set_title(f"Reported floods, {d['dates'][busiest]} ({int(hit.sum())} counties)")
    for x in ax[1:]:
        x.set_xticks([]); x.set_yticks([]); x.set_aspect(1.3)
    plt.tight_layout(); plt.savefig(FIGURE, dpi=80); plt.close(fig)

    print(f"\nTest years {out['test_years']}: {out['test_rows']:,} county-days, {out['test_flood_share']:.2%} with a flood report")
    for name in rivals:
        s = out[name]
        print(f"  {name:36s} AP {s['average_precision']:.3f}   at its cut-off: precision {s['precision']:.1%}  recall {s['recall']:.1%}  CSI {s['csi']:.3f}")
    print(f"saved {MODEL}, {RESULTS}, {FIGURE}")


if __name__ == "__main__":
    main()
