"""Build the spread model's 12 input layers for any place and day from live or archived data.

One call gives the (12, 64, 64) array in backend.spread_model.INPUT_FEATURES order, raw units,
in the training tiles' orientation (backend/spread_grid.py), plus a record of where every layer
came from. The recipe for each channel and the evidence behind it are in
docs/spread_forecast_plan.md, "How we rebuild each channel from live data".

Missing data is NaN. backend.spread_model.normalize turns NaN into the train mean, the same way
training handled missing pixels, so nothing here is made up.
"""

import datetime as dt
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from backend import fire_points, gridmet, terrain
from backend.spread_grid import TileGrid
from backend.spread_model import INPUT_FEATURES, TILE_SIZE

WEATHER_CHANNELS = ("th", "vs", "tmmn", "tmmx", "sph", "pr", "erc")
# Not available without an Earthdata login (decision of 2026-11-11). NaN = train mean.
FILLED_CHANNELS = {
    "NDVI": "No live source without a NASA Earthdata login; filled with the training average.",
    "population": "No live source without a NASA Earthdata login; filled with the training average.",
}

# Layer status shown to people
STATIC, LIVE, LAGGED, FILLED, MISSING = "static", "live", "lagged", "filled", "missing"


class SpreadInputError(Exception):
    """Raised in strict mode when a layer can't be built."""


@dataclass
class SpreadInputs:
    grid: TileGrid
    fire_day: dt.date  # day t: the fire the model starts from
    forecast_day: dt.date  # day t+1: the day the forecast is for
    inputs: np.ndarray  # (12, 64, 64) float32, INPUT_FEATURES order, raw units, NaN = missing
    layers: dict = field(default_factory=dict)  # channel -> {status, source, dates, note, ...}

    @property
    def batch(self):
        """The inputs as a batch of one, ready for backend.spread_model.predict."""
        return self.inputs[None]


def _as_date(day):
    return day if isinstance(day, dt.date) else dt.date.fromisoformat(str(day))


def training_weather_days(fire_day):
    """The gridMET days the training tiles used: day t-1 and day t (median of two = their mean)."""
    return [fire_day - dt.timedelta(days=1), fire_day]


def live_weather_days(fire_day, latest=None):
    """The training days if gridMET has them yet, otherwise the two latest days it has.

    gridMET runs about two days behind, so a forecast from today's fire usually gets weather
    from two to three days earlier. The layer status says so.
    """
    wanted = training_weather_days(fire_day)
    latest = latest or gridmet.latest_day()
    if latest >= wanted[-1]:
        return wanted
    return [latest - dt.timedelta(days=1), latest]


def build_inputs(
    lat,
    lon,
    fire_day,
    weather_days=None,
    geometry="degree",
    fire_source="MODIS",
    fire_satellites=("Terra",),
    points=None,
    strict=False,
):
    """The model inputs for the tile around (lat, lon), starting from the fire seen on fire_day (UTC).

    weather_days: gridMET days to average. None = live_weather_days(fire_day): the training
        window (t-1, t) when it exists, else the latest two days gridMET has.
    fire_source: "MODIS" (Terra, like training) with VIIRS NOAA-20/21 as the fallback, or "VIIRS".
    points: optional pre-fetched fire detections {"MODIS": df, "VIIRS": df} (backtest).
    strict: raise SpreadInputError if any layer fails, instead of leaving it NaN.
    """
    fire_day = _as_date(fire_day)
    grid = TileGrid.around(lat, lon, geometry=geometry)
    if weather_days is None:
        weather_days = live_weather_days(fire_day)
    weather_days = [_as_date(day) for day in weather_days]
    weather_status = LIVE if weather_days == training_weather_days(fire_day) else LAGGED

    jobs = {
        "elevation": lambda: (terrain.elevation(grid), {}),
        "pdsi": lambda: gridmet.pdsi_layer(grid, fire_day),
        "PrevFireMask": lambda: fire_points.fire_mask(
            grid, fire_day, prefer=fire_source, satellites=fire_satellites, points=points
        ),
    }
    for var in WEATHER_CHANNELS:
        jobs[var] = lambda var=var: gridmet.weather_layer(grid, var, weather_days)

    results, errors = {}, {}
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = {name: pool.submit(job) for name, job in jobs.items()}
        for name, future in futures.items():
            try:
                results[name] = future.result()
            except Exception as error:  # one failed layer shouldn't sink the forecast
                errors[name] = f"{type(error).__name__}: {error}"
    if strict and errors:
        raise SpreadInputError("; ".join(f"{name}: {message}" for name, message in errors.items()))

    stack = np.full((len(INPUT_FEATURES), TILE_SIZE, TILE_SIZE), np.nan, dtype=np.float32)
    layers = {}
    for channel, name in enumerate(INPUT_FEATURES):
        if name in FILLED_CHANNELS:
            layers[name] = {"status": FILLED, "source": None, "note": FILLED_CHANNELS[name]}
            continue
        if name in errors:
            layers[name] = {"status": MISSING, "error": errors[name], "note": "Failed to load; filled with the training average."}
            continue
        layer, info = results[name]
        layer = np.asarray(layer, dtype=np.float32)
        if layer.shape != (TILE_SIZE, TILE_SIZE):
            raise SpreadInputError(f"{name} came back with shape {layer.shape}")
        stack[channel] = layer
        layers[name] = _describe(name, info, weather_days, weather_status, fire_day)
        nan_share = float(np.isnan(layer).mean())
        if nan_share:
            layers[name]["nan_share"] = round(nan_share, 4)

    return SpreadInputs(
        grid=grid,
        fire_day=fire_day,
        forecast_day=fire_day + dt.timedelta(days=1),
        inputs=stack,
        layers=layers,
    )


def _describe(name, info, weather_days, weather_status, fire_day):
    """Where one layer came from, for the API and the backtest log."""
    info = dict(info or {})
    if name == "elevation":
        return {"status": STATIC, "source": "AWS Terrain Tiles (Terrarium), 1 km mean", **info}
    if name in WEATHER_CHANNELS:
        lag_days = (fire_day - weather_days[-1]).days
        return {
            "status": weather_status,
            "source": "gridMET",
            "dates": [day.isoformat() for day in weather_days],
            "lag_days": lag_days,
            **{key: value for key, value in info.items() if key not in ("days_used",)},
        }
    if name == "pdsi":
        return {"status": LAGGED, "source": "gridMET drought (PDSI)", **info}
    if name == "PrevFireMask":
        family = info.pop("source", None)
        return {"status": LIVE, **info, "family": family, "source": f"NASA FIRMS {family}" if family else "NASA FIRMS"}
    return {"status": LIVE, **info}
