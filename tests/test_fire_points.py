"""FIRMS area fetching (backend/firms_client.py) and today's fire mask (backend/fire_points.py).

Unit tests stub every FIRMS call. Tests marked network ask FIRMS for real data and run
only with --network (they use a handful of transactions out of 5000 per 10 minutes).
"""

from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from backend import fire_points as fp
from backend import firms_client
from backend.firms_client import FirmsError
from backend.spread_grid import CENTER_INDEX, TileGrid

AVAILABILITY = pd.DataFrame(
    [
        ("MODIS_NRT", "2026-09-01", "2026-11-11"),
        ("MODIS_SP", "2000-11-01", "2026-08-31"),
        ("VIIRS_NOAA20_NRT", "2026-08-01", "2026-11-11"),
        ("VIIRS_NOAA20_SP", "2018-04-01", "2026-07-31"),
        ("VIIRS_NOAA21_NRT", "2024-01-17", "2026-11-11"),
        ("VIIRS_NOAA21_SP", "2023-03-01", "2026-07-31"),
        ("VIIRS_SNPP_NRT", "2026-08-01", "2026-11-02"),
        ("VIIRS_SNPP_SP", "2012-01-20", "2026-07-31"),
    ],
    columns=["data_id", "min_date", "max_date"],
)
for _column in ("min_date", "max_date"):
    AVAILABILITY[_column] = pd.to_datetime(AVAILABILITY[_column]).dt.date

CENTER = (38.4175, -122.1747)  # a real Terra detection in the LNU Lightning Complex, 2020-08-20
DAY = date(2020, 8, 20)


def modis_rows(points, day=DAY, satellite="Terra"):
    """A MODIS-shaped FIRMS table with one row per (lat, lon)."""
    return pd.DataFrame(
        {
            "latitude": [p[0] for p in points],
            "longitude": [p[1] for p in points],
            "brightness": 330.0,
            "scan": 1.0,
            "track": 1.0,
            "acq_date": day.isoformat(),
            "acq_time": 555,
            "satellite": satellite,
            "instrument": "MODIS",
            "confidence": 80,
            "version": "6.03",
            "bright_t31": 290.0,
            "frp": 12.5,
            "daynight": "N",
        }
    )


def viirs_rows(points, day=DAY, satellite="N20"):
    return pd.DataFrame(
        {
            "latitude": [p[0] for p in points],
            "longitude": [p[1] for p in points],
            "bright_ti4": 330.0,
            "scan": 0.4,
            "track": 0.4,
            "acq_date": day.isoformat(),
            "acq_time": 910,
            "satellite": satellite,
            "instrument": "VIIRS",
            "confidence": "n",
            "version": "2",
            "bright_ti5": 290.0,
            "frp": 3.0,
            "daynight": "N",
        }
    )


@pytest.fixture(autouse=True)
def fresh_cache(monkeypatch):
    monkeypatch.setattr(fp, "_cache", {})
    monkeypatch.setattr(fp, "CACHE_DIR", None)


@pytest.fixture
def stub_firms(monkeypatch):
    """Stub FIRMS: answers[source] is a DataFrame or an Exception. Records every call."""
    answers, calls = {}, []

    def fake_fetch_area(source, area, days=1, date=None, **_):
        calls.append((source, area, days, date))
        answer = answers.get(source, pd.DataFrame(columns=firms_client.MODIS_COLUMNS))
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(fp, "data_availability", lambda refresh=False: AVAILABILITY.copy())
    monkeypatch.setattr(fp, "fetch_area", fake_fetch_area)
    return answers, calls


# --- source selection -------------------------------------------------------------------

@pytest.mark.parametrize(
    "day, family, expected",
    [
        (date(2020, 8, 20), "MODIS", ["MODIS_SP"]),
        (date(2026, 8, 31), "MODIS", ["MODIS_SP"]),
        (date(2026, 9, 1), "MODIS", ["MODIS_NRT"]),
        (date(2026, 11, 10), "MODIS", ["MODIS_NRT"]),
        (date(2020, 8, 20), "VIIRS", ["VIIRS_NOAA20_SP"]),  # NOAA-21 starts in 2023
        (date(2026, 7, 15), "VIIRS", ["VIIRS_NOAA20_SP", "VIIRS_NOAA21_SP"]),
        (date(2026, 10, 1), "VIIRS", ["VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT"]),
    ],
)
def test_sources_for_picks_archive_then_near_real_time(stub_firms, day, family, expected):
    assert fp.sources_for(day, family) == expected


def test_sources_for_never_uses_suomi_npp(stub_firms):
    for day in (date(2020, 7, 1), date(2026, 7, 1), date(2026, 10, 1)):
        assert not any("SNPP" in source for source in fp.sources_for(day, "VIIRS"))


def test_sources_for_raises_when_nothing_covers_the_day(stub_firms):
    with pytest.raises(FirmsError):
        fp.sources_for(date(2030, 1, 1), "MODIS")
    with pytest.raises(FirmsError):
        fp.sources_for(date(2015, 7, 1), "VIIRS")  # before NOAA-20


def test_sources_for_rejects_unknown_family(stub_firms):
    with pytest.raises(ValueError):
        fp.sources_for(DAY, "GOES")


# --- normalizing and filtering ------------------------------------------------------------

@pytest.mark.parametrize(
    "raw, expected",
    [("Terra", "Terra"), ("T", "Terra"), ("Aqua", "Aqua"), ("A", "Aqua"), ("N", "SNPP"),
     ("N20", "N20"), ("N21", "N21"), (" n20 ", "N20"), ("Landsat", "Landsat")],
)
def test_normalize_satellite(raw, expected):
    assert fp.normalize_satellite(raw) == expected


def test_fire_points_keeps_one_day_inside_the_box(stub_firms):
    answers, calls = stub_firms
    table = pd.concat(
        [
            modis_rows([(38.5, -122.0)], satellite="T"),
            modis_rows([(38.6, -122.1)], day=DAY + timedelta(days=1)),  # wrong day
            modis_rows([(45.0, -122.1)]),  # outside the box
        ],
        ignore_index=True,
    )
    table["acq_time"] = table["acq_time"].astype(object)
    table.loc[0, "acq_time"] = "0554"  # text with padding still becomes an int
    answers["MODIS_SP"] = table
    points = fp.fire_points(38.0, -123.0, 39.0, -121.0, DAY, "MODIS")
    assert list(points.columns) == fp.POINT_COLUMNS
    assert len(points) == 1
    row = points.iloc[0]
    assert (row.satellite, row.acq_time, row.source, row.acq_date) == ("Terra", 554, "MODIS_SP", "2020-08-20")
    assert calls == [("MODIS_SP", "-123.0000,38.0000,-121.0000,39.0000", 1, DAY)]


def test_fire_points_merges_both_viirs_satellites(stub_firms):
    answers, _ = stub_firms
    day = date(2026, 10, 1)
    answers["VIIRS_NOAA20_NRT"] = viirs_rows([(38.5, -122.0)], day=day, satellite="N20")
    answers["VIIRS_NOAA21_NRT"] = viirs_rows([(38.6, -122.0)], day=day, satellite="N21")
    points = fp.fire_points(38.0, -123.0, 39.0, -121.0, day, "VIIRS")
    assert sorted(points.satellite) == ["N20", "N21"]
    assert sorted(points.source) == ["VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT"]


def test_fire_points_caches_archive_requests(stub_firms):
    answers, calls = stub_firms
    answers["MODIS_SP"] = modis_rows([(38.5, -122.0)])
    fp.fire_points(38.0, -123.0, 39.0, -121.0, DAY, "MODIS")
    fp.fire_points(38.0, -123.0, 39.0, -121.0, DAY, "MODIS")
    assert len(calls) == 1


def test_fire_points_disk_cache(stub_firms, monkeypatch, tmp_path):
    answers, calls = stub_firms
    answers["MODIS_SP"] = modis_rows([(38.5, -122.0)])
    monkeypatch.setattr(fp, "CACHE_DIR", tmp_path)
    first = fp.fire_points(38.0, -123.0, 39.0, -121.0, DAY, "MODIS")
    monkeypatch.setattr(fp, "_cache", {})  # forget memory, keep disk
    second = fp.fire_points(38.0, -123.0, 39.0, -121.0, DAY, "MODIS")
    assert len(calls) == 1 and len(list(tmp_path.iterdir())) == 1
    pd.testing.assert_frame_equal(first, second, check_dtype=False)


def test_fire_points_empty_answer(stub_firms):
    points = fp.fire_points(38.0, -123.0, 39.0, -121.0, DAY, "MODIS")
    assert len(points) == 0 and list(points.columns) == fp.POINT_COLUMNS


# --- the fire mask ------------------------------------------------------------------------

def test_fire_mask_uses_terra_only_by_default(stub_firms):
    answers, _ = stub_firms
    grid = TileGrid.around(*CENTER)
    answers["MODIS_SP"] = pd.concat(
        [modis_rows([CENTER], satellite="Terra"), modis_rows([(grid.lats[10, 10], grid.lons[10, 10])], satellite="Aqua")],
        ignore_index=True,
    )
    mask, info = fp.fire_mask(grid, DAY)
    assert mask.dtype == np.float32 and mask.shape == (64, 64)
    assert mask.sum() == 1 and mask[CENTER_INDEX, CENTER_INDEX] == 1
    assert info["source"] == "MODIS" and info["fallback"] is False
    assert info["satellites_used"] == ["Terra"] and info["n_points"] == 1
    assert info["firms_sources"] == ["MODIS_SP"]

    mask, info = fp.fire_mask(grid, DAY, satellites=("Terra", "Aqua"))
    assert mask.sum() == 2 and info["satellites_used"] == ["Aqua", "Terra"]


def test_fire_mask_falls_back_to_viirs_when_modis_has_nothing(stub_firms):
    answers, _ = stub_firms
    grid = TileGrid.around(*CENTER)
    answers["MODIS_SP"] = modis_rows([CENTER], satellite="Aqua")  # Aqua only: not used by default
    answers["VIIRS_NOAA20_SP"] = viirs_rows([CENTER])
    mask, info = fp.fire_mask(grid, DAY)
    assert mask[CENTER_INDEX, CENTER_INDEX] == 1
    assert info["source"] == "VIIRS" and info["fallback"] is True
    assert "MODIS" in info["reason"]


def test_fire_mask_falls_back_to_viirs_when_modis_fails(stub_firms):
    answers, _ = stub_firms
    grid = TileGrid.around(*CENTER)
    answers["MODIS_SP"] = FirmsError("FIRMS returned HTTP 503: down")
    answers["VIIRS_NOAA20_SP"] = viirs_rows([CENTER])
    mask, info = fp.fire_mask(grid, DAY)
    assert mask.sum() == 1 and info["source"] == "VIIRS" and info["fallback"] is True
    assert "HTTP 503" in info["reason"] and info["errors"]


def test_fire_mask_zeros_when_nothing_found(stub_firms):
    grid = TileGrid.around(*CENTER)
    mask, info = fp.fire_mask(grid, DAY)
    assert mask.sum() == 0 and mask.dtype == np.float32
    assert info["source"] == "MODIS" and info["n_points"] == 0 and info["fallback"] is False
    assert info["reason"].startswith("no detections")


def test_fire_mask_raises_when_every_request_fails(stub_firms):
    # Failing to reach FIRMS is not the same as "no fire today": it must not become a zero mask
    answers, _ = stub_firms
    answers["MODIS_SP"] = FirmsError("MODIS broke")
    answers["VIIRS_NOAA20_SP"] = FirmsError("VIIRS broke")
    with pytest.raises(FirmsError, match="MODIS broke.*VIIRS broke"):
        fp.fire_mask(TileGrid.around(*CENTER), DAY)


def test_fire_mask_one_failure_and_no_detections_is_zero_with_reason(stub_firms):
    answers, _ = stub_firms
    answers["MODIS_SP"] = FirmsError("MODIS broke")
    answers["VIIRS_NOAA20_SP"] = viirs_rows([])
    mask, info = fp.fire_mask(TileGrid.around(*CENTER), DAY)
    assert mask.sum() == 0 and "MODIS broke" in info["reason"]


def test_sources_for_refreshes_stale_availability(monkeypatch):
    # The cached ranges end yesterday; a refresh shows today's NRT data
    stale = pd.DataFrame({"data_id": ["MODIS_NRT"], "min_date": [date(2026, 9, 1)], "max_date": [date(2026, 11, 11)]})
    fresh = pd.DataFrame({"data_id": ["MODIS_NRT"], "min_date": [date(2026, 9, 1)], "max_date": [date(2026, 11, 12)]})
    calls = []
    monkeypatch.setattr(fp, "data_availability", lambda refresh=False: calls.append(refresh) or (fresh if refresh else stale))
    assert fp.sources_for(date(2026, 11, 12), "MODIS") == ["MODIS_NRT"]
    assert calls == [False, True]


def test_fire_mask_prefer_viirs(stub_firms):
    answers, _ = stub_firms
    grid = TileGrid.around(*CENTER)
    answers["MODIS_SP"] = modis_rows([CENTER])
    answers["VIIRS_NOAA20_SP"] = viirs_rows([(grid.lats[3, 4], grid.lons[3, 4])])
    mask, info = fp.fire_mask(grid, DAY, prefer="VIIRS")
    assert info["source"] == "VIIRS" and info["fallback"] is False and mask[3, 4] == 1 and mask.sum() == 1


def test_fire_mask_queries_a_padded_box(stub_firms):
    answers, calls = stub_firms
    grid = TileGrid.around(*CENTER)
    answers["MODIS_SP"] = modis_rows([CENTER])
    fp.fire_mask(grid, DAY)
    south, west, north, east = grid.bounds
    box = [float(v) for v in calls[0][1].split(",")]
    assert box[0] < west and box[1] < south and box[2] > east and box[3] > north
    assert abs((west - box[0]) - fp.QUERY_PAD_DEG) < 1e-3


def test_fire_mask_with_given_points_makes_no_requests(stub_firms):
    _, calls = stub_firms
    grid = TileGrid.around(*CENTER)
    modis = fp._tidy(modis_rows([CENTER, (grid.lats[0, 63], grid.lons[0, 63])]), "MODIS_SP", DAY, -90, -180, 90, 180)
    mask, info = fp.fire_mask(grid, DAY, points={"MODIS": modis})
    assert calls == []
    assert mask.sum() == 2 and mask[0, 63] == 1 and info["firms_sources"] == ["MODIS_SP"]
    # A family that isn't given counts as unavailable
    empty = modis.iloc[0:0]
    mask, info = fp.fire_mask(grid, DAY, points={"MODIS": empty})
    assert calls == [] and mask.sum() == 0 and "VIIRS" in info["reason"]


@pytest.mark.parametrize("row0_is_south", [True, False])
def test_fire_mask_matches_grid_cells(stub_firms, row0_is_south):
    answers, _ = stub_firms
    grid = TileGrid.around(*CENTER, row0_is_south=row0_is_south)
    rng = np.random.default_rng(0)
    south, west, north, east = grid.bounds
    lats = rng.uniform(south - 0.1, north + 0.1, 200)
    lons = rng.uniform(west - 0.1, east + 0.1, 200)
    answers["MODIS_SP"] = modis_rows(list(zip(lats, lons)))
    mask, info = fp.fire_mask(grid, DAY, footprint_scale=0)  # centre pixels only: tests alignment
    rows, cols = grid.cell_of(lats, lons)
    expected = np.zeros((64, 64), dtype=np.float32)
    expected[rows[rows >= 0], cols[rows >= 0]] = 1
    assert np.array_equal(mask, expected)
    assert info["n_points"] == int((rows >= 0).sum())
    # The northern edge must be the last row when south-up, the first when north-up
    north_row = 63 if row0_is_south else 0
    assert grid.lats[north_row, 0] > grid.lats[63 - north_row, 0]


def test_fire_mask_rejects_unknown_family(stub_firms):
    with pytest.raises(ValueError):
        fp.fire_mask(TileGrid.around(*CENTER), DAY, prefer="GOES")


# --- firms_client ------------------------------------------------------------------------

class FakeResponse:
    def __init__(self, text, status_code=200):
        self.text = text
        self.status_code = status_code


def test_fetch_viirs_builds_the_same_url_as_before(monkeypatch):
    seen = []

    def fake_get(url, timeout):
        seen.append(url)
        return FakeResponse("latitude,longitude\n1,2\n")

    monkeypatch.setattr(firms_client.requests, "get", fake_get)
    table = firms_client.fetch_viirs(map_key="KEY")
    assert seen == [f"{firms_client.FIRMS_AREA_URL}/KEY/VIIRS_NOAA20_NRT/-125,24,-66,50/2"]
    assert len(table) == 1
    firms_client.fetch_viirs(area="-1,2,3,4", days=5, source="VIIRS_NOAA21_NRT", map_key="KEY")
    assert seen[-1] == f"{firms_client.FIRMS_AREA_URL}/KEY/VIIRS_NOAA21_NRT/-1,2,3,4/5"
    with pytest.raises(FirmsError):
        firms_client.fetch_viirs(source="MODIS_NRT", map_key="KEY")  # unchanged: NRT VIIRS only


def test_build_area_url():
    url = firms_client.build_area_url("KEY", "MODIS_SP", "-123,36,-118,41", 1, date(2020, 8, 20))
    assert url == f"{firms_client.FIRMS_AREA_URL}/KEY/MODIS_SP/-123,36,-118,41/1/2020-08-20"
    assert firms_client.build_area_url("KEY", "MODIS_NRT", "1,2,3,4", 3).endswith("/MODIS_NRT/1,2,3,4/3")
    assert firms_client.build_area_url("KEY", "VIIRS_NOAA20_SP", "1,2,3,4", 1, "2020-08-20").endswith("/1/2020-08-20")
    for bad_days in (0, 6, 10, 1.5):
        with pytest.raises(FirmsError):
            firms_client.build_area_url("KEY", "MODIS_SP", "1,2,3,4", bad_days)
    with pytest.raises(FirmsError):
        firms_client.build_area_url("KEY", "GOES_NRT", "1,2,3,4", 1)
    with pytest.raises(FirmsError):
        firms_client.build_area_url("KEY", "MODIS_SP", "1,2,3,4", 1, "20-08-2020")


def test_fetch_area_empty_and_errors_hide_the_key(monkeypatch):
    answers = iter([
        FakeResponse(""),
        FakeResponse("Invalid MAP_KEY SECRETKEY", 400),
        FakeResponse("Invalid MAP_KEY SECRETKEY"),
    ])
    monkeypatch.setattr(firms_client.requests, "get", lambda url, timeout: next(answers))
    empty = firms_client.fetch_area("MODIS_SP", "1,2,3,4", 1, "2020-08-20", map_key="SECRETKEY")
    assert len(empty) == 0 and list(empty.columns) == firms_client.MODIS_COLUMNS
    for _ in range(2):
        with pytest.raises(FirmsError) as error:
            firms_client.fetch_area("MODIS_SP", "1,2,3,4", 1, map_key="SECRETKEY")
        assert "SECRETKEY" not in str(error.value) and "<key>" in str(error.value)


def test_data_availability_parses_dates_and_caches(monkeypatch):
    monkeypatch.setattr(firms_client, "_availability", {"table": None, "fetched_at": 0.0})
    calls = []

    def fake_get(url, timeout):
        calls.append(url)
        return FakeResponse("data_id,min_date,max_date\nMODIS_SP,2000-11-01,2026-08-31\n")

    monkeypatch.setattr(firms_client.requests, "get", fake_get)
    first = firms_client.data_availability(map_key="KEY")
    second = firms_client.data_availability(map_key="KEY")
    assert len(calls) == 1 and calls[0].endswith("/data_availability/csv/KEY/all")
    assert first.loc[0, "max_date"] == date(2026, 8, 31)
    pd.testing.assert_frame_equal(first, second)


# --- real FIRMS (run with --network) --------------------------------------------------------

@pytest.mark.network
def test_network_modis_archive_california():
    table = firms_client.fetch_area("MODIS_SP", "-123,36,-118,41", 1, "2020-08-20")
    assert len(table) > 1000
    assert set(table["satellite"]) == {"Terra", "Aqua"}
    assert set(table["acq_date"]) == {"2020-08-20"}


@pytest.mark.network
def test_network_modis_near_real_time_satellite_names():
    yesterday = datetime.now(timezone.utc).date() - timedelta(days=1)
    points = fp.fire_points(24, -125, 50, -66, yesterday, "MODIS")
    assert len(points) > 0
    assert set(points["satellite"]) <= {"Terra", "Aqua"}
    assert set(points["source"]) == {"MODIS_NRT"}


@pytest.mark.network
def test_network_fire_mask_on_a_real_fire():
    grid = TileGrid.around(*CENTER)
    mask, info = fp.fire_mask(grid, DAY)
    assert info["source"] == "MODIS" and info["fallback"] is False and info["firms_sources"] == ["MODIS_SP"]
    assert mask[CENTER_INDEX, CENTER_INDEX] == 1  # the detection the tile is centred on
    assert mask.sum() >= 10  # the LNU complex was large that day


@pytest.mark.network
def test_network_viirs_archive():
    points = fp.fire_points(36, -123, 41, -118, DAY, "VIIRS")
    assert len(points) > 1000
    assert set(points["satellite"]) == {"N20"} and set(points["source"]) == {"VIIRS_NOAA20_SP"}


def test_key_is_redacted_even_when_it_straddles_the_message_cut(monkeypatch):
    key = "abcdef0123456789abcdef0123456789"
    body = "x" * 187 + key + " is not a valid MAP_KEY"

    class Answer:
        status_code = 400
        text = body

    monkeypatch.setattr(firms_client.requests, "get", lambda url, timeout: Answer())
    with pytest.raises(FirmsError) as caught:
        firms_client.fetch_area("MODIS_NRT", "-125,24,-66,50", map_key=key)
    message = str(caught.value)
    assert key[:8] not in message and key[-8:] not in message and "abcdef0123456" not in message



# --- detection footprints ------------------------------------------------------------------


def _one_detection(grid, row, col, scan, track):
    lat, lon = float(grid.lats[row, col]), float(grid.lons[row, col])
    return pd.DataFrame({"latitude": [lat], "longitude": [lon], "scan": [scan], "track": [track]}), lat, lon


def test_footprint_covers_the_scaled_rectangle_and_nothing_else():
    grid = TileGrid.around(*CENTER)
    points, lat, lon = _one_detection(grid, 31, 31, scan=4.8, track=2.0)  # edge-of-swath MODIS pixel
    mask = fp.footprint_mask(grid, points, scale=0.5)
    assert mask[31, 31] == 1
    half_ns = 0.5 * 2.0 / 2 / fp.KM_PER_DEG_LAT
    half_ew = 0.5 * 4.8 / 2 / (fp.KM_PER_DEG_LAT * np.cos(np.radians(lat)))
    marked = mask == 1
    # Every marked pixel touches the rectangle (its centre is within half a pixel of it)
    assert np.all(np.abs(grid.lats[marked] - lat) <= half_ns + grid.lat_step / 2 + 1e-9)
    assert np.all(np.abs(grid.lons[marked] - lon) <= half_ew + grid.lon_step / 2 + 1e-9)
    # 2.4 km east-west spans more pixels than 1 km north-south
    rows, cols = np.nonzero(mask)
    assert np.ptp(cols) > np.ptp(rows)


def test_footprint_scale_zero_or_missing_size_is_the_centre_pixel():
    grid = TileGrid.around(*CENTER)
    points, _, _ = _one_detection(grid, 10, 20, scan=4.8, track=2.0)
    assert fp.footprint_mask(grid, points, scale=0).sum() == 1
    no_size = points.assign(scan=np.nan, track=np.nan)
    assert fp.footprint_mask(grid, no_size, scale=0.5).sum() == 1


def test_footprint_of_a_detection_just_outside_marks_the_edge():
    grid = TileGrid.around(*CENTER)
    south, west, north, east = grid.bounds
    points = pd.DataFrame({"latitude": [(south + north) / 2], "longitude": [east + 0.2 * grid.lon_step],
                           "scan": [4.0], "track": [1.0]})
    mask = fp.footprint_mask(grid, points, scale=0.5)
    assert mask[:, -1].sum() >= 1 and mask[:, :-2].sum() == 0


def test_fire_mask_uses_footprints_by_default(stub_firms):
    answers, _ = stub_firms
    grid = TileGrid.around(*CENTER)
    rows = modis_rows([CENTER]).assign(scan=4.8, track=2.0)
    answers["MODIS_SP"] = rows
    mask, info = fp.fire_mask(grid, DAY)
    assert info["footprint_scale"] == fp.FOOTPRINT_SCALE == 0.5
    assert mask.sum() > 1 and info["n_points"] == 1
