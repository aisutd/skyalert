"""Download everything the flood-prediction model trains on (about 250 MB, into data/).

  - NOAA Storm Events: every reported flood and flash flood, with county and date   -> data/storm_events/
  - CPC daily rain: rain gauge analysis for the lower 48 states, 0.25 degree grid    -> data/cpc_rain/
  - Census county list: each county's centre point and land area                      -> data/counties.txt

Run from the flood-prediction folder:
    python fetch_data.py                 # 2015-2024
    python fetch_data.py --years 2010 2024
Files already downloaded are skipped.
"""
import argparse
import io
import os
import re
import zipfile

import requests

EVENTS_URL = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
RAIN_URL = "https://downloads.psl.noaa.gov/Datasets/cpc_us_precip/RT/precip.V1.0.{year}.nc"  # 2007 onward
COUNTIES_URL = "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2023_Gazetteer/2023_Gaz_counties_national.zip"


def download(url, dest):
    if os.path.exists(dest):
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    with open(dest + ".part", "wb") as f:
        f.write(r.content)
    os.replace(dest + ".part", dest)  # no half-written files if stopped
    print(f"  {dest}  ({len(r.content) / 1e6:.1f} MB)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs=2, default=[2015, 2024], metavar=("FIRST", "LAST"))
    a = ap.parse_args()
    years = range(a.years[0], a.years[1] + 1)

    listing = requests.get(EVENTS_URL, timeout=60).text  # file names carry a changing "created" date, so look them up
    for y in years:
        names = sorted(set(re.findall(rf"StormEvents_details-ftp_v1\.0_d{y}_c\d+\.csv\.gz", listing)))
        if not names:
            raise SystemExit(f"no Storm Events file for {y}")
        download(EVENTS_URL + names[-1], f"data/storm_events/details_{y}.csv.gz")
        download(RAIN_URL.format(year=y), f"data/cpc_rain/precip_{y}.nc")

    if not os.path.exists("data/counties.txt"):
        z = zipfile.ZipFile(io.BytesIO(requests.get(COUNTIES_URL, timeout=120).content))
        with open("data/counties.txt", "wb") as f:
            f.write(z.read(z.namelist()[0]))
        print("  data/counties.txt", flush=True)
    print("done")


if __name__ == "__main__":
    main()
