import pandas as pd
from pathlib import Path

REQUIRED_COLUMNS = ["latitude", "longitude", "acq_date", "acq_time", "satellite", "confidence", "frp", "bright_ti4"]
DUPLICATE_KEY = ["latitude", "longitude", "acq_date", "acq_time", "satellite"]
# Measurements that must be numbers. A bad value becomes missing (NaN, sent as null),
# and the row is kept: the fire is still real even if one reading is missing.
NUMERIC_COLUMNS = ["frp", "bright_ti4", "bright_ti5"]

def clean_viirs(raw_df):
    """
    Cleans the VIIRS fire data by performing the following steps:
    1. Checks if the required columns are present in the dataframe.
    2. Converts latitude and longitude columns to numeric, invalid values become NaN.
    3. Drops rows with blank latitude and longitude values.
    4. Drops rows with invalid latitude and longitude values (latitude must be between -90 and 90, longitude must be between -180 and 180).
    5. Drops rows with no detection date or time, since a detection can't be placed in time without them.
    6. Drops duplicate rows based on the specified duplicate key, keeping the first occurrence.
    7. Converts the measurement columns (frp, bright_ti4, bright_ti5) to numbers; invalid values become NaN.
    8. Returns the cleaned dataframe.

    """
    df = raw_df.copy()  # Create a copy of the dataframe to avoid modifying the original
    n = len(df)

    #quits if required columns are not present in the dataframe
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}") #stops the program and prints the missing columns
    
    #convert latitude and longitude columns to numeric, invalid values become NaN
    df["latitude"] = pd.to_numeric(df["latitude"], errors='coerce') #changes latitude column to numeric, invalid values become NaN
    df["longitude"] = pd.to_numeric(df["longitude"], errors='coerce') #changes longitude column to numeric, invalid values become NaN

    print(f"Rows before: {n}")

    #drop blank latitude and longtitude rows
    df = df.dropna(subset=['latitude', 'longitude'])
    print(f"Blank data: {n - len(df)}")
    n = len(df)

    #drop rows with invalid latitude and longitude values
    df = df[df["latitude"].abs() <= 90]
    df = df[df["longitude"].abs() <= 180]
    print(f"Invalid locations: {n - len(df)}")
    n = len(df)

    #drop rows without a usable date or time. acq_time is HHMM in UTC, e.g. 815 or "0815"
    df["acq_time"] = pd.to_numeric(df["acq_time"], errors='coerce')
    df = df[df["acq_date"].notna() & df["acq_time"].between(0, 2359)]
    df["acq_time"] = df["acq_time"].astype(int)
    print(f"Missing date or time: {n - len(df)}")
    n = len(df)

    #drop duplicate rows based on the specified duplicate key, keeping the first occurrence
    df = df.drop_duplicates(subset=DUPLICATE_KEY, keep='first')
    print(f"Duplicate rows: {n - len(df)}")
    n = len(df)

    #measurements become numbers; a bad value becomes NaN but the detection is kept
    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors='coerce')
    print(f"Missing FRP values (rows kept): {int(df['frp'].isna().sum())}")

    print(f"Rows after: {n}")
    return df

if __name__ == "__main__":
    #import data to data variable from sample.csv file
    csv_path = Path(__file__).parent.parent / "sample.csv"
    data = pd.read_csv(csv_path, dtype={"acq_time": str})
    #load into dataframe
    data_frame = pd.DataFrame(data)
    cleaned_data = clean_viirs(data_frame)
    print(cleaned_data)