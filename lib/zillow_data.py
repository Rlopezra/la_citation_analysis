import math
from pathlib import Path

import pandas as pd

# ==========================================
# CONFIG
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]      # lib/ -> project root, independent of the working directory
FILE_PATH = PROJECT_ROOT / "data" / "zillow_rental_data.csv"

# Zillow labels ZIPs by neighborhood or neighboring city (e.g. 91307 "Bell Canyon", 90061 "Willowbrook"),
# so filtering on City == "Los Angeles" drops ZIPs that cover city households. Filter on the county instead;
# the spatial overlay in build_district_table decides which ZIPs actually matter to each district.
COUNTY_NAME = "Los Angeles County"

# A ZIP-year is kept if it has at least this fraction of the months that exist in the file for that
# year (12 for a full year -> 6 months; 7 for a partial 2026 -> 4 months). This replaces the old rule that
# dropped any ZIP with a single missing month since 2022, which removed ZIPs that hold many households in
# the lower-income districts (e.g. 91342, 90011, 90003) and left some districts with 0% rent coverage.
MIN_MONTH_FRACTION = 0.5

# Optional on-disk cache so repeated imports don't rebuild the dataset.
# The version tag is part of the file name so a cache built under an older rule is never reused.
CACHE_VERSION = "minmonths"
CACHE_DIR = Path(__file__).resolve().parent / "cache"
CACHE_PATH = CACHE_DIR / f"zillow_rental_data_{CACHE_VERSION}.parquet"
USE_CACHE = True

ID_COLS = ["RegionID", "SizeRank", "RegionName", "RegionType", "StateName", "State", "City", "Metro", "CountyName"]

# ==========================================
# Load data
# ==========================================

def _load_data(path: Path = FILE_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def _la_rental_data(df: pd.DataFrame) -> pd.DataFrame:
    # Filter for LA County (not the City label; see COUNTY_NAME above)
    la_rental = df[(df["State"] == "CA") & (df["CountyName"] == COUNTY_NAME)].copy()

    # Grab 2020s column names as a list
    date_2020s_cols = [col for col in la_rental.columns if col.startswith(("2022", "2023", "2024", "2025", "2026"))]

    # Filter the DataFrame with BOTH metadata and dates together
    la_2020s = la_rental[ID_COLS + date_2020s_cols]

    return la_2020s


def _pivot_long_df(df: pd.DataFrame) -> pd.DataFrame:
    date_2020s_cols = [c for c in df.columns if c not in ID_COLS]

    la_long = df.melt(
        id_vars=ID_COLS,           # Non-numeric data
        value_vars=date_2020s_cols,  # Date columns
        var_name="Date",           # New column name for dates
        value_name="Rent_Price"    # New column name for rental prices
    )

    # Convert Date column to datetime format
    la_long["Date"] = pd.to_datetime(la_long["Date"])

    # Extract Year for aggregation
    la_long["Year"] = la_long["Date"].dt.year
    la_long = la_long.sort_values(by=["RegionName", "Date"]).reset_index(drop=True)

    # Missing months are left missing: nothing is imputed. In hyper-polarized markets like Los Angeles,
    # applying a single citywide central tendency to missing periods introduces severe selection bias,
    # artificially pulling lower-income areas up and high-value pockets down. ZIPs are not dropped for a
    # few missing months either; instead each ZIP-year must meet a minimum-months rule (see
    # _calculate_yearly_average).
    return la_long


def _calculate_yearly_average(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"RegionName": "ZipCode"})

    # months that exist in the file for each year (12 for full years, fewer for a partial current year)
    months_available = df.groupby("Year")["Date"].nunique()

    yearly_summary = (
        df.groupby(["ZipCode", "Year"])["Rent_Price"]
        .agg(
            Average_Rent="mean",        # mean of the months that exist
            Total_Rent="sum",
            Months_Recorded="count"     # non-missing months
        )
        .reset_index()
    )

    min_months = yearly_summary["Year"].map(lambda y: math.ceil(MIN_MONTH_FRACTION * months_available[y]))
    enough = yearly_summary["Months_Recorded"] >= min_months

    # ZIP-years without enough months get no average (NaN), not a misleading one from 1-2 months.
    # Months_Recorded is kept as-is so the reason stays visible.
    yearly_summary["Average_Rent"] = yearly_summary["Average_Rent"].where(enough)
    yearly_summary["Total_Rent"] = yearly_summary["Total_Rent"].where(enough)
    return yearly_summary


def _yearly_average_long(df: pd.DataFrame) -> pd.DataFrame:
    df_wide = df.pivot(
        index="ZipCode",
        columns="Year",
        values=["Average_Rent", "Total_Rent", "Months_Recorded"]
    )

    # Flatten the MultiIndex headers
    df_wide.columns = [f"{metric}_{year}" for metric, year in df_wide.columns]

    # Push 'ZipCode' back from index to column
    df_wide = df_wide.reset_index()

    assert df_wide["ZipCode"].is_unique, "one row per ZIP expected"
    return df_wide

# ==========================================
# PUBLIC ENTRY POINT
# ==========================================
def get_zillow_data(
    use_cache: bool = USE_CACHE,
    force_refresh: bool = False
) -> pd.DataFrame:
    """
    Build (or load from cache) the Zillow rental data for Los Angeles County, CA.

    Returns one row per ZIP with Average_Rent_YYYY, Total_Rent_YYYY and Months_Recorded_YYYY columns.
    Average_Rent_YYYY is NaN for ZIPs with too few months that year (see MIN_MONTH_FRACTION).

    Parameters
    ----------
    use_cache : bool
        If True, read/write a local parquet cache to avoid rebuilding the dataset.
    force_refresh : bool
        If True, ignore any existing cache and rebuild from the raw CSV.

    Returns
    -------
    pd.DataFrame
    """

    if use_cache and not force_refresh and CACHE_PATH.exists():
        try:
            return pd.read_parquet(CACHE_PATH)
        except Exception as e:
            print(f"Warning: could not read cache ({e}). Rebuilding.")

    zillow_data = _load_data(FILE_PATH)
    la_zillow_data = _la_rental_data(zillow_data)
    la_zillow_data = _pivot_long_df(la_zillow_data)
    la_zillow_data = _calculate_yearly_average(la_zillow_data)
    la_zillow_data = _yearly_average_long(la_zillow_data)

    if use_cache:
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            la_zillow_data.to_parquet(CACHE_PATH)
        except Exception as e:
            print(f"Warning: could not write cache ({e}). Continuing without caching.")

    return la_zillow_data

if __name__ == "__main__":
    df = get_zillow_data(force_refresh=True)
    print(df.shape)
    print(df.head())