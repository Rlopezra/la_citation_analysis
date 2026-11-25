from pathlib import Path

import polars as pl
import datetime


# ==========================================
# CONFIG
# ==========================================
YEAR = 2021
MONTH = 12
DAY = 31
File_Path = "data/Parking_Citations_20260813.csv"

# Optional on-disk cache so repeated imports don't re-hit the Census API.
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_PATH = CACHE_DIR / f"cleaned_parking_citation_data.parquet"
USE_CACHE = True
 
# ==========================================
# VARIABLE LISTS / RENAME MAPS
# ==========================================

COLUMNS_TO_READ = [
    "ticket_number",
    "issue_date",
    "issue_time",
    "rp_state_plate",
    "plate_expiry_date",
    "location",
    "violation_description",
    "fine_amount",
    "agency_desc",
    "body_style_desc",
    "loc_lat",
    "loc_long",
    "geocodelocation",
]
# ==========================================
# Load data
# ==========================================

#Functions to add
# Load data function that has the link and columns to read?
def _load_data(file_path : str = File_Path, columns_to_read: list = COLUMNS_TO_READ) -> pl.DataFrame:
    citation_df = (
            pl.scan_csv(
                file_path,
                schema_overrides={
                    "ticket_number": pl.Utf8,
                    "fine_amount": pl.Utf8,
                    "issue_time": pl.Utf8,
                    "plate_expiry_date": pl.Utf8,
                    "loc_lat": pl.Utf8,
                    "loc_long": pl.Utf8,
                },
            )
            .select(columns_to_read)
            .with_columns(
                pl.col("fine_amount").str.replace_all(",", "").cast(pl.Float64), # strip thousand separator and convert to numeric
                pl.col("issue_date").str.strptime(pl.Date, "%Y %b %d %I:%M:%S %p", strict=False), #convert issue data to actual calendar date
                pl.col("issue_time")
                .str.replace_all(",", "")
                .str.zfill(4)
                .str.strptime(pl.Time, "%H%M", strict=False),
                pl.col("loc_lat").str.replace_all(",", "").cast(pl.Float64, strict=False),
                pl.col("loc_long").str.replace_all(",", "").cast(pl.Float64, strict=False),
            )
            .collect()
        )

    return citation_df
    

 
# ==========================================
# FEATURE ENGINEERING
# ==========================================

def _filter_after_date(citation_data: pl.DataFrame, cutoff: datetime.date = datetime.date(YEAR, MONTH, DAY)) -> pl.DataFrame:
    return citation_data.filter(pl.col("issue_date") > cutoff)



def _lump_agency_desc(citation_data: pl.DataFrame, agency_cutoff: int = 1000) -> pl.DataFrame:
    return citation_data.with_columns(
        pl.when(pl.len().over("agency_desc") >= agency_cutoff)
        .then(pl.col("agency_desc"))
        .otherwise(pl.lit("Other"))
        .alias("agency_desc_lumped")
    )
    
#Function to lump violatoin description
""">= 10,000 (Top 38 Categories): This captures 98.7% of your dataset. Choose this if you are doing machine learning 
or deeper statistical analysis and don't want to lose important mid-tier 
categories like BLK BIKE PATH OR LANE (~11k) or PRIVATE PROPERTY (~11k)."""

def _lump_violation_description(citation_data: pl.DataFrame, violation_cutoff: int = 10000) -> pl.DataFrame:
    return citation_data.with_columns(
        pl.when(pl.col("violation_description").count().over("violation_description") >= violation_cutoff)
        .then(pl.col("violation_description"))
        .otherwise(pl.lit("Other"))
        .alias("violation_description_lumped")
    )


# Nulls makeup around ~4% of the data, so I checked the unique value combinations between body style and 
# body style desc and saw that all of the nulls were codes with no matching description. So filling in with unknown

def _fill_unknown_values(citation_data: pl.DataFrame, cols: list = ["body_style_desc"]) -> pl.DataFrame:
    return citation_data.with_columns(
        [pl.col(col).fill_null("Unknown") for col in cols]
    )

# ==========================================
# PUBLIC ENTRY POINT
# ==========================================

def get_citation_data(
    file_path: str = File_Path,
    year: int = YEAR,
    month: int = MONTH,
    day: int = DAY,
    cols_to_read: list = COLUMNS_TO_READ,
    agency_cutoff: int = 1000,
    violation_cutoff: int = 10000,
    cols_to_fill: list = ["body_style_desc"],
    use_cache: bool = USE_CACHE,
    force_refresh: bool = False,
) -> pl.DataFrame:

    """
    Build (or load from cache) the parking citation data by filtering the tickets after a certain date
    and performing some feature engineering
 
    Parameters
    ----------
    year : int
        The year value to filter parking citation day
    month : int
        The month value to filter parking citation day
    day : int
        The year value to filter parking citation day
    cols_to_read: list
        List of the columns we want to read. This is necessary since the file is very large
    agency_cutoff: int
        Agencies that gave less that # tickets are lumped together
    violation_cutoff: int
        Violation cutoff threshold where values below are lumped togehter
    cols_to_fill: list
        List of columns to where we fill None/NaNs with 'Unknown'
    use_cache : bool
        If True, read/write a local parquet cache to avoid re-hitting the API.
    force_refresh : bool
        If True, ignore any existing cache and re-download.
 
    Returns
    -------
    pl.DataFrame (GeoDataFrame, since block groups carry geometry)
    """
 
    if use_cache and not force_refresh and CACHE_PATH.exists():
        try:
            return pl.read_parquet(CACHE_PATH)
        except Exception as e:
            print(f"Warning: could not read cache ({e}). Rebuilding.")

    parking_citation_data = _load_data(file_path, cols_to_read)
    parking_citation_data = _filter_after_date(parking_citation_data, datetime.date(year, month, day))
    parking_citation_data = _lump_agency_desc(parking_citation_data, agency_cutoff)
    parking_citation_data = _lump_violation_description(parking_citation_data, violation_cutoff)
    parking_citation_data = _fill_unknown_values(parking_citation_data, cols_to_fill)
    
    if use_cache:
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            parking_citation_data.write_parquet(CACHE_PATH)
        except Exception as e:
            print(f"Warning: could not write cache ({e}). Continuing without caching.")
 
    return parking_citation_data

if __name__ == "__main__":
    df = get_citation_data(force_refresh=True)
    print(df.shape)
    print(df.head())