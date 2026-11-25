"""
census_data.py
 
Downloads ACS5 block-group and tract level data for Los Angeles County, CA,
merges them, backfills unreliable block-group estimates with tract-level
estimates (based on coefficient of variation), and engineers a set of
derived features.
 
Usage from another script:
 
    from census_data import get_census_data

    df = get_census_data()          # downloads fresh data (uses cache if available)
    df = get_census_data(force_refresh=True)   # ignore cache, re-download

IMPORTANT (GeoParquet):
    The cached file written by this module is GeoParquet — geometry is
    stored as WKB with CRS metadata that only a geospatial-aware reader
    understands. NEVER read the cache file with plain `pandas.read_parquet()`
    directly; the geometry column will silently come back as raw WKB bytes
    with no CRS attached, and no error will be raised to tell you so. Always
    go through `get_census_data()`, or if you must read the cache file
    directly for some reason, use `load_cached_census()` below — both routes
    use `geopandas.read_parquet()` under the hood and validate the result.

Set CENSUS_CACHE_PATH / USE_CACHE below to control caching behavior.
"""
 
from pathlib import Path
 
import numpy as np
import pandas as pd
import geopandas as gpd
 
import censusdis.data as ced
from censusdis.datasets import ACS5
from censusdis.states import CA
from censusdis.counties.california import LOS_ANGELES
 
from api_keys import census_api
 
# ==========================================
# CONFIG
# ==========================================
YEAR = 2024
CV_THRESHOLD = 0.3  # coefficient-of-variation threshold for reliability
 
# Optional on-disk cache so repeated imports don't re-hit the Census API.
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_PATH = CACHE_DIR / f"merged_census_data_{YEAR}.parquet"
USE_CACHE = True
 
 
# ==========================================
# VARIABLE LISTS / RENAME MAPS
# ==========================================
SELECTED_BLOCK_ACS_VARIABLES = [
    # Core Demographics & Denominators
    "B19013_001E",  # Median Household Income
    "B19013_001M",  # Median Household Income MOE
    "B25010_001E",  # Average Household Size (Total)
    "B25010_001M",  # Average Household Size (Total) MOE
    "B01003_001E",  # Total Population
    "B11001_001E",  # Total Households
 
    # Race & Ethnicity (Mutually Exclusive)
    "B02001_002E",  # White alone
    "B02001_003E",  # Black or African American alone
    "B02001_004E",  # American Indian and Alaska Native alone
    "B02001_005E",  # Asian alone
    "B02001_006E",  # Native Hawaiian and Other Pacific Islander alone
    "B02001_007E",  # Some Other Race alone
    "B02001_008E",  # Two or More Races

    # Housing: tenure (B25003), units in structure (B25024), rent (B25064), rent burden (B25070)
    "B25003_001E", "B25003_002E", "B25003_003E",
    "B25024_001E", "B25024_002E", "B25024_003E", "B25024_004E", "B25024_005E", "B25024_006E",
    "B25024_007E", "B25024_008E", "B25024_009E", "B25024_010E", "B25024_011E",
    "B25064_001E",  # Median gross rent
    "B25064_001M",  # Median gross rent MOE
    "B25070_001E", "B25070_002E", "B25070_003E", "B25070_004E", "B25070_005E", "B25070_006E",
    "B25070_007E", "B25070_008E", "B25070_009E", "B25070_010E", "B25070_011E",
]
 
# Variables at the Census Tract level
SELECTED_TRACT_ACS_VARIABLES = [
    "B19013_001E",  # Median Household Income
    "B25010_001E",  # Average Household Size (Total)
    "B25064_001E",  # Median gross rent (tract fallback for unreliable block-group estimates)
 
    # Means of Transportation to Work (Disaggregated)
    "B08006_001E",  # Total Workers 16+
    "B08006_003E",  # Drove alone
    "B08006_005E",  # Carpooled: 2-person
    "B08006_006E",  # Carpooled: 3-person
    "B08006_007E",  # Carpooled: 4-or-more-person
    "B08006_009E",  # Public Transit: Bus
    "B08006_010E",  # Public Transit: Subway or elevated rail
    "B08006_011E",  # Public Transit: Long-distance or commuter rail
    "B08006_012E",  # Public Transit: Light rail, streetcar, or trolley
    "B08006_013E",  # Public Transit: Ferryboat
    "B08006_014E",  # Bicycle
    "B08006_015E",  # Walked
    "B08006_016E",  # Taxi, motorcycle, or other means
    "B08006_017E",  # Worked from home
 
    # Work Location / Commute Flows
    "B08007_003E",  # Worked in county of residence
    "B08007_004E",  # Worked outside county of residence
 
    # Travel Time to Work
    "B08012_001E",  # Total Workers (Travel Time Universe)
    "B08012_002E",  # Less than 5 minutes
    "B08012_003E",  # 5 to 9 minutes
    "B08012_004E",  # 10 to 14 minutes
    "B08012_005E",  # 15 to 19 minutes
    "B08012_006E",  # 20 to 24 minutes
    "B08012_007E",  # 25 to 29 minutes
    "B08012_008E",  # 30 to 34 minutes
    "B08012_009E",  # 35 to 39 minutes
    "B08012_010E",  # 40 to 44 minutes
    "B08012_011E",  # 45 to 59 minutes
    "B08012_012E",  # 60 to 89 minutes
    "B08012_013E",  # 90 or more minutes
 
    # Vehicle Exposure & Workers Per Household
    "B25046_001E",  # Aggregate vehicles (Total)
    "B25046_002E",  # Aggregate vehicles (Owner-occupied)
    "B25046_003E",  # Aggregate vehicles (Renter-occupied)
    "B08202_001E",  # Workers per household (Total Universe)
    "B08202_002E",  # 0 workers
    "B08202_003E",  # 1 worker
    "B08202_004E",  # 2 workers
    "B08202_005E",  # 3 or more workers
]
 
# ------------------------------------------
# 1. Core Demographics & Research Targets
# ------------------------------------------
CORE_VARIABLES = {
    "B19013_001E": "median_household_income",
    "B19013_001M": "median_household_income_moe",
    "B25010_001E": "avg_household_size_total",
    "B25010_001M": "avg_household_size_total_moe",
    "B11001_001E": "total_household",
}
 
# ------------------------------------------
# 2. Race & Ethnicity (Mutually Exclusive)
# Universe: Total Population (B01003_001E)
# ------------------------------------------
RACE_VARIABLES = {
    "B01003_001E": "total_population",
    "B02001_002E": "pop_white_alone",
    "B02001_003E": "pop_black_alone",
    "B02001_004E": "pop_aian_alone",
    "B02001_005E": "pop_asian_alone",
    "B02001_006E": "pop_nhpi_alone",
    "B02001_007E": "pop_other_race_alone",
    "B02001_008E": "pop_two_or_more_races",  # Excludes 009E/010E to avoid double-counting
}
 
# ------------------------------------------
# 3. Means of Transportation to Work
# Universe: Total Workers 16+ (B08006_001E)
# ------------------------------------------
COMMUTE_VARIABLES = {
    "B08006_001E": "commute_total_workers",
    "B08006_003E": "commute_drove_alone",
    "B08006_005E": "commute_carpool_2p",
    "B08006_006E": "commute_carpool_3p",
    "B08006_007E": "commute_carpool_4p_plus",
    "B08006_009E": "commute_transit_bus",
    "B08006_010E": "commute_transit_subway",
    "B08006_011E": "commute_transit_commuter_rail",
    "B08006_012E": "commute_transit_light_rail",
    "B08006_013E": "commute_transit_ferry",
    "B08006_014E": "commute_bicycle",
    "B08006_015E": "commute_walked",
    "B08006_016E": "commute_taxi_motorcycle_other",
    "B08006_017E": "commute_worked_from_home",
}
 
# ------------------------------------------
# 4. Work Location / Commute Flows
# Universe: Total Workers 16+
# ------------------------------------------
WORK_LOCATION_VARIABLES = {
    "B08007_003E": "work_in_county_of_residence",
    "B08007_004E": "work_outside_county_of_residence",
}
 
# ------------------------------------------
# 5. Travel Time to Work (Disaggregated Bins)
# Universe: Workers 16+ Who Did Not Work At Home
# ------------------------------------------
TRAVEL_TIME_VARIABLES = {
    "B08012_001E": "travel_time_total_workers",
    "B08012_002E": "travel_time_under_5_min",
    "B08012_003E": "travel_time_5_to_9_min",
    "B08012_004E": "travel_time_10_to_14_min",
    "B08012_005E": "travel_time_15_to_19_min",
    "B08012_006E": "travel_time_20_to_24_min",
    "B08012_007E": "travel_time_25_to_29_min",
    "B08012_008E": "travel_time_30_to_34_min",
    "B08012_009E": "travel_time_35_to_39_min",
    "B08012_010E": "travel_time_40_to_44_min",
    "B08012_011E": "travel_time_45_to_59_min",
    "B08012_012E": "travel_time_60_to_89_min",
    "B08012_013E": "travel_time_90_plus_min",
}
 
# ------------------------------------------
# 6. Vehicle Exposure & Workers Per Household
# ------------------------------------------
VEHICLE_AND_WORKER_VARIABLES = {
    "B25046_001E": "agg_vehicles_total",
    "B25046_002E": "agg_vehicles_owner",
    "B25046_003E": "agg_vehicles_renter",
    "B08202_001E": "workers_per_hh_total",
    "B08202_002E": "workers_per_hh_0",
    "B08202_003E": "workers_per_hh_1",
    "B08202_004E": "workers_per_hh_2",
    "B08202_005E": "workers_per_hh_3_plus",
}
 
# ------------------------------------------
# 7. Housing: tenure, units in structure, rent, rent burden (block-group level)
# Counts only; ratios (renter share, multi-unit share, rent-burden share) are computed at the
# DISTRICT level from summed counts, never averaged from block-group ratios.
# ------------------------------------------
HOUSING_VARIABLES = {
    # B25003 Tenure (universe: occupied housing units)
    "B25003_001E": "occupied_units",
    "B25003_002E": "owner_occupied",
    "B25003_003E": "renter_occupied",
    # B25024 Units in structure (universe: housing units)
    "B25024_001E": "units_total",
    "B25024_002E": "units_1_detached",
    "B25024_003E": "units_1_attached",
    "B25024_004E": "units_2",
    "B25024_005E": "units_3_4",
    "B25024_006E": "units_5_9",
    "B25024_007E": "units_10_19",
    "B25024_008E": "units_20_49",
    "B25024_009E": "units_50_plus",
    "B25024_010E": "units_mobile_home",
    "B25024_011E": "units_other",
    # B25064 Median gross rent (universe: renter-occupied units paying cash rent)
    "B25064_001E": "median_gross_rent",
    "B25064_001M": "median_gross_rent_moe",
    # B25070 Gross rent as % of household income (universe: renter-occupied units)
    "B25070_001E": "rb_total",
    "B25070_002E": "rb_lt_10",
    "B25070_003E": "rb_10_14",
    "B25070_004E": "rb_15_19",
    "B25070_005E": "rb_20_24",
    "B25070_006E": "rb_25_29",
    "B25070_007E": "rb_30_34",
    "B25070_008E": "rb_35_39",
    "B25070_009E": "rb_40_49",
    "B25070_010E": "rb_50_plus",
    "B25070_011E": "rb_not_computed",
}

MERGED_RENAME_DICT = (
    CORE_VARIABLES
    | RACE_VARIABLES
    | COMMUTE_VARIABLES
    | WORK_LOCATION_VARIABLES
    | TRAVEL_TIME_VARIABLES
    | VEHICLE_AND_WORKER_VARIABLES
    | HOUSING_VARIABLES
)
 
 
# ==========================================
# DOWNLOAD
# ==========================================
def _download_block_groups(year: int = YEAR) -> gpd.GeoDataFrame:
    gdf_blocks = ced.download(
        ACS5,
        year,
        download_variables=SELECTED_BLOCK_ACS_VARIABLES,
        state=CA,
        county=LOS_ANGELES,
        block_group="*",
        with_geometry=True,
        remove_water=True,
        api_key=census_api,
    )
    return gdf_blocks.rename(columns=MERGED_RENAME_DICT)
 
 
def _download_tracts(year: int = YEAR) -> pd.DataFrame:
    gdf_tracts = ced.download(
        ACS5,
        year,
        download_variables=SELECTED_TRACT_ACS_VARIABLES,
        state=CA,
        county=LOS_ANGELES,
        tract="*",
        with_geometry=False,
        remove_water=True,
        api_key=census_api,
    )
    return gdf_tracts.rename(columns=MERGED_RENAME_DICT)
 
 
# ==========================================
# KEY BUILDING / MERGE
# ==========================================
def _add_tract_keyid(gdf_tracts: pd.DataFrame) -> pd.DataFrame:
    gdf_tracts = gdf_tracts.copy()
    gdf_tracts["STATE"] = gdf_tracts["STATE"].astype(str).str.zfill(2)
    gdf_tracts["COUNTY"] = gdf_tracts["COUNTY"].astype(str).str.zfill(3)
    gdf_tracts["TRACT"] = gdf_tracts["TRACT"].astype(str).str.zfill(6)
    gdf_tracts["keyid"] = gdf_tracts["STATE"] + gdf_tracts["COUNTY"] + gdf_tracts["TRACT"]
    return gdf_tracts.drop(columns=["STATE", "COUNTY", "TRACT"])
 
 
def _add_block_keyid(gdf_blocks: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf_blocks = gdf_blocks.copy()
    gdf_blocks["STATE"] = gdf_blocks["STATE"].astype(str).str.zfill(2)
    gdf_blocks["COUNTY"] = gdf_blocks["COUNTY"].astype(str).str.zfill(3)
    gdf_blocks["TRACT"] = gdf_blocks["TRACT"].astype(str).str.zfill(6)
    gdf_blocks["keyid"] = gdf_blocks["STATE"] + gdf_blocks["COUNTY"] + gdf_blocks["TRACT"]
    return gdf_blocks
 
 
# ==========================================
# FEATURE ENGINEERING
# ==========================================
def _null_special_values(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """ACS marks 'no estimate' with large negative sentinels (e.g. -666666666). Rent and its MOE
    are always positive, so any non-positive value is a sentinel, not data."""
    df = df.copy()
    for c in cols:
        df[c] = df[c].where(df[c] > 0)
    return df


def _add_cv_columns(gdf_blocks: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf_blocks = gdf_blocks.copy()
    gdf_blocks["cv_income"] = (
        gdf_blocks["median_household_income_moe"] / 1.645
    ) / gdf_blocks["median_household_income"]
    gdf_blocks["cv_household_size"] = (
        gdf_blocks["avg_household_size_total_moe"] / 1.645
    ) / gdf_blocks["avg_household_size_total"]
    gdf_blocks["cv_rent"] = (
        gdf_blocks["median_gross_rent_moe"] / 1.645
    ) / gdf_blocks["median_gross_rent"]
    return gdf_blocks
 
 
def _backfill_unreliable_estimates(merge_df: pd.DataFrame, cv_threshold: float = CV_THRESHOLD) -> pd.DataFrame:
    """If block-group CV exceeds threshold (or the value is NaN), fall back to the tract-level estimate."""
    merge_df = merge_df.copy()
 
    condition_income = (
        (merge_df["cv_income"] > cv_threshold)
        | (merge_df["median_household_income_blocks"].isna())
    )
    condition_household = (
        (merge_df["cv_household_size"] > cv_threshold)
        | (merge_df["avg_household_size_total_blocks"].isna())
    )
 
    condition_rent = (
        (merge_df["cv_rent"] > cv_threshold)
        | (merge_df["median_gross_rent_blocks"].isna())
    )
    merge_df["median_gross_rent"] = np.where(
        condition_rent,
        merge_df["median_gross_rent_tracts"],
        merge_df["median_gross_rent_blocks"],
    )

    merge_df["median_income"] = np.where(
        condition_income,
        merge_df["median_household_income_tracts"],
        merge_df["median_household_income_blocks"],
    )
    merge_df["avg_household_size"] = np.where(
        condition_household,
        merge_df["avg_household_size_total_tracts"],
        merge_df["avg_household_size_total_blocks"],
    )
    return merge_df
 
 
def _add_derived_features(merge_df: pd.DataFrame) -> pd.DataFrame:
    merge_df = merge_df.copy()
 
    # Vehicle Exposure Index (vei): physical density of vehicles per household.
    merge_df["vei"] = merge_df["agg_vehicles_total"] / merge_df["total_household"]
 
    # Commuter Driving Ratio (cdr): share of workers who drive/carpool.
    merge_df["cdr"] = (
        merge_df["commute_drove_alone"]
        + merge_df["commute_carpool_2p"]
        + merge_df["commute_carpool_3p"]
        + merge_df["commute_carpool_4p_plus"]
    ) / merge_df["commute_total_workers"]
 
    # Outflow Commuter Index (oci): share of households with 2+ workers.
    merge_df["oci"] = (
        merge_df["work_outside_county_of_residence"]
    ) / merge_df["workers_per_hh_total"]
 
    # Household Worker Intensity (hwi): share of households with 2+ workers.
    merge_df["hwi"] = (
        merge_df["workers_per_hh_2"] + merge_df["workers_per_hh_3_plus"]
    ) / merge_df["workers_per_hh_total"]
 
    # Extreme Commute Exposure (ece): share of workers with 45+ min commutes.
    merge_df["ece"] = (
        merge_df["travel_time_45_to_59_min"]
        + merge_df["travel_time_60_to_89_min"]
        + merge_df["travel_time_90_plus_min"]
    ) / merge_df["travel_time_total_workers"]
 
    return merge_df


# ==========================================
# GEOSPATIAL SAFETY HELPERS
# ==========================================
def _validate_geo(df: pd.DataFrame, context: str = "") -> gpd.GeoDataFrame:
    """
    Fail loudly, at the point of the mistake, if geometry/CRS has been lost.

    Without this check, a GeoDataFrame silently demoted to a plain DataFrame
    (or a CRS silently dropped) would still write to parquet successfully —
    it just wouldn't be usable spatially downstream, and nothing would tell
    you that until some much later, harder-to-diagnose step failed.
    """
    label = f" ({context})" if context else ""
    if not isinstance(df, gpd.GeoDataFrame):
        raise TypeError(
            f"Expected a GeoDataFrame{label} but got {type(df).__name__} — "
            "geometry was lost somewhere upstream."
        )
    if "geometry" not in df.columns:
        raise ValueError(f"GeoDataFrame{label} has no 'geometry' column.")
    if df.crs is None:
        raise ValueError(
            f"GeoDataFrame{label} has no CRS set — do not cache or merge this "
            "downstream until a CRS is assigned."
        )
    return df


def load_cached_census(path: Path = CACHE_PATH) -> gpd.GeoDataFrame:
    """
    The ONLY sanctioned way to read the cache file directly, if you need to
    bypass get_census_data() for some reason (e.g. inspecting the file from
    a notebook). Always use geopandas, never plain pandas, to read this file:
    plain `pandas.read_parquet()` will silently return raw WKB bytes with no
    CRS instead of raising an error, which is worse than a crash.
    """
    df = gpd.read_parquet(path)
    return _validate_geo(df, context=f"loaded from {path}")


# ==========================================
# PUBLIC ENTRY POINT
# ==========================================
def get_census_data(
    year: int = YEAR,
    cv_threshold: float = CV_THRESHOLD,
    use_cache: bool = USE_CACHE,
    force_refresh: bool = False,
) -> gpd.GeoDataFrame:
    """
    Build (or load from cache) the merged block-group + tract ACS5 dataset
    for Los Angeles County, CA, with engineered features.
 
    Parameters
    ----------
    year : int
        ACS5 vintage year to download.
    cv_threshold : float
        Coefficient-of-variation threshold above which a block-group estimate
        is considered unreliable and backfilled with the tract-level value.
    use_cache : bool
        If True, read/write a local parquet cache to avoid re-hitting the API.
    force_refresh : bool
        If True, ignore any existing cache and re-download.
 
    Returns
    -------
    gpd.GeoDataFrame
        Block-group grain, CRS-aware. Always access the cache through this
        function (or load_cached_census()) rather than reading the parquet
        file directly with pandas — see module docstring.
    """
 
    if use_cache and not force_refresh and CACHE_PATH.exists():
        try:
            return load_cached_census(CACHE_PATH)
        except (FileNotFoundError, OSError, TypeError, ValueError) as e:
            # Narrowed from a bare `except Exception`: these are the specific
            # failure modes we know how to recover from (missing/corrupted
            # cache, or a cache file that somehow lost its geometry/CRS).
            # Anything else (e.g. a bug introduced elsewhere) should not be
            # silently swallowed and masked by a "just re-download" fallback.
            print(f"Warning: could not read cache ({type(e).__name__}: {e}). Rebuilding.")
 
    gdf_blocks = _download_block_groups(year)
    gdf_tracts = _download_tracts(year)
 
    gdf_blocks = _null_special_values(gdf_blocks, ["median_gross_rent", "median_gross_rent_moe"])
    gdf_tracts = _null_special_values(gdf_tracts, ["median_gross_rent"])
    gdf_blocks = _add_cv_columns(gdf_blocks)
 
    gdf_tracts = _add_tract_keyid(gdf_tracts)
    gdf_blocks = _add_block_keyid(gdf_blocks)

    # .merge() called on the GeoDataFrame (not the free function pd.merge())
    # so that GeoDataFrame-ness is guaranteed by construction, not just an
    # incidental consequence of which argument pandas happens to dispatch on.
    merge_df = gdf_blocks.merge(gdf_tracts, on="keyid", how="left")
    merge_df = merge_df.rename(
        columns={
            "median_household_income_x": "median_household_income_blocks",
            "avg_household_size_total_x": "avg_household_size_total_blocks",
            "median_household_income_y": "median_household_income_tracts",
            "avg_household_size_total_y": "avg_household_size_total_tracts",
            "median_gross_rent_x": "median_gross_rent_blocks",
            "median_gross_rent_y": "median_gross_rent_tracts",
        }
    )
 
    merge_df = _backfill_unreliable_estimates(merge_df, cv_threshold)
    merge_df = _add_derived_features(merge_df)

    # Defensive check right before caching: fail loudly here if something
    # upstream silently demoted the GeoDataFrame or dropped its CRS, rather
    # than writing a quietly-broken cache file that fails downstream instead.
    merge_df = _validate_geo(merge_df, context="pre-cache-write")
 
    if use_cache:
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            merge_df.to_parquet(CACHE_PATH)
        except Exception as e:
            print(f"Warning: could not write cache ({e}). Continuing without caching.")
 
    return merge_df
 
 
if __name__ == "__main__":
    df = get_census_data(force_refresh=True)
    print(df.shape)
    print(df.head())