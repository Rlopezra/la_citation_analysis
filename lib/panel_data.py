"""
panel_data.py

Builds the district x year panel used in the negative binomial regression:

  census + Zillow rent  -> district table   (build_district_table)
  parking citations     -> district x year  (citations_by_district, annual_rates)
  merged on District.

Usage:
    from lib.panel_data import get_panel_data

    panel_df = get_panel_data()                      # builds, or loads the cache
    panel_df = get_panel_data(force_refresh=True)    # rebuild the panel and overwrite its cache

Notes:
  * The panel is a plain DataFrame (no geometry), so its cache is read with pandas.
    The census cache is GeoParquet and must be read with geopandas; this one must not be.
  * The cache file name includes the years, so different year ranges never share a cache.
  * A cache that is missing required columns (for example one built before new census
    fields were added) is rejected and rebuilt automatically.
  * force_refresh rebuilds the panel only. The census and citation loaders keep their own
    caches; call get_census_data(force_refresh=True) first if the Census data must be re-downloaded.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

from lib.census_data import get_census_data
from lib.zillow_data import get_zillow_data
from lib.parking_data import get_citation_data
from lib.district_pipeline import build_district_table
from lib.citation_join import citations_by_district, annual_rates

# ==========================================
# CONFIG
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]          # lib/ -> project root, independent of the working directory
SHAPEFILE_DIR = PROJECT_ROOT / "data" / "shapefiles"
DISTRICT_SHP = SHAPEFILE_DIR / "LA_City_Council_Districts" / "LA_City_Council_Districts_(Adopted_2021).shp"
ZIP_SHP = SHAPEFILE_DIR / "LA_County_ZIP_Codes" / "LA_County_ZIP_Codes_(Hosted).shp"

CACHE_DIR = Path(__file__).resolve().parent / "cache"
USE_CACHE = True

DEFAULT_YEARS = (2022, 2023, 2024, 2025)    # complete citation years; the data ends mid-2026
N_DISTRICTS = 15

# Columns build_district_table must return. If any are missing, lib/district_pipeline.py is out of date.
DISTRICT_COLUMNS = [
    "median_income", "avg_household_size", "rent_2025", "coverage_2025",
    "renter_share", "multiunit_share", "median_gross_rent", "rent_burden_share",
    "transit_commute_share",
]
# Columns every valid panel must have. A cache without them was built by an older pipeline.
REQUIRED_COLUMNS = ["District", "year", "tickets", "tickets_per_household", "tickets_per_capita"] + DISTRICT_COLUMNS


def _cache_path(years: tuple[int, ...]) -> Path:
    return CACHE_DIR / ("panel_data_" + "_".join(str(y) for y in years) + ".parquet")


def _check_district_table(table: pd.DataFrame) -> None:
    """Fail in seconds, before the slow citation join, if the district pipeline is out of date."""
    missing = [c for c in DISTRICT_COLUMNS if c not in table.columns]
    if missing:
        raise ValueError(
            f"build_district_table did not return {missing}. lib/district_pipeline.py is probably out "
            "of date: copy the latest version, then run get_census_data(force_refresh=True) so the "
            "census layer has the housing columns it needs."
        )


def _validate_panel(panel: pd.DataFrame, years: tuple[int, ...]) -> pd.DataFrame:
    """Fail loudly if the panel is not one complete row per district-year."""
    missing = [c for c in REQUIRED_COLUMNS if c not in panel.columns]
    if missing:
        raise ValueError(f"panel is missing required columns: {missing}")
    expected = N_DISTRICTS * len(years)
    if len(panel) != expected:
        raise ValueError(f"panel has {len(panel)} rows; expected {expected} ({N_DISTRICTS} districts x {len(years)} years)")
    if panel.duplicated(["District", "year"]).any():
        raise ValueError("panel has duplicate (District, year) rows")
    if set(panel["year"]) != set(years):
        raise ValueError(f"panel years {sorted(set(panel['year']))} do not match requested {sorted(years)}")
    nulls = panel[REQUIRED_COLUMNS].isna().sum()
    if nulls.any():
        raise ValueError(f"nulls in required columns: {nulls[nulls > 0].to_dict()}")
    return panel


def load_cached_panel(path: Path, years: tuple[int, ...] = DEFAULT_YEARS) -> pd.DataFrame:
    """Read the panel cache with pandas (it has no geometry) and validate it."""
    return _validate_panel(pd.read_parquet(path), years)


def _get_shapefiles() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    for p in (DISTRICT_SHP, ZIP_SHP):
        if not p.exists():
            raise FileNotFoundError(f"shapefile not found: {p}")
    return gpd.read_file(DISTRICT_SHP), gpd.read_file(ZIP_SHP)


def get_panel_data(
    years: tuple[int, ...] = DEFAULT_YEARS,
    use_cache: bool = USE_CACHE,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Build (or load from cache) the district x year panel.

    Parameters
    ----------
    years : tuple of int
        Calendar years of citations to include. Demographics are a single ACS vintage and are
        the same in every year.
    use_cache : bool
        If True, read/write a parquet cache so the 8.8M-row citation join is not repeated.
    force_refresh : bool
        If True, ignore the panel cache and rebuild it.

    Returns
    -------
    pd.DataFrame with one row per district and year (15 x len(years) rows).
    """
    years = tuple(years)
    cache_path = _cache_path(years)

    if use_cache and not force_refresh and cache_path.exists():
        try:
            return load_cached_panel(cache_path, years)
        except (FileNotFoundError, OSError, ValueError) as e:
            print(f"Warning: could not use cache ({type(e).__name__}: {e}). Rebuilding.")

    census_gdf = get_census_data()
    rent_df = get_zillow_data()
    district_gdf, zip_gdf = _get_shapefiles()

    res = build_district_table(census_gdf, district_gdf, zip_gdf, rent_df)
    _check_district_table(res.table)

    cit = citations_by_district(get_citation_data(), district_gdf)
    rates = annual_rates(cit.table, res.table, years=years)

    panel = (
        rates.drop(columns=["total_household", "total_population"])
             .merge(res.table, on="District", validate="many_to_one")
    )
    panel = _validate_panel(panel, years)

    if use_cache:
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            panel.to_parquet(cache_path)
        except Exception as e:
            print(f"Warning: could not write cache ({e}). Continuing without caching.")

    return panel