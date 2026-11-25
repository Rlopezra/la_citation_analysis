"""
district_pipeline.py

Builds one row per LA City Council district from two sources:

  * ACS block-group + tract census data -> area-share allocation, household weights
  * ZIP-level Zillow rent               -> household-weighted via block groups

Rules this module enforces (each one was a real bug at some point):

  1. Block-group areas and household totals come from the ORIGINAL county-wide census layer,
     never from the district overlay output. Otherwise every share sums to 100% and
     block groups that extend beyond the city are counted in full.
  2. All areas are measured in an equal-area CRS (EPSG:3310).
  3. Tract-level counts are repeated on every block group in the tract, so they are split by
     household share before being allocated to districts.
  4. Ratios are computed from summed numerators and denominators, never by averaging ratios.
  5. Rent is a household-weighted mean over ZIPs that HAVE rent data, and every district
     reports its coverage (the share of its households in ZIPs with data).

Usage:
    res = build_district_table(census_gdf, district_gdf, zip_gdf, rent_df)
    res.table        # one row per district: demographics + rent + coverage
    res.zip_weights  # households per (District, zip), for auditing
"""

from __future__ import annotations

import re
from typing import NamedTuple

import numpy as np
import pandas as pd
import geopandas as gpd

# ==========================================
# CONFIG
# ==========================================
AREA_CRS = 3310                    # California Albers (equal-area): .area is in square meters
CV_THRESHOLD = 0.3                 # block-group estimate counts as "fallback" above this CV
MIN_COUNTY_BLOCK_GROUPS = 5000     # the full LA County layer has ~6,600; fewer means it was clipped
ZIP_COL = "ZIPCODE"                # ZIP column in la_zip_shp
DISTRICT_COLS = ["District", "NAME", "District_N"]

# Sourced at block-group level (extensive)
BG_EXTENSIVE = [
    "total_population", "total_household",
    "pop_white_alone", "pop_black_alone", "pop_aian_alone", "pop_asian_alone",
    "pop_nhpi_alone", "pop_other_race_alone", "pop_two_or_more_races",
]

# Housing counts, sourced at block-group level (extensive)
HOUSING_EXTENSIVE = [
    "occupied_units", "owner_occupied", "renter_occupied",
    "units_total", "units_1_detached", "units_1_attached", "units_2", "units_3_4",
    "units_5_9", "units_10_19", "units_20_49", "units_50_plus", "units_mobile_home", "units_other",
    "rb_total", "rb_lt_10", "rb_10_14", "rb_15_19", "rb_20_24", "rb_25_29",
    "rb_30_34", "rb_35_39", "rb_40_49", "rb_50_plus", "rb_not_computed",
]
BG_EXTENSIVE = BG_EXTENSIVE + HOUSING_EXTENSIVE

# Sourced at TRACT level (extensive, but repeated on every block group in the tract)
TRACT_EXTENSIVE = [
    "commute_total_workers", "commute_drove_alone", "commute_carpool_2p",
    "commute_carpool_3p", "commute_carpool_4p_plus", "commute_transit_bus",
    "commute_transit_subway", "commute_transit_commuter_rail",
    "commute_transit_light_rail", "commute_transit_ferry", "commute_bicycle",
    "commute_walked", "commute_taxi_motorcycle_other", "commute_worked_from_home",
    "work_in_county_of_residence", "work_outside_county_of_residence",
    "travel_time_total_workers", "travel_time_under_5_min", "travel_time_5_to_9_min",
    "travel_time_10_to_14_min", "travel_time_15_to_19_min", "travel_time_20_to_24_min",
    "travel_time_25_to_29_min", "travel_time_30_to_34_min", "travel_time_35_to_39_min",
    "travel_time_40_to_44_min", "travel_time_45_to_59_min", "travel_time_60_to_89_min",
    "travel_time_90_plus_min", "agg_vehicles_total", "agg_vehicles_owner",
    "agg_vehicles_renter", "workers_per_hh_total", "workers_per_hh_0",
    "workers_per_hh_1", "workers_per_hh_2", "workers_per_hh_3_plus",
]

# intensive variable -> weight column. Rent is weighted by RENTER households: owner-heavy block
# groups have no renters paying rent, so total households would be the wrong weight.
INTENSIVE = {
    "median_income": "hh_w",
    "avg_household_size": "hh_w",
    "median_gross_rent": "renter_w",
}


class DistrictResults(NamedTuple):
    table: pd.DataFrame
    zip_weights: pd.DataFrame


# ==========================================
# KEYS AND LAYER PREPARATION
# ==========================================
def _bg_key(df: pd.DataFrame) -> pd.Series:
    return (
        df["STATE"].astype(str).str.zfill(2)
        + df["COUNTY"].astype(str).str.zfill(3)
        + df["TRACT"].astype(str).str.zfill(6)
        + df["BLOCK_GROUP"].astype(str)
    )


def _tract_key(df: pd.DataFrame) -> pd.Series:
    return (
        df["STATE"].astype(str).str.zfill(2)
        + df["COUNTY"].astype(str).str.zfill(3)
        + df["TRACT"].astype(str).str.zfill(6)
    )


def prepare_census(census: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Reproject (returns a new object, the cached input is not mutated) and add keys BEFORE any overlay."""
    census = census.to_crs(AREA_CRS)
    census["bg_key"] = _bg_key(census)
    census["tract_key"] = _tract_key(census)
    return census


def prepare_districts(districts: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return districts.to_crs(AREA_CRS)[DISTRICT_COLS + ["geometry"]].copy()


def prepare_zips(zips: gpd.GeoDataFrame, zip_col: str = ZIP_COL) -> gpd.GeoDataFrame:
    """One valid polygon per ZIP, string key 'zip'. The county file has multi-part ZIPs stored
    as separate rows and at least one invalid geometry."""
    z = zips.copy()
    z["geometry"] = z.geometry.make_valid()
    z = z.to_crs(AREA_CRS)
    z["zip"] = z[zip_col].astype(str).str.zfill(5)
    return z[["zip", "geometry"]].dissolve(by="zip").reset_index()


# ==========================================
# BLOCK-GROUP REFERENCE (original, unclipped)
# ==========================================
def block_group_reference(census: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Full block-group area, household count, and each block group's household share of its tract.
    Must be built from the original county-wide census layer.
    """
    assert "District" not in census.columns, (
        "reference must be the original census layer, not the district overlay output"
    )
    n_bg = census["bg_key"].nunique()
    assert n_bg > MIN_COUNTY_BLOCK_GROUPS, (
        f"reference has only {n_bg} block groups; the full county layer has ~6,600 (looks clipped to the city)"
    )

    # A block group can span several rows if its geometry is multi-part: its area is the sum of the parts,
    # and attributes must be de-duplicated before any tract total is computed.
    if census["bg_key"].duplicated().any():
        assert census.groupby("bg_key")["total_household"].nunique().max() <= 1, (
            "duplicate bg_key rows carry different data, not just split geometry"
        )

    bg_area = census.geometry.area.groupby(census["bg_key"]).sum().rename("bg_area")
    bg = (
        census.drop_duplicates("bg_key")[["bg_key", "tract_key", "total_household"]]
        .rename(columns={"total_household": "bg_households"})
        .merge(bg_area, on="bg_key", validate="one_to_one")
    )

    tract_hh = bg.groupby("tract_key")["bg_households"].transform("sum")
    tract_n = bg.groupby("tract_key")["bg_key"].transform("count")
    # household-proportional share of the tract; equal split if the tract has no households
    bg["hh_share_of_tract"] = np.where(tract_hh > 0, bg["bg_households"] / tract_hh, 1.0 / tract_n)
    return bg[["bg_key", "bg_area", "bg_households", "hh_share_of_tract"]]


# ==========================================
# OVERLAY (done once, reused by census and rent)
# ==========================================
def district_block_group_fragments(districts: gpd.GeoDataFrame, census: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    overlap = (set(districts.columns) & set(census.columns)) - {"geometry"}
    assert not overlap, f"column name collision between districts and census: {sorted(overlap)}"
    frags = gpd.overlay(districts, census, how="intersection")
    frags["frag_area"] = frags.geometry.area
    return frags


def _check_shares(share_by_bg: pd.Series, require_partial: bool) -> None:
    assert (share_by_bg <= 1.0001).all(), "a block group's pieces exceed its original area"
    if require_partial:
        # A real overlay against the city boundary always leaves block groups partly outside.
        assert (share_by_bg < 0.98).any(), (
            "every block group's shares sum to 100%: bg_area was derived from clipped fragments, "
            "not the original block-group geometry"
        )


# ==========================================
# CENSUS: ALLOCATE AND AGGREGATE
# ==========================================
def allocate(frags: gpd.GeoDataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    g = frags.merge(
        ref[["bg_key", "bg_area", "hh_share_of_tract"]], on="bg_key", how="left", validate="many_to_one"
    )
    assert g["bg_area"].notna().all(), "overlay block groups missing from the census reference"
    g["share"] = g["frag_area"] / g["bg_area"]          # denominator = ORIGINAL block-group area
    _check_shares(g.groupby("bg_key")["share"].sum(), require_partial=True)

    out = pd.DataFrame(g.drop(columns="geometry")).copy()

    for c in BG_EXTENSIVE:
        out[f"{c}_a"] = out[c] * out["share"]
    for c in TRACT_EXTENSIVE:
        out[f"{c}_a"] = out[c] * out["hh_share_of_tract"] * out["share"]

    out["hh_w"] = out["total_household_a"]                      # weight for income, household size
    out["renter_w"] = out["renter_occupied_a"]                  # weight for gross rent

    # households whose block-group estimate was replaced by the tract fallback (for disclosure)
    out["income_fallback"] = out["median_household_income_blocks"].isna() | (out["cv_income"] > CV_THRESHOLD)
    out["hh_income_fallback"] = out["hh_w"].where(out["income_fallback"], 0.0)

    # renter households whose rent estimate was replaced by the tract fallback (for disclosure)
    out["rent_fallback"] = out["median_gross_rent_blocks"].isna() | (out["cv_rent"] > CV_THRESHOLD)
    out["renter_hh_rent_fallback"] = out["renter_w"].where(out["rent_fallback"], 0.0)

    # households with vehicle data (keeps the vei numerator and denominator consistent)
    out["hh_for_vei"] = out["hh_w"].where(out["agg_vehicles_total"].notna(), 0.0)
    return out


def aggregate_demographics(out: pd.DataFrame) -> pd.DataFrame:
    sum_cols = [f"{c}_a" for c in BG_EXTENSIVE + TRACT_EXTENSIVE] + ["hh_income_fallback", "renter_hh_rent_fallback", "hh_for_vei"]
    d = out.groupby("District")[sum_cols].sum(min_count=1)
    d.columns = [c[:-2] if c.endswith("_a") else c for c in d.columns]

    # household-weighted means, excluding rows where the value is missing
    for c, wcol in INTENSIVE.items():
        valid = out[c].notna()
        w = out[wcol].where(valid, 0.0)
        num = (out[c].fillna(0.0) * w).groupby(out["District"]).sum()
        den = w.groupby(out["District"]).sum()
        d[c] = num / den

    d["income_fallback_hh_share"] = d["hh_income_fallback"] / d["total_household"]
    d["rent_fallback_renter_share"] = d["renter_hh_rent_fallback"] / d["renter_occupied"]

    # ratio measures: ratio of sums, never a mean of block-group ratios
    d["vei"] = d["agg_vehicles_total"] / d["hh_for_vei"]
    d["cdr"] = (d["commute_drove_alone"] + d["commute_carpool_2p"]
                + d["commute_carpool_3p"] + d["commute_carpool_4p_plus"]) / d["commute_total_workers"]
    d["hwi"] = (d["workers_per_hh_2"] + d["workers_per_hh_3_plus"]) / d["workers_per_hh_total"]
    d["ece"] = (d["travel_time_45_to_59_min"] + d["travel_time_60_to_89_min"]
                + d["travel_time_90_plus_min"]) / d["travel_time_total_workers"]
    # replaces the old `oci` (outside-county workers / households, which isn't a share)
    d["outside_county_worker_share"] = d["work_outside_county_of_residence"] / (
        d["work_in_county_of_residence"] + d["work_outside_county_of_residence"])

    # housing ratios, from summed counts
    d["renter_share"] = d["renter_occupied"] / d["occupied_units"]
    multi = ["units_2", "units_3_4", "units_5_9", "units_10_19", "units_20_49", "units_50_plus"]
    d["multiunit_share"] = d[multi].sum(axis=1) / d["units_total"]
    d["large_multiunit_share"] = (d["units_20_49"] + d["units_50_plus"]) / d["units_total"]
    rb_valid = d["rb_total"] - d["rb_not_computed"]              # renters with a computable burden
    d["rent_burden_share"] = (d["rb_30_34"] + d["rb_35_39"] + d["rb_40_49"] + d["rb_50_plus"]) / rb_valid
    d["severe_rent_burden_share"] = d["rb_50_plus"] / rb_valid
    # existing commute columns, for the public-infrastructure hypothesis
    d["transit_commute_share"] = (
        d["commute_transit_bus"] + d["commute_transit_subway"] + d["commute_transit_commuter_rail"]
        + d["commute_transit_light_rail"] + d["commute_transit_ferry"]
    ) / d["commute_total_workers"]

    for c in ["white", "black", "aian", "asian", "nhpi", "other_race"]:
        d[f"share_{c}"] = d[f"pop_{c}_alone"] / d["total_population"]
    d["share_two_or_more"] = d["pop_two_or_more_races"] / d["total_population"]

    d = d.drop(columns=["hh_for_vei", "hh_income_fallback", "renter_hh_rent_fallback"])
    labels = out[DISTRICT_COLS].drop_duplicates("District").set_index("District")
    return labels.join(d).reset_index()


# ==========================================
# RENT: ZIP WEIGHTS AND AGGREGATION
# ==========================================
def zip_household_weights(frags: gpd.GeoDataFrame, zips: gpd.GeoDataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Households per (District, zip): each district x block-group piece is split by ZIP, and each
    sub-piece gets its share of the ORIGINAL block group's households."""
    sub = gpd.overlay(frags[["District", "bg_key", "geometry"]], zips, how="intersection")
    sub["sub_area"] = sub.geometry.area
    sub = sub.merge(ref[["bg_key", "bg_area", "bg_households"]], on="bg_key", how="left", validate="many_to_one")
    sub["share"] = sub["sub_area"] / sub["bg_area"]
    _check_shares(sub.groupby("bg_key")["share"].sum(), require_partial=False)
    sub["hh_w"] = sub["bg_households"] * sub["share"]
    return sub.groupby(["District", "zip"], as_index=False)["hh_w"].sum()


def aggregate_rent(weights: pd.DataFrame, rent_df: pd.DataFrame) -> pd.DataFrame:
    """Household-weighted mean rent per district and year, over ZIPs that have rent data that year.
    `coverage_YYYY` is the share of the district's households living in ZIPs with data."""
    rent = rent_df.copy()
    rent["zip"] = rent["ZipCode"].astype(str).str.zfill(5)
    assert rent["zip"].is_unique, "rent table must have one row per ZIP"
    rent_cols = sorted(c for c in rent.columns if re.fullmatch(r"Average_Rent_\d{4}", c))

    w = weights.merge(rent[["zip"] + rent_cols], on="zip", how="left", validate="many_to_one")
    total_hh = w.groupby("District")["hh_w"].sum()

    out = pd.DataFrame(index=total_hh.index)
    for col in rent_cols:
        yr = col[-4:]
        has = w[col].notna()
        num = (w[col] * w["hh_w"]).where(has, 0.0).groupby(w["District"]).sum()
        den = w["hh_w"].where(has, 0.0).groupby(w["District"]).sum()
        out[f"rent_{yr}"] = num / den
        out[f"coverage_{yr}"] = den / total_hh
    out["n_zips"] = w.groupby("District")["zip"].nunique()
    return out.reset_index()


# ==========================================
# ORCHESTRATION
# ==========================================
def build_district_table(
    census: gpd.GeoDataFrame,
    districts: gpd.GeoDataFrame,
    zips: gpd.GeoDataFrame,
    rent_df: pd.DataFrame,
    zip_col: str = ZIP_COL,
) -> DistrictResults:
    census = prepare_census(census)
    districts = prepare_districts(districts)
    zips = prepare_zips(zips, zip_col)

    ref = block_group_reference(census)                       # from the ORIGINAL census layer
    frags = district_block_group_fragments(districts, census)  # overlay once, reuse for census and rent

    demographics = aggregate_demographics(allocate(frags, ref))
    weights = zip_household_weights(frags, zips, ref)
    rent = aggregate_rent(weights, rent_df)

    table = demographics.merge(rent, on="District", how="left", validate="one_to_one")
    return DistrictResults(table, weights)


if __name__ == "__main__":
    from lib.census_data import get_census_data
    from lib.zillow_data import get_zillow_data

    census_gdf = get_census_data()
    district_gdf = gpd.read_file(
        "data/shapefiles/LA_City_Council_Districts/LA_City_Council_Districts_(Adopted_2021).shp")
    zip_gdf = gpd.read_file("data/shapefiles/LA_County_ZIP_Codes/LA_County_ZIP_Codes_(Hosted).shp")
    rent_df = get_zillow_data()      # needs ZipCode + Average_Rent_YYYY columns, one row per ZIP

    res = build_district_table(census_gdf, district_gdf, zip_gdf, rent_df)
    print(f"Allocated population: {res.table['total_population'].sum():,.0f}")
    print(f"Allocated households: {res.table['total_household'].sum():,.0f}")
    print(res.table[["District", "total_population", "median_income", "rent_2025", "coverage_2025"]]
          .round(2).to_string(index=False))