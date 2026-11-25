"""
citation_join.py

Assigns each parking citation to a LA City Council district (point-in-polygon) and aggregates to
District x month x violation category. Every row is accounted for: each citation is either assigned
or excluded for exactly one stated reason, so the exclusions can be disclosed instead of silently
dropped.

Usage:
    from lib.citation_join import citations_by_district, annual_rates

    cit = citations_by_district(citation_df, district_gdf)       # cit.table, cit.qa
    rates = annual_rates(cit.table, district_table, years=(2022, 2023, 2024, 2025))
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd
import geopandas as gpd
import polars as pl

BBOX_MARGIN_DEG = 0.05      # pre-filter margin around the district layer (cheap, avoids building junk points)

EXCLUSION_ORDER = ["bad_date", "null_coords", "zero_coords", "outside_bbox", "outside_city"]


class CitationResults(NamedTuple):
    table: pd.DataFrame      # District, month, category, tickets, revenue
    qa: dict                 # row accounting


def _reason_codes(dt, lat, lon, max_date, bbox):
    """One exclusion reason per row (first match wins), or '' if the row is usable so far."""
    minx, miny, maxx, maxy = bbox
    reason = np.full(len(dt), "", dtype=object)
    conds = [
        ("bad_date", dt.isna().to_numpy() | (dt > max_date).to_numpy()),
        ("null_coords", lat.isna().to_numpy() | lon.isna().to_numpy()),
        ("zero_coords", ((lat.abs() < 1e-6) & (lon.abs() < 1e-6)).to_numpy()),
        ("outside_bbox", ~((lon.between(minx, maxx)) & (lat.between(miny, maxy))).to_numpy()),
    ]
    for name, mask in conds:
        reason[(reason == "") & mask] = name
    return reason


def citations_by_district(
    citations: pl.DataFrame,
    districts: gpd.GeoDataFrame,
    *,
    lat_col: str = "loc_lat",
    lon_col: str = "loc_long",
    date_col: str = "issue_date",
    fine_col: str = "fine_amount",
    category_col: str = "violation_description_lumped",
    max_date: str | pd.Timestamp | None = None,
    chunk_size: int = 1_000_000,
) -> CitationResults:
    """
    max_date: citations dated after this are treated as data-entry errors (default: today).
    Point-in-polygon is done in EPSG:4326 (only the 15 district polygons are reprojected, not
    millions of points). No equal-area CRS is needed for containment tests.
    """
    max_date = pd.Timestamp.today().normalize() if max_date is None else pd.Timestamp(max_date)
    d = districts.to_crs(4326)[["District", "geometry"]].copy()
    minx, miny, maxx, maxy = d.total_bounds
    bbox = (minx - BBOX_MARGIN_DEG, miny - BBOX_MARGIN_DEG, maxx + BBOX_MARGIN_DEG, maxy + BBOX_MARGIN_DEG)

    qa = {k: 0 for k in ["total_rows"] + EXCLUSION_ORDER + ["assigned"]}
    pieces = []

    for start in range(0, len(citations), chunk_size):
        ch = citations.slice(start, chunk_size).to_pandas()
        
        dt = pd.to_datetime(ch[date_col], errors="coerce")
        lat, lon = ch[lat_col], ch[lon_col]
        reason = _reason_codes(dt, lat, lon, max_date, bbox)
        qa["total_rows"] += len(ch)

        ok = reason == ""
        for name in ["bad_date", "null_coords", "zero_coords", "outside_bbox"]:
            qa[name] += int((reason == name).sum())

        pts = gpd.GeoDataFrame(
            {"row": np.flatnonzero(ok)},
            geometry=gpd.points_from_xy(lon.to_numpy()[ok], lat.to_numpy()[ok]),
            crs=4326,
        )
        joined = gpd.sjoin(pts, d, how="left", predicate="within")
        joined = joined[~joined.index.duplicated(keep="first")]      # a point exactly on a shared edge

        in_city = joined["District"].notna().to_numpy()
        qa["outside_city"] += int((~in_city).sum())
        qa["assigned"] += int(in_city.sum())

        rows = joined.loc[in_city, "row"].to_numpy()
        sel = ch.iloc[rows]
        pieces.append(
            pd.DataFrame({
                "District": joined.loc[in_city, "District"].astype(int).to_numpy(),
                "month": dt.iloc[rows].dt.to_period("M").to_numpy(),
                "category": sel[category_col].to_numpy(),
                "fine": sel[fine_col].to_numpy(),
            })
            .groupby(["District", "month", "category"], observed=True)
            .agg(tickets=("fine", "size"), revenue=("fine", "sum"))
            .reset_index()
        )

    table = (
        pd.concat(pieces, ignore_index=True)
        .groupby(["District", "month", "category"], as_index=False)[["tickets", "revenue"]].sum()
    )
    assert qa["assigned"] + sum(qa[k] for k in EXCLUSION_ORDER) == qa["total_rows"], "row accounting does not add up"
    assert table["tickets"].sum() == qa["assigned"], "aggregated tickets != assigned rows"
    return CitationResults(table, qa)


def annual_rates(
    cit_table: pd.DataFrame,
    district_table: pd.DataFrame,
    years=(2022, 2023, 2024, 2025),
    categories: list[str] | None = None,
) -> pd.DataFrame:
    """
    District x year tickets, revenue, and rates. Use complete years only (the data ends mid-2026).
    Denominators are the single ACS vintage in district_table, used for every year.
    categories: restrict to certain violation categories (None = all).
    """
    t = cit_table.copy()
    t["year"] = t["month"].dt.year
    t = t[t["year"].isin(years)]
    if categories is not None:
        t = t[t["category"].isin(categories)]
    a = t.groupby(["District", "year"], as_index=False)[["tickets", "revenue"]].sum()

    a = a.merge(district_table[["District", "total_household", "total_population"]], on="District", validate="many_to_one")
    a["tickets_per_household"] = a["tickets"] / a["total_household"]
    a["tickets_per_capita"] = a["tickets"] / a["total_population"]
    a["revenue_per_household"] = a["revenue"] / a["total_household"]
    return a