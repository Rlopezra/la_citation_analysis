import json

import streamlit as st
import altair as alt
import pandas as pd
from scipy.stats import spearmanr
from pathlib import Path

st.set_page_config(
    page_title="Parking Citations by LA Council District",
    page_icon=":art:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- data ---
district_summary = pd.read_csv("data/results/district_summary.csv")
headline_numbers = pd.read_csv("data/results/headline_numbers.csv")
nb_poisson_numbers = pd.read_csv("data/results/nb_poisson_labeled.csv")
nb_one_out_in = pd.read_csv("data/results/nb_leave_one_district_out.csv")


@st.cache_data
def load_districts():
    path = Path(__file__).parent / "data" / "results" / "council_districts.geojson"
    with open(path) as f:
        return json.load(f)["features"]


metric_labels = {
    "Tickets / household": "avg_tickets_per_household",
    "Ticket Change 2022-25": "pct_change_tickets",
    "Renter share": "renter_share",
    "Multi-unit share": "multiunit_share",
    "Transit commuters": "transit_commute_share",
    "Rent burdened": "rent_burden_share",
    "Median income": "median_income",
    "Household size": "avg_household_size",
}

FAMILIES = {
    "Primary": ("data/results/results_primary.csv", "4 pre-declared tests, Bonferroni-corrected across 4."),
    "Secondary": ("data/results/results_secondary.csv",
                  "6 diagnostic tests, reported uncorrected. The two per-vehicle tests were added after the first results."),
    "Housing & transit": ("data/results/results_tertiary.csv",
                          "10 tests defined after earlier results, Bonferroni-corrected across 10 tests."),
}
TEST_LABELS = {
    "avg_tickets_per_household": "Tickets / household", "avg_tickets_per_capita": "Tickets / person",
    "avg_tickets_per_vehicle": "Tickets / vehicle", "pct_change_tickets": "Change in tickets",
    "median_income": "Median income", "avg_household_size": "Household size",
    "rent_2025_masked": "Rent (2025)", "renter_share": "Renter share",
    "multiunit_share": "Multi-unit share", "rent_burden_share": "Rent burden",
    "severe_rent_burden_share": "Severe rent burden", "transit_commute_share": "Transit commuters",
}
# regression terms, with the units the model uses
TERM_LABELS = {
    "renter_10pt": "Renter share (+10 pts)",
    "income_10k": "Income (+$10k)",
    "avg_household_size": "Household size (+1 person)",
}
nb_poisson_numbers["term_label"] = nb_poisson_numbers["term"].map(TERM_LABELS).fillna(nb_poisson_numbers["term"])

# --- numbers reused in text (computed from the data so the text can't drift from the results) ---
rho, _ = spearmanr(district_summary["renter_share"], district_summary["avg_tickets_per_household"])

nb_result = nb_poisson_numbers.query(
    "family == 'Negative Binomial' and specification == 'renter + income' and term == 'renter_10pt'"
)
full_data_estimate = nb_result["rate_ratio"].iloc[0]
pct_more = (full_data_estimate - 1) * 100
loo_min = nb_one_out_in["renter_rate_ratio"].min()
loo_max = nb_one_out_in["renter_rate_ratio"].max()
nb_disp = nb_poisson_numbers.loc[nb_poisson_numbers["family"] == "Negative Binomial", "pearson_dispersion"].mean()
pois_disp = nb_poisson_numbers.loc[nb_poisson_numbers["family"].str.startswith("Poisson"), "pearson_dispersion"].mean()

# --- sidebar (controls the scatter on the Overview tab) ---
st.sidebar.header("Scatterplot Settings")
selected_x_label = st.sidebar.selectbox("Choose X-axis", list(metric_labels), index=2)
selected_y_label = st.sidebar.selectbox("Choose Y-axis", list(metric_labels), index=0)
x_axis = metric_labels[selected_x_label]
y_axis = metric_labels[selected_y_label]
st.sidebar.caption("These settings control the scatter plot on the Overview tab.")

st.title("Parking Citations by LA Council District")
st.markdown(
    f"**Takeaway:** districts with more renters receive more parking tickets per household "
    f"(Spearman ρ = {rho:.2f} across the 15 council districts). Income, household size and rent burden do not "
    f"show the same pattern, and nothing we tested explains which districts saw tickets grow."
)

tab_overview, tab_results, tab_regression, tab_methods, tab_writeup = st.tabs(
    ["Overview", "Results tables", "Regression", "Methods and limitations", "Write-up"]
)

# ===================== Overview tab =====================
with tab_overview:
    ticket_val = headline_numbers.loc[headline_numbers["metric"] == "tickets_2025", "value"].iloc[0]
    high = district_summary.loc[district_summary["avg_tickets_per_household"].idxmax()]
    low = district_summary.loc[district_summary["avg_tickets_per_household"].idxmin()]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Citywide tickets (2025)", f"{ticket_val:,.0f}")
    col2.metric("Highest-rate district", f"District {int(high['District'])}",
                f"{high['avg_tickets_per_household']:.2f} tickets/household", delta_color="off")
    col3.metric("Lowest-rate district", f"District {int(low['District'])}",
                f"{low['avg_tickets_per_household']:.2f} tickets/household", delta_color="off")
    col4.metric("Renter share vs. tickets per household", f"ρ = {rho:.2f}")

    st.subheader(f"Scatter Plot: {selected_x_label} vs {selected_y_label}")
    scatter_base = alt.Chart(district_summary).encode(
        x=alt.X(f"{x_axis}:Q", title=selected_x_label, scale=alt.Scale(zero=False)),
        y=alt.Y(f"{y_axis}:Q", title=selected_y_label, scale=alt.Scale(zero=False)),
        tooltip=["District:N", f"{x_axis}:Q", f"{y_axis}:Q"],
    )
    dots = scatter_base.mark_circle(size=80, opacity=0.9, color="#2f6db5")
    dot_labels = scatter_base.mark_text(dx=9, dy=-7, fontSize=11, color="#555555").encode(text="District:N")
    st.altair_chart((dots + dot_labels).properties(height=450).interactive(), use_container_width=True)
    st.caption("Each dot is a council district, labeled with its district number.")

    # ---- District map ----
    st.subheader("District Map")
    map_label = st.selectbox("Color the map by", list(metric_labels), index=0, key="map_metric")
    map_col = metric_labels[map_label]

    # number format for the legend and tooltip, by column
    fmt = {"avg_tickets_per_household": ".2f", "pct_change_tickets": "+.1%", "renter_share": ".1%",
           "multiunit_share": ".1%", "transit_commute_share": ".1%", "rent_burden_share": ".1%",
           "median_income": "$,.0f", "avg_household_size": ".2f"}

    # change in tickets can be negative, so center that scale on zero; everything else is low-to-high
    if map_col == "pct_change_tickets":
        color_scale = alt.Scale(scheme="redblue", reverse=True, domainMid=0)
    else:
        color_scale = alt.Scale(scheme="blues")

    # attach each district's numbers to its shape as plain GeoJSON features (table columns live under "properties.")
    stats = district_summary.set_index("District")[list(metric_labels.values())].to_dict("index")
    districts = load_districts()
    shape_features = [
        {"type": "Feature", "geometry": f["geometry"],
         "properties": {"District": f["properties"]["District"], **stats[f["properties"]["District"]]}}
        for f in districts
    ]
    label_points = [
        {"District": f["properties"]["District"], "lon": f["properties"]["lon"], "lat": f["properties"]["lat"]}
        for f in districts
    ]

    shapes = alt.Chart(alt.Data(values=shape_features)).mark_geoshape(stroke="white", strokeWidth=1).encode(
        color=alt.Color(f"properties.{map_col}:Q", scale=color_scale,
                        legend=alt.Legend(title=map_label, format=fmt[map_col])),
        tooltip=[alt.Tooltip("properties.District:N", title="District")] + [
            alt.Tooltip(f"properties.{col}:Q", title=label, format=fmt[col]) for label, col in metric_labels.items()
        ],
    )
    labels = alt.Chart(alt.Data(values=label_points)).mark_text(
        fontSize=12, fontWeight="bold", color="black"
    ).encode(longitude="lon:Q", latitude="lat:Q", text="District:N")

    district_map = alt.layer(shapes, labels).project(type="mercator").properties(height=600)
    st.altair_chart(district_map, use_container_width=True)
    st.caption("Hover over a district for its key fields. District numbers are shown on the map.")

# ===================== Results tables tab =====================
with tab_results:
    tab_tests, tab_data = st.tabs(["Test results", "District data"])

    with tab_tests:
        family = st.radio("Test family", list(FAMILIES), horizontal=True)
        path, caption = FAMILIES[family]
        st.caption(caption)

        t = pd.read_csv(path)
        t["outcome"] = t["outcome"].map(TEST_LABELS)
        t["predictor"] = t["predictor"].map(TEST_LABELS)
        t["sign_stable_loo"] = t["sign_stable_loo"].map({True: "✓", False: "✗"})
        if family != "Secondary":
            t["significant"] = (t["p_bonferroni"] < 0.05).map({True: "✓", False: ""})
        t = t.rename(columns={"outcome": "Outcome", "predictor": "Predictor", "p_perm": "p",
                              "p_bonferroni": "Bonferroni p", "sign_stable_loo": "Stable (LOO)",
                              "significant": "Significant"})
        st.dataframe(t, hide_index=True, use_container_width=True)
        st.download_button("Download this table (CSV)", t.to_csv(index=False), f"{family}.csv", "text/csv")
        st.caption("rho = Spearman rank correlation. p = permutation test (10,000 shuffles). "
                   "Stable = sign of rho unchanged when any one district is dropped. n = 15 districts.")

    with tab_data:
        cols = ["District", "avg_tickets_per_household", "pct_change_tickets", "renter_share",
                "multiunit_share", "transit_commute_share", "rent_burden_share",
                "median_income", "avg_household_size"]
        st.dataframe(
            district_summary[cols], hide_index=True,
            column_config={
                "avg_tickets_per_household": st.column_config.NumberColumn("Tickets / household", format="%.2f"),
                "pct_change_tickets": st.column_config.NumberColumn("Ticket Change 2022-25", format="percent"),
                "renter_share": st.column_config.NumberColumn("Renter share", format="percent"),
                "multiunit_share": st.column_config.NumberColumn("Multi-unit share", format="percent"),
                "transit_commute_share": st.column_config.NumberColumn("Transit commuters", format="percent"),
                "rent_burden_share": st.column_config.NumberColumn("Rent burdened", format="percent"),
                "median_income": st.column_config.NumberColumn("Median income", format="dollar"),
                "avg_household_size": st.column_config.NumberColumn("Household size", format="%.2f"),
            },
        )

# ===================== Regression tab =====================
with tab_regression:
    st.subheader("Introduction")
    st.markdown("Negative binomial model of ticket counts with households as the exposure, so results read as changes in tickets per household. Standard errors clustered by district. n = 15 districts, 60 district-years.")

    with st.container(border=True):
        st.subheader("Section 1: Headline Result")

        reg_col1, reg_col2, reg_col3 = st.columns([2, 2, 2])
        reg_col1.metric("Renter-share rate ratio", f"{full_data_estimate:.2f}", "+10 pts (renter + income model)")
        reg_col2.metric("95% Confidence Interval", f"{nb_result['rr_low'].iloc[0]:,.2f} to {nb_result['rr_high'].iloc[0]:,.2f}")
        reg_col3.metric("Leave-one-out range", f"{loo_min:.2f} to {loo_max:.2f}", delta_color="off")

        st.markdown(f"Districts with 10 percentage points more renters see about {pct_more:.0f}% more tickets per household, holding income constant.")

    with st.container(border=True):
        st.subheader("Section 2: Model Comparison")
        model = st.radio("Specification picker:", list(nb_poisson_numbers['specification'].unique()), horizontal=True)
        model_data = nb_poisson_numbers[(nb_poisson_numbers['specification'] == model) & (nb_poisson_numbers['family'] == 'Negative Binomial')]

        chart_col, table_col = st.columns([2, 1])

        with chart_col:
            # 1. Base chart encoding the term on the y-axis
            base = alt.Chart(model_data).encode(y=alt.Y("term_label:N", sort=None, title=None))

            # 2. Confidence interval bars (horizontal rules using x and x2)
            errors = base.mark_rule(strokeWidth=2, color="#4c78a8").encode(
                x=alt.X("rr_low:Q", title="Rate Ratio (95% CI)",
                        scale=alt.Scale(domain=[0.6, 1.9], zero=False)),
                x2=alt.X2("rr_high:Q"),
            )

            # 3. Point estimates (dots)
            points = base.mark_point(filled=True, size=120, color="#1f77b4").encode(
                x="rate_ratio:Q",
                tooltip=[alt.Tooltip("term_label:N", title="Term"),
                         alt.Tooltip("rate_ratio:Q", title="Rate ratio", format=".2f"),
                         alt.Tooltip("rr_low:Q", title="CI low", format=".2f"),
                         alt.Tooltip("rr_high:Q", title="CI high", format=".2f")],
            )

            # 4. "No effect" vertical reference line at 1.0 (fixed value)
            reference_line = (alt.Chart(pd.DataFrame({"x": [1.0]})).mark_rule(
                strokeDash=[4, 4], color="gray", strokeWidth=1.5).encode(x="x:Q"))

            # 5. Layer them together (reference line + error bars + point estimates)
            forest_plot = (reference_line + errors + points).properties(
                width=600, height=350, title="Forest plot: rate ratio with CI bars",
            )
            st.altair_chart(forest_plot, use_container_width=True)

        with table_col:
            display_df = pd.DataFrame({
                "Term": model_data["term_label"],
                "Rate Ratio": model_data["rate_ratio"].round(2),
                # Combine lower and upper bounds into a single clean string column
                "Confidence Interval": (model_data["rr_low"].round(2).astype(str)
                                        + " to " + model_data["rr_high"].round(2).astype(str)),
            })
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    with st.container(border=True):
        st.subheader("Section 3: Why Negative Binomial, not Poisson")

        # Align Negative Binomial vs Poisson side-by-side
        pivoted = nb_poisson_numbers.pivot_table(
            index=['specification', 'term_label'],
            columns='family',
            values=['rate_ratio', 'pearson_dispersion']
        ).reset_index()

        # Flatten multi-index columns for easy viewing
        pivoted.columns = ['_'.join(col).strip('_') for col in pivoted.columns.values]

        formatted_table = pd.DataFrame({
            'Specification': pivoted['specification'] + " (" + pivoted['term_label'] + ")",
            'NB rate ratio': pivoted['rate_ratio_Negative Binomial'].round(2),
            'Poisson rate ratio': pivoted['rate_ratio_Poisson (clustered SE)'].round(2),
            'NB dispersion': pivoted['pearson_dispersion_Negative Binomial'].round(2),
            'Poisson dispersion': pivoted['pearson_dispersion_Poisson (clustered SE)'].round(2)
        })

        st.dataframe(formatted_table, hide_index=True, use_container_width=True)
        st.info(
            f"Poisson dispersion ≈ {pois_disp:,.0f} vs. ≈ {nb_disp:.1f} for NB. Poisson assumes "
            "variance equals the mean; these counts are far more variable than that. "
            "Negative binomial dispersion values close to 1.0 indicate a good fit, "
            "whereas Poisson shows severe overdispersion. Poisson is shown with district-clustered standard errors."
        )

    with st.container(border=True):
        st.subheader("Section 4: Robustness (leave-one-district-out)")

        # 1. Base chart encoding the dropped district on the x-axis
        base = alt.Chart(nb_one_out_in).encode(
            x=alt.X("dropped_district:N", title="District Dropped", sort=None)
        )

        # 2. Point estimates (dots)
        points = base.mark_point(filled=True, size=130, color="#1f77b4").encode(
            y=alt.Y("renter_rate_ratio:Q", title="Renter-Share Rate Ratio",
                    scale=alt.Scale(domain=[1.0, 1.7], zero=False)),
            tooltip=[alt.Tooltip("dropped_district:N", title="District dropped"),
                     alt.Tooltip("renter_rate_ratio:Q", title="Rate ratio", format=".2f")],
        )
        # 3. Reference line at 1.0 (null effect)
        ref_one = (
            alt.Chart(pd.DataFrame({"y": [1.0]}))
            .mark_rule(strokeDash=[4, 4], color="gray", strokeWidth=1.5)
            .encode(y="y:Q")
        )
        # 4. Reference line at the full-data estimate
        ref_full = (
            alt.Chart(pd.DataFrame({"y": [full_data_estimate]}))
            .mark_rule(color="firebrick", strokeWidth=1.5, strokeDash=[2, 2])
            .encode(y="y:Q")
        )

        # 5. Layer everything together
        loo_chart = (ref_one + ref_full + points).properties(
            width=600, height=350, title="Leave-One-Out Sensitivity Analysis by District"
        )
        st.altair_chart(loo_chart, use_container_width=True)
        st.caption(
            f"No single district drives the result: the rate ratio stays between "
            f"{loo_min:.2f} and {loo_max:.2f} when any one district is removed "
            f"(red dashed line = estimate using all 15 districts, {full_data_estimate:.2f})."
        )

    with st.container(border=True):
        st.subheader("Section 5: How to read this / caveats")
        st.markdown(
            f"""
- **Rate ratio.** A value of 1.0 means no change. The renter-share rate ratio of {full_data_estimate:.2f} means that a district with
  10 percentage points more renters has about {pct_more:.0f}% more tickets per household, holding income constant.
- **Only 15 clusters.** Standard errors are clustered by district, but with so few districts they tend to be too small, so the
  p-values and confidence intervals here are optimistic. The permutation test on the Results tab is the more conservative number.
- **Districts, not years.** Renter share and income do not change from year to year in this model, so it compares districts to each
  other. It cannot say what happens inside one district over time.
- **Association, not cause.** Renter-heavy districts may differ in ways we did not measure, such as street density, parking supply
  or how many visitors and commuters they attract.
            """
        )

# ===================== Methods and limitations tab =====================
with tab_methods:
    st.subheader("Data")
    st.markdown(
        """
- **Parking citations:** City of Los Angeles data portal, 2022 to 2025. Each citation is assigned to a council district with a
  spatial join on its location, using the council district boundaries.
- **Census:** ACS 5-year 2024 estimates (income, household size, housing and commuting) at the block group and tract level.
- **Rent:** Zillow ZIP-code rent for all Los Angeles County ZIPs. A ZIP's yearly rent is the average of its monthly values, used only
  when at least half of that year's months are available. ZIP rents are carried to districts with household weights, and a district's
  rent is left out when ZIPs with rent data cover under 70% of its households (District 7, at 68%).
        """
    )
    st.subheader("Method")
    st.markdown(
        """
- **Building district figures:** block groups are split across districts by area (area-weighted interpolation in an equal-area
  projection), with tract-level counts first split across block groups by household share. District totals add up to 3,855,903
  people and 1,438,336 households, which checks that nothing is counted twice.
- **Rates:** tickets per household (primary), per person and per vehicle (secondary checks).
- **Tests:** Spearman rank correlation with permutation p-values (10,000 shuffles), a Bonferroni correction within the primary
  and housing families, and a leave-one-district-out check on the direction of every result.
- **Regression:** negative binomial model of district-year ticket counts with households as the exposure, dispersion estimated by
  maximum likelihood, and standard errors clustered by district.
        """
    )
    st.subheader("Limitations")
    st.markdown(
        """
- **Small sample.** There are only 15 districts, so only large effects can be detected and clustered p-values are optimistic.
- **Resident-based rates.** Rates divide by residents, households and vehicles in each district. Visitors and commuters also get
  tickets, which likely explains why District 14 sits above every trend line.
- **Districts, not change over time.** The predictors are constant across years, so results compare districts to each other.
- **Exploratory housing family.** The housing and transit tests were chosen after seeing earlier results, so treat them as leads,
  not confirmed findings. The per-vehicle checks were also added after the first results.
- **Association only.** None of these results show that renters or housing cause more tickets.
        """
    )

# ===================== Writeup tab =====================
with tab_writeup:
    text = (Path(__file__).parent / "data" / "writeup.md").read_text(encoding="utf-8")
    before, after = text.split("### Housing and transit", 1)
    st.markdown(before)
    st.image("data/results/trend_plots.png", caption="Figure 1: trend fits for the primary tests")
    st.markdown("### Housing and transit" + after)
    st.image("data/results/trend_plots_housing.png", caption="Figure 2: trend fits for housing and transit")