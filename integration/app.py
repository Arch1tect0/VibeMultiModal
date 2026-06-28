"""
Incident Intelligence Dashboard
Explore integrated incidents and drill through to original source records.
"""

import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path

# ---------------------------------------------------------------------------
# Color palette and chart styling
# ---------------------------------------------------------------------------
SOURCE_COLORS = ["#6B8FBA", "#82A894", "#C4A484", "#9B8BB4", "#7BA7BC", "#A8B8A0"]
SEVERITY_COLORS = {"High": "#C97C7C", "Medium": "#D4A85A", "Low": "#7BAE96"}

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#FAFBFD",
    font=dict(color="#2F3A4A", size=13),
    margin=dict(t=20, b=20, l=20, r=20),
    height=350,
    xaxis=dict(gridcolor="#E8EDF3"),
    yaxis=dict(gridcolor="#E8EDF3"),
)

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Incident Intelligence Dashboard",
    page_icon="🔍",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; max-width: 1200px; }
    h1 { color: #2F3A4A; font-weight: 700; }
    h2, h3 { color: #3E4C61; font-weight: 600; }
    .dashboard-subtitle { color: #5C6B7A; font-size: 1.05rem; margin-bottom: 1.5rem; }
    .kpi-card {
        background: #FFFFFF; border: 1px solid #E3E9F0; border-radius: 12px;
        padding: 1rem 1.1rem; box-shadow: 0 1px 3px rgba(47, 58, 74, 0.06);
    }
    .kpi-label {
        color: #6B7A8C; font-size: 0.85rem; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.04em;
    }
    .kpi-value { color: #2F3A4A; font-size: 1.8rem; font-weight: 700; }
    span[data-baseweb="tag"] {
        background-color: #E8EEF6 !important; color: #2F3A4A !important;
        border: 1px solid #D5DEEA !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Load datasets — paths work in the repo (integration/) or local project folder
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
REPO_ROOT = BASE_DIR.parent


def resolve_data_file(*candidates: Path) -> str:
    """Use the first path that exists so the app runs locally and on GitHub."""
    for path in candidates:
        if path.exists():
            return str(path)
    return str(candidates[0])


INTEGRATED_FILE = resolve_data_file(BASE_DIR / "final_integrated_incidents.csv")
AUDIO_FILE = resolve_data_file(
    REPO_ROOT / "audio" / "audio_output.csv",
    BASE_DIR / "audio_output.csv",
)
IMAGE_FILE = resolve_data_file(
    REPO_ROOT / "image" / "outputs" / "results.csv",
    BASE_DIR / "Image.csv",
)
TEXT_FILE = resolve_data_file(
    REPO_ROOT / "text" / "Text.csv",
    BASE_DIR / "Text.csv",
)
PDF_FILE = resolve_data_file(
    REPO_ROOT / "pdf" / "pdf_output.csv",
    BASE_DIR / "pdf.csv",
)

DISPLAY_COLUMNS = [
    "Incident_ID", "Source", "Event", "Location", "Time", "Score", "Severity",
]


@st.cache_data
def load_csv(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)


integrated_df = load_csv(INTEGRATED_FILE)
audio_df = load_csv(AUDIO_FILE)
image_df = load_csv(IMAGE_FILE)
text_df = load_csv(TEXT_FILE)
pdf_df = load_csv(PDF_FILE)

display_df = integrated_df[[c for c in DISPLAY_COLUMNS if c in integrated_df.columns]]


def filter_incidents_by_text(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """Return rows where any key text field contains the search query."""
    if not query.strip():
        return df

    search = query.strip().lower()
    return df[
        df["Incident_ID"].astype(str).str.lower().str.contains(search, na=False)
        | df["Source"].astype(str).str.lower().str.contains(search, na=False)
        | df["Event"].astype(str).str.lower().str.contains(search, na=False)
        | df["Location"].astype(str).str.lower().str.contains(search, na=False)
        | df["Severity"].astype(str).str.lower().str.contains(search, na=False)
    ]


def incident_option_label(row: pd.Series) -> str:
    """Build a readable dropdown label (like a Power BI slicer display)."""
    return (
        f"{row['Incident_ID']}  ·  {row['Source']}  ·  "
        f"{row['Event']}  ·  {row['Severity']}"
    )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Incident Intelligence Dashboard")
st.markdown(
    '<p class="dashboard-subtitle">Analyze incidents extracted from multiple data sources '
    "and drill down into the original records for deeper investigation.</p>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar filters (apply to both tabs)
# ---------------------------------------------------------------------------
st.sidebar.header("🔎 Filters")

selected_sources = st.sidebar.multiselect(
    "Source",
    options=sorted(display_df["Source"].dropna().unique()),
    default=sorted(display_df["Source"].dropna().unique()),
)
selected_severities = st.sidebar.multiselect(
    "Severity",
    options=sorted(display_df["Severity"].dropna().unique()),
    default=sorted(display_df["Severity"].dropna().unique()),
)
selected_events = st.sidebar.multiselect(
    "Event",
    options=sorted(display_df["Event"].dropna().unique()),
    default=sorted(display_df["Event"].dropna().unique()),
)
keyword = st.sidebar.text_input(
    "Keyword search",
    placeholder="Search Incident ID, Event, or Location...",
)

filtered_df = display_df.copy()

if selected_sources:
    filtered_df = filtered_df[filtered_df["Source"].isin(selected_sources)]
else:
    filtered_df = filtered_df.iloc[0:0]

if selected_severities:
    filtered_df = filtered_df[filtered_df["Severity"].isin(selected_severities)]
else:
    filtered_df = filtered_df.iloc[0:0]

if selected_events:
    filtered_df = filtered_df[filtered_df["Event"].isin(selected_events)]
else:
    filtered_df = filtered_df.iloc[0:0]

filtered_df = filter_incidents_by_text(filtered_df, keyword)

total_incidents = len(filtered_df)
high_severity_count = (filtered_df["Severity"] == "High").sum()
num_sources = filtered_df["Source"].nunique()
avg_score = filtered_df["Score"].mean() if total_incidents > 0 else 0.0

# ---------------------------------------------------------------------------
# Main tabs — Overview vs Drill Through
# ---------------------------------------------------------------------------
tab_overview, tab_drill = st.tabs(["📊 Overview", "🔍 Incident Drill Through"])

# ========================= OVERVIEW TAB =====================================
with tab_overview:
    k1, k2, k3, k4 = st.columns(4)
    for col, label, value in [
        (k1, "Total Incidents", f"{total_incidents:,}"),
        (k2, "High Severity", f"{high_severity_count:,}"),
        (k3, "Sources", f"{num_sources:,}"),
        (k4, "Average Score", f"{avg_score:.2f}"),
    ]:
        with col:
            st.markdown(
                f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
                f'<div class="kpi-value">{value}</div></div>',
                unsafe_allow_html=True,
            )

    st.divider()

    chart_left, chart_right = st.columns(2)

    with chart_left:
        st.subheader("Incidents by Source")
        if total_incidents > 0:
            source_counts = filtered_df["Source"].value_counts().reset_index()
            source_counts.columns = ["Source", "Count"]
            fig = px.bar(source_counts, x="Source", y="Count", text="Count")
            fig.update_traces(
                marker_color=SOURCE_COLORS[: len(source_counts)],
                textposition="outside",
            )
            fig.update_layout(**CHART_LAYOUT, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No incidents match the current filters.")

    with chart_right:
        st.subheader("Incidents by Severity")
        if total_incidents > 0:
            order = ["High", "Medium", "Low"]
            sev_counts = filtered_df["Severity"].value_counts().reindex(order, fill_value=0)
            sev_df = sev_counts.reset_index()
            sev_df.columns = ["Severity", "Count"]
            fig = px.bar(sev_df, x="Severity", y="Count", text="Count",
                         category_orders={"Severity": order})
            fig.update_traces(
                marker_color=[SEVERITY_COLORS.get(s, "#A0A8B3") for s in sev_df["Severity"]],
                textposition="outside",
            )
            fig.update_layout(**CHART_LAYOUT, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No incidents match the current filters.")

    st.divider()

    st.subheader("High Severity Review")
    high_df = filtered_df[filtered_df["Severity"] == "High"]
    if len(high_df) > 0:
        st.dataframe(high_df, use_container_width=True, hide_index=True)
    else:
        st.info("No high severity incidents match the current filters.")

    st.divider()

    st.subheader("All Filtered Incidents")
    st.caption(f"Showing {total_incidents:,} incident(s) based on your filter selections.")

    if total_incidents > 0:
        st.dataframe(filtered_df, use_container_width=True, hide_index=True)
        st.download_button(
            label="Download filtered CSV",
            data=filtered_df.to_csv(index=False).encode("utf-8"),
            file_name="filtered_incidents.csv",
            mime="text/csv",
        )
    else:
        st.info("No incidents match the current filters. Try adjusting your sidebar selections.")

# ========================= DRILL THROUGH TAB ===============================
with tab_drill:
    st.subheader("Incident Drill Through")
    st.markdown(
        "Trace any integrated incident back to its original source record. "
        "Use the search box to filter the list, then pick an incident from the dropdown."
    )

    if total_incidents == 0:
        st.warning("Apply filters that return at least one incident to use drill-through.")
    else:
        # Power BI-style slicer: type to filter, then select from narrowed list
        search_col, count_col = st.columns([4, 1])
        with search_col:
            drill_search = st.text_input(
                "Search incidents",
                placeholder="Type to match ID, Source, Event, Location, or Severity...",
                key="drill_search",
                help="Results update as you type — similar to a Power BI slicer search.",
            )
        with count_col:
            drill_matches = filter_incidents_by_text(filtered_df, drill_search)
            st.metric("Matches", f"{len(drill_matches):,}")

        if len(drill_matches) == 0:
            st.warning("No incidents match your search. Try a different keyword.")
        else:
            # Build readable dropdown options from search results
            option_labels = drill_matches.apply(incident_option_label, axis=1).tolist()
            label_to_id = dict(
                zip(drill_matches.apply(incident_option_label, axis=1), drill_matches["Incident_ID"])
            )

            selected_label = st.selectbox(
                "Select an Incident ID",
                options=option_labels,
                help="Dropdown only shows incidents that match your search text.",
            )
            selected_incident = label_to_id[selected_label]

            incident_row = integrated_df[
                integrated_df["Incident_ID"] == selected_incident
            ].iloc[0]

            # Summary card — use native bordered container (HTML wrappers break with widgets)
            st.markdown("### Integrated Incident Summary")
            with st.container(border=True):
                m1, m2, m3 = st.columns(3)
                m1.metric("Incident ID", incident_row["Incident_ID"])
                m2.metric("Source", incident_row["Source"])
                m3.metric("Severity", incident_row["Severity"])

                m4, m5, m6 = st.columns(3)
                m4.metric("Event", incident_row["Event"])
                m5.metric("Location", incident_row["Location"])
                m6.metric("Time", incident_row["Time"])

                st.metric("Score", f"{incident_row['Score']:.2f}")

            incident_id = str(selected_incident)
            original_id = str(incident_row.get("Original_ID", incident_id))

            source_df = None
            id_column = None

            if incident_id.startswith("AUD"):
                source_df = audio_df
                id_column = "Call_ID"
            elif incident_id.startswith("IMG"):
                source_df = image_df
                id_column = "Image_ID"
            elif incident_id.startswith("TXT"):
                source_df = text_df
                id_column = "Text_ID"
            elif incident_id.startswith("DOC"):
                source_df = pdf_df
                id_column = "Report_ID"

            if source_df is None:
                st.warning("Original source record could not be found.")
                st.caption(f"Unknown incident prefix for: {incident_id}")
            else:
                match = source_df[source_df[id_column].astype(str) == original_id]

                if match.empty:
                    st.warning("Original source record could not be found.")
                    st.caption(f"Looked for {id_column} = {original_id}")
                else:
                    original_record = match.iloc[0]
                    st.subheader("Original Source Record")

                    if incident_id.startswith("TXT"):
                        st.markdown("**Text source details**")
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Topic", original_record.get("Topic", "N/A"))
                        c2.metric("Sentiment", original_record.get("Sentiment", "N/A"))
                        c3.metric("Confidence Score", incident_row.get("Score", "N/A"))
                        if pd.notna(original_record.get("Raw_Text")):
                            st.markdown("**Original text / article**")
                            st.info(str(original_record["Raw_Text"]))

                    elif incident_id.startswith("AUD"):
                        st.markdown("**Audio source details**")
                        c1, c2 = st.columns(2)
                        c1.metric("Detected Event", original_record.get("Extracted_Event", "N/A"))
                        c2.metric("Urgency Score", original_record.get("Urgency_Score", "N/A"))
                        if pd.notna(original_record.get("Transcript")):
                            st.markdown("**Transcript**")
                            st.info(str(original_record["Transcript"]))

                    elif incident_id.startswith("IMG"):
                        st.markdown("**Image source details**")
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Detected Objects", original_record.get("Objects_Detected", "N/A"))
                        c2.metric("OCR Text", original_record.get("Text_Extracted", "N/A") or "None")
                        c3.metric("Confidence Score", original_record.get("Scene_Decision_Confidence", "N/A"))

                    elif incident_id.startswith("DOC"):
                        st.markdown("**PDF source details**")
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Document Category", original_record.get("Incident_Type", "N/A"))
                        c2.metric("Confidence Score", incident_row.get("Score", "N/A"))
                        c3.metric("Report ID", original_record.get("Report_ID", "N/A"))
                        if pd.notna(original_record.get("Summary")):
                            st.markdown("**Extracted text**")
                            st.info(str(original_record["Summary"]))

                    with st.expander("View Full Source Details"):
                        st.dataframe(
                            pd.DataFrame([original_record]),
                            use_container_width=True,
                            hide_index=True,
                        )
