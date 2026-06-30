# Incident Intelligence Dashboard

An interactive **Streamlit** dashboard for exploring and investigating incidents integrated from multiple data sources. The application provides high-level analytics, filtering, and drill-through capabilities that allow users to trace integrated incidents back to their original source records.

---

## Features

- 📊 Interactive dashboard with key performance indicators (KPIs)
- 🔎 Filter incidents by:
  - Source
  - Severity
  - Event type
  - Keyword search
- 📈 Visual analytics including:
  - Incidents by Source
  - Incidents by Severity
- 🚨 High Severity incident review table
- 📄 View all filtered incidents
- ⬇️ Export filtered results as CSV
- 🔍 Drill-through functionality to inspect original source records for each integrated incident
- ⚡ Fast loading through Streamlit data caching

---

## Project Structure

```
.
├── app.py                              # Streamlit application
├── requirements.txt                    # Python dependencies
├── final_integrated_incidents.csv      # Integrated incident dataset
├── final_integration.ipynb             # Data integration notebook
├── multimodal_pipeline_architecture.pdf
└── .streamlit/
    └── config.toml
```

The application is also designed to work with optional source datasets located in sibling folders:

```
audio/
    audio_output.csv

image/
    outputs/results.csv

text/
    Text.csv

pdf/
    pdf_output.csv
```

If these folders are unavailable, the application will attempt to load local copies in the project directory.

---

## Requirements

- Python 3.9+
- Streamlit
- Pandas
- Plotly

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Application

Start the Streamlit server:

```bash
streamlit run app.py
```

The dashboard will open in your default web browser.

---

## Dashboard Overview

### Overview Tab

Provides an executive summary of the integrated incident dataset, including:

- Total Incidents
- High Severity Incidents
- Number of Sources
- Average Incident Score

Visualizations include:

- Incidents by Source
- Incidents by Severity

Additional tables display:

- High Severity incidents
- Complete filtered incident list

Users can export filtered data as a CSV file.

---

### Incident Drill Through

The drill-through page enables users to:

1. Search incidents by:
   - Incident ID
   - Source
   - Event
   - Location
   - Severity
2. Select an incident from a filtered dropdown.
3. View the integrated incident summary.
4. Inspect the corresponding original source record for detailed investigation.

---

## Data

The primary dataset is:

```
final_integrated_incidents.csv
```

Expected fields include:

| Column | Description |
|---------|-------------|
| Incident_ID | Unique incident identifier |
| Source | Source system or modality |
| Event | Incident/event type |
| Location | Event location |
| Time | Event timestamp |
| Score | Confidence or relevance score |
| Severity | Incident severity |

---

## Technologies Used

- Streamlit
- Pandas
- Plotly Express
- Python

---

## Performance

The application uses Streamlit's caching (`@st.cache_data`) to minimize repeated CSV loading and improve dashboard responsiveness.

---

## Customization

The dashboard can be customized by:

- Replacing the integrated incident dataset
- Modifying the color palettes in `app.py`
- Adding additional visualizations
- Extending the filtering options
- Integrating additional data sources

---

## Future Improvements

Potential enhancements include:

- Database backend support
- Real-time incident ingestion
- Authentication and user management
- Geographic mapping of incidents
- Advanced analytics and trend detection
- Machine learning-based incident prioritization
- PDF report generation

---

## License

This project is provided for educational and research purposes. Modify and distribute according to your organization's licensing requirements.