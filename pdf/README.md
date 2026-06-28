# LESO 1033 MRAP Training Plan — Dynamic Extraction Pipeline

A fully automated Python pipeline that extracts structured data from scanned PDF documents (FOIA-released Arkansas LESO 1033 MRAP Training Plan Proposals) and outputs a clean CSV. No records are hardcoded — every row is derived from live OCR output.

---

## How It Works

```
PDF → pdf2image (page images) → pytesseract (OCR text)
    → page grouping (boundary detection)
    → regex field extraction
    → CSV output
```

The pipeline runs in six steps:

| Step | Function | What it does |
|------|----------|--------------|
| 1 | `ocr_pdf()` | Converts each PDF page to an image and runs Tesseract OCR |
| 2 | `group_pages()` | Groups consecutive pages into logical documents using boundary signals |
| 3 | `extract_fields()` | Applies regex patterns to extract structured fields from each document group |
| 4 | `build_records()` | Assembles all extracted fields into ordered records with auto-generated IDs |
| 5 | `write_csv()` | Writes records to a UTF-8 CSV file |
| 6 | `print_summary()` | Prints a console summary of extracted document types, agencies, and years |

---

## Output Fields

Each CSV row contains the following columns:

| Field | Description |
|-------|-------------|
| `Report_ID` | Auto-generated ID (e.g. `RPT_001`) |
| `Incident_Type` | Classified document type (e.g. Training Lesson Plan, Invoice, SOP) |
| `Date` | First date found in the document |
| `Location` | City/location in Arkansas extracted via regex |
| `Agency` | Law enforcement agency name (most frequent match) |
| `Officer` | First ranked officer/signer name found |
| `Summary` | First 5 meaningful lines of the document, truncated to 300 chars |
| `Suspect_Description` | Always `N/A` (not applicable for this document type) |
| `Outcome` | Keyword-matched outcome or last meaningful sentence |
| `Source_Pages` | Page range from the original PDF (e.g. `3-7`) |

---

## Configuration

Edit the `CONFIG` block near the top of the script:

```python
PDF_PATH    = "/path/to/your/file.pdf"         # Input PDF
OUTPUT_PATH = "/path/to/output.csv"            # Output CSV
OCR_DPI     = 200   # Higher DPI = better quality, slower processing
SUMMARY_LEN = 300   # Max characters for the Summary field
```

---

## Usage

Run directly from the command line:

```bash
python leso_extraction_pipeline.py
```

Or pass a PDF path as an argument (requires uncommenting `sys.argv` line in `__main__`):

```bash
python leso_extraction_pipeline.py /path/to/file.pdf
```

---

## Installation

### Python dependencies

```bash
pip install pdf2image pytesseract
```

### System dependencies

**macOS (Homebrew):**
```bash
brew install tesseract poppler
```

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr poppler-utils
```

**Windows:**
- Install [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
- Install [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases)
- Add both to your system PATH

---

## Document Type Classification

The pipeline classifies each logical document group into one of the following types using regex keyword matching:

- Invoice
- Internal Memorandum
- LEA Equipment Request
- Training Lesson Plan
- Policies and Procedures
- Standard Operating Procedure
- LESO Intended Use Declaration
- Vendor Capability Statement
- Training Certificate
- MRAP Policy Letter
- Aviation Unit Policy
- Policy/Correspondence *(default fallback)*

---

## Notes

- Designed for **scanned PDFs** — Tesseract OCR handles documents without a digital text layer.
- Page grouping relies on detecting headers like `To Whom It May Concern`, `RE:`, `Lesson Plan`, `MEMORANDUM`, etc. Results will vary with OCR quality and DPI settings.
- OCR quality scales with DPI: `200` is a reasonable default; increase to `300` for better accuracy at the cost of speed.
- The pipeline is stateless — re-running it will overwrite the output CSV.
