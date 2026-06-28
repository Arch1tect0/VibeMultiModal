# Audio Analyst — Multimodal Incident Report Analyzer

**Student Role:** Student 1 — Audio Analyst  
**Modality:** Emergency audio calls / witness voice statements  
**Output:** `audio_output.csv`

---

## Overview

This module transcribes real 911 emergency audio calls and extracts structured incident information including event type, location, sentiment, and urgency score.

---

## Dataset

- **Name:** 911 Recordings First 6 Seconds
- **Source:** Kaggle — `911-recordings-first-6-seconds`
- **Path on Kaggle:** `/kaggle/input/911-recordings-first-6-seconds/911_first6sec/`
- **Format:** `.wav` audio files + `911_metadata.csv`

---

## Models & Tools

| Tool | Purpose |
|------|---------|
| OpenAI Whisper Small (via HuggingFace) | Speech-to-text transcription |
| spaCy `en_core_web_sm` | Named Entity Recognition for location extraction |
| Keyword matching | Event type classification |
| Custom scoring function | Urgency score (0–1) and sentiment labeling |

---

## Output Schema

File: `audio_output.csv`

| Column | Description |
|--------|-------------|
| `Call_ID` | Unique ID in `AUD-001` format |
| `Transcript` | Full transcribed text from audio |
| `Extracted_Event` | Classified incident type (e.g. Homicide, Fire, Traffic Accident) |
| `Location` | Location entity extracted by spaCy |
| `Sentiment` | Calm / Concerned / Distressed |
| `Urgency_Score` | Float 0–1 based on keyword frequency |

### Urgency Score Thresholds
- **0.0 – 0.19** → Calm
- **0.20 – 0.59** → Concerned
- **0.60 – 1.0** → Distressed

Score = number of urgency keyword hits / 3, capped at 1.0

---

## Requirements

```
openai-whisper==20230124
transformers
torch
librosa
pandas
spacy
```

Install with:
```bash
pip install openai-whisper transformers torch librosa pandas spacy
python -m spacy download en_core_web_sm
```

---

## How to Run

1. Upload notebook to Kaggle
2. Add the following datasets as inputs:
   - `911-recordings-first-6-seconds`
   - `bond005/openai-whisper-small` (HuggingFace Whisper Small weights)
3. Enable **GPU accelerator** and **Internet** in session settings
4. Run cells in order — output saves to `/kaggle/working/audio_output.csv`

---

## What Worked

- ✅ **Whisper Small** via HuggingFace `transformers` successfully transcribed all 20 audio clips
- ✅ Loading Whisper model weights from a local Kaggle dataset (`bond005/openai-whisper-small`) bypassed internet download issues
- ✅ `spaCy en_core_web_sm` correctly extracted location entities (e.g. "the Hudson River", "St. James", "the Stevens Bar")
- ✅ Keyword-based event extraction correctly identified Homicide, Shooting, Traffic Accident, Medical Emergency, and Fire events
- ✅ Urgency scoring produced meaningful differentiation (e.g. "I just killed my daughter" scored 0.67 Distressed vs simple calls scoring 0.0 Calm)
- ✅ Output CSV saved cleanly with all required columns in `AUD-` prefixed format

---

## What Did NOT Work

- ❌ **Original Kaggle notebook (Wav2Vec2)** — failed with `MissingSchema` error due to Kaggle's internal model cache proxy returning a broken relative URL instead of a full HTTPS URL. Unfixable with the environment's old `transformers` version.
- ❌ **`pip install moviepy`** — failed initially due to Kaggle internet being disabled by default. Fixed by enabling internet in session settings, but `moviepy` was ultimately not needed and dropped entirely.
- ❌ **Wav2Vec2 model from Kaggle Models tab (TensorFlow2 version)** — incompatible with the PyTorch-based notebook. Had to find the PyTorch variant instead.
- ❌ **Upgrading `transformers` to fix Wav2Vec2** — even after upgrading, the old PyTorch version (1.7.0) on the Python 3.7 Kaggle environment caused `AttributeError: 'Tensor' object has no attribute 'tile'`, making Wav2Vec2 unusable.
- ❌ **Loading Whisper tokenizer locally** — the `bond005/openai-whisper-small` dataset was missing `vocab.json`, so the tokenizer could not be loaded from that path. Workaround: upgrade `transformers` and load via the model path directly using `WhisperProcessor`.
- ❌ **`WhisperProcessor` import on old transformers** — `cannot import name 'WhisperProcessor'` on the default Kaggle Python 3.7 environment. Fixed by running `pip install --upgrade transformers` and restarting the kernel.

---

## Solution Path Summary

1. Tried Wav2Vec2 → failed (environment too old)
2. Tried upgrading transformers for Wav2Vec2 → failed (PyTorch too old)
3. Switched to Whisper Small → worked after upgrading transformers and restarting kernel
4. Loaded model weights from local Kaggle dataset to avoid internet dependency
5. Used GPU accelerator to resolve PyTorch version conflicts
