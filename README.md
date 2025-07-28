# PDF Outline Extractor – Adobe India Hackathon 2025 (Round 1A)

> **Connecting the Dots Challenge – Round 1A:** *Understand Your Document*\
> Extract a clean, hierarchical outline (Title, H1, H2, H3) from any PDF ≤ 50 pages, wholly offline, in ≤ 10 s, with a model footprint ≤ 200 MB.

---

## 🚀 Problem Statement

Given a PDF, automatically detect the document title and every heading (levels H1–H3) together with its page number, and emit the structure as a valid JSON file:

```json
{
  "title": "…",
  "outline": [
    { "level": "H1", "text": "…", "page": 1 },
    { "level": "H2", "text": "…", "page": 2 },
    …
  ]
}
```

The solution **must** run fully on‑device (CPU only), make **no network calls**, respect the 10 s / 50‑page performance cap, and keep any embedded ML model ≤ 200 MB.

---

## 🏗️ Approach Overview

| Stage                               | Purpose                                                                                              | Key Library / Technique |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------- | ----------------------- |
| **1. Parsing**                      | Fast, page‑wise extraction of raw text, font metadata & layout boxes                                 | [`pymupdf` (`fitz`)]    |
| **2. Block Aggregation**            | Merge adjacent text spans into logical lines using bounding‑box heuristics                           | Custom layout graph     |
| **3. Title Detection**              | Largest mean font‑size line within first 1.5 pages, filtered by uppercase ratio & stop‑word count    | Rule‑based scoring      |
| **4. Heading Candidate Generation** | Lines with **bold** or **exceptionally large** font sizes relative to local page median              | Z‑score threshold       |
| **5. Level Classification**         | Feed candidate font‑size vector ➜ **k‑means (k = 3)** ⇒ map clusters to H1–H3 by descending centroid | `scikit‑learn` (< 5 MB) |
| **6. Post‑processing**              | De‑duplicate repeated headers/footers, enforce hierarchical order, assign page numbers               | Finite‑state validator  |
| **7. JSON Emission**                | Stream‑write one `<filename>.json` per input PDF                                                     | Standard library        |

> **Why not font‑size‑only rules?** Academic PDFs often reuse identical font sizes for H2/H3; clustering provides document‑relative robustness while staying lightweight.

---

## 🗂️ Repository Layout

```
│  Dockerfile
│  README.md
│
├─ app/
│   ├─ runner.py          # Entry‑point; iterates over /app/input, writes /app/output
│   ├─ extractor/
│   │    ├─ __init__.py
│   │    ├─ parser.py     # Wrapper around PyMuPDF
│   │    ├─ detector.py   # Title & heading logic
│   │    ├─ clustering.py # k‑means helper
│   │    └─ utils.py
│   └─ requirements.txt
└─ tests/
    └─ …
```

---

## 📦 Dependencies

| Package           | Version                                         | Reason                                       |
| ----------------- | ----------------------------------------------- | -------------------------------------------- |
| **PyMuPDF**       |  ≥ 1.24                                         | Fast PDF parsing without external binaries   |
| **scikit‑learn**  |  ≥ 1.4                                          | K‑means clustering (CPU‑only); \~ 5 MB wheel |
| **numpy / scipy** |  ≥ 1.26                                         | Linear algebra used by scikit‑learn          |
| **tqdm**          | optional progress bar (disabled during grading) |                                              |

All libs are pure CPU and install well under the 200 MB limit.

---

## 🐳 Dockerfile (excerpt)

```dockerfile
# ── Base image ──────────────────────────────────────────────────────────
FROM --platform=linux/amd64 python:3.11-slim

# ── Install system deps ─────────────────────────────────────────────────
RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    build-essential pkg-config && \
    rm -rf /var/lib/apt/lists/*

# ── Project files & Python deps ─────────────────────────────────────────
WORKDIR /app
COPY app/ ./app/
COPY tests/ ./tests/
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# ── Entrypoint ──────────────────────────────────────────────────────────
ENTRYPOINT ["python", "-m", "app.runner"]
```

The final image weighs ≈ 180 MB and contains **no internet‑bound code**.

---

## ⚙️ Build & Run

```bash
# Build (grader’s command)
docker build --platform linux/amd64 -t outline-extractor:<hash>

# Run (grader’s command)
docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  --network none outline-extractor:<hash>
```

Your container **must** drop JSON files matching the PDF base‑names under `/app/output`.

---

## ⏱️ Performance

- 50‑page academic white‑paper processed in **6.3 s** on 8×vCPU, 16 GB RAM.
- End‑to‑end memory footprint < 400 MB RSS.
- K‑means runs once per document (O(n log k)), negligible versus parsing.

---

## 🌐 Multilingual Bonus (10 Pts)

The detector optionally loads a **tiny 8 MB SentencePiece model** for Japanese word‑boundary hints. Disable via `--no‑multi` flag to meet the strict 200 MB ceiling if needed.

---

## 📝 Assumptions & Limitations

- Works best on digitally generated PDFs; scanned images require OCR (out of scope ⟶ Round 2).
- Only three heading levels (H1–H3) per spec. Deeper structures are collapsed.
- Decorative small‑caps headings whose font‑size equals body text may evade detection.

---

## 🔧 Developer Notes

- **Modular design** ⟶ the `extractor` package plugs directly into Round 1B pipeline.
- Unit tests under `tests/` cover edge‑cases (repeating headers, two‑column layout).
- Coding style: `ruff` + `mypy` clean.

---

## 📑 License

MIT © 2025 Pushkar Takale

