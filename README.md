# Consumer Tech Review Intelligence

Local React/Vite frontend and FastAPI backend for the supplied 9,000-review
Amazon/Amazon UK sample. Whole-review sentiment uses the existing
`sentiment_pipeline.pkl`; aspect opinions use the
[MIT-licensed `yangheng/deberta-v3-base-absa-v1.1` model](https://huggingface.co/yangheng/deberta-v3-base-absa-v1.1)
with paired review-context and target-aspect inputs. The aspect model loads on
the first aspect-analysis request and downloads roughly 700 MB to the local
model cache if it is not already present. Review text is never sent to a hosted
inference API.

## Setup and start

From this folder in PowerShell, install the Python dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Start the backend in one terminal:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

Start the frontend in a second terminal:

```powershell
Set-Location .\frontend
npm install
npm run dev
```

Open the Vite URL shown in the frontend terminal (normally
`http://localhost:5173`). The API is available at `http://localhost:8000` and
its interactive documentation at `http://localhost:8000/docs`. The existing
`sentiment_pipeline.pkl` and `consumer_tech_reviews_9000.csv` must remain beside
`api.py`.

Run focused checks from the project root:

```powershell
python -m unittest test_dashboard.py
```

## Product metadata

`product_metadata.csv` is the editable local mapping template. Source-provided
product titles are shown where available; otherwise the dashboard uses product
IDs. The comparison page can compare any two named products with review data;
verified categories are an optional filter, and results label a category only
when both products share the same verified category. Add a mapping row only
after checking each name, brand, or category against a reliable source, include
that source in `metadata_source`, and set `verification_status` to `verified`.
Only explicitly mapped verified brands and categories enable those filters.
Unverified entries are ignored. Fields are `product_id`, `product_name`,
`brand`, `category`, `metadata_source`, and `verification_status`.

## Data and model limitations

The dashboard reports predictions on a sampled training dataset, not a
representative population survey. Its 7,500 older records have year-only dates
and unverified keyword-based category hints. The other 1,500 records have
Amazon UK source product titles and day-level dates; their category hints are
not assumed to be verified. There are currently no verified category mappings,
so the optional category filter is unavailable. Brand filters likewise require
verified mappings. Current coverage is
Amazon and Amazon UK only, with no live social-media feeds or 2025–2026 data.
Overall source labels were derived from star ratings and are not human
sentiment annotations.

The aspect model predicts only negative, neutral, and positive. The inference
layer separately represents factual mentions without opinion cues, uncertain
comparison contexts, and mixed review evidence; those are not claimed as
pretrained model classes. If the aspect model cannot load, the API reports the
weaker, sentence-level fallback. Uploaded CSVs must contain `review_text`, are
limited to 10 MB, and are processed locally; rating/sentiment columns are not
model inputs.

`sentiment.ipynb` contains the grouped overall-sentiment training and
evaluation workflow. `aspect_analysis.ipynb` retains the original rules
baseline and includes the pretrained aspect-model and human-annotation
evaluation workflow. The annotation template is separate from model
predictions; automatic labels are never treated as human ground truth.
