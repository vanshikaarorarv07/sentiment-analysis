"""Shared preprocessing and inference for the review dashboard and notebooks."""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "consumer_tech_reviews_9000.csv"
SENTIMENT_MODEL_PATH = ROOT / "sentiment_pipeline.pkl"
PRODUCT_METADATA_PATH = ROOT / "product_metadata.csv"
ABSA_MODEL_ID = "yangheng/deberta-v3-base-absa-v1.1"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ABSA_CONFIDENCE_THRESHOLD = 0.60

ASPECT_KEYWORDS = {
    "battery": ("battery life", "battery", "battery backup"),
    "sound": ("sound quality", "sound", "audio", "bass", "volume"),
    "camera": ("camera", "photo", "picture", "video quality"),
    "display": ("screen", "display", "brightness"),
    "performance": ("performance", "speed", "lag", "slow", "crashing"),
    "comfort": ("comfortable", "uncomfortable", "comfort", "ear pain"),
    "build quality": ("build quality", "durable", "flimsy", "sturdy"),
    "price/value": ("price", "expensive", "cheap", "value for money"),
    "connectivity": ("bluetooth", "wifi", "wi-fi", "connection"),
    "charging": ("charging", "charger", "charges"),
}

_OPINION_CUES = (
    "amazing", "annoying", "awesome", "bad", "barely", "beautiful", "best",
    "broke", "broken",
    "cheap", "comfortable", "crack", "cracked", "crashes", "crashing",
    "dark", "dim", "difficult", "disappoint", "disconnects", "drains", "easy",
    "excellent", "fail", "failed", "failing", "fantastic", "flimsy", "hurt",
    "frustrating", "good", "great", "hate", "horrible", "impressive",
    "lasted", "lasts", "long", "long-lasting", "love", "low", "no longer charges", "not working",
    "only", "overheats", "overpriced", "poor", "perfect", "problem",
    "quality", "refused to charge", "reliable", "short", "slow", "stopped working",
    "sturdy", "sucks", "terrible", "uncomfortable", "unreliable", "unusable",
    "useless", "weak", "works well", "worst", "worth", "can't", "cannot",
    "doesn't work", "does not work", "won't", "wouldn't",
)
_OTHER_PRODUCT_CUE = re.compile(
    r"\b(?:my old|previous|another|other|compared (?:to|with)|"
    r"versus|vs\.?)\b",
    re.IGNORECASE,
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|[;\n]+")
_CONTRAST_SPLIT = re.compile(
    r"\b(?:but|however|although|yet|though)\b", re.IGNORECASE
)
_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")


class UploadValidationError(ValueError):
    """Raised when a user-provided review CSV cannot be safely analysed."""


def clean_text(text: object) -> str:
    """Match training preprocessing while preserving the original evidence elsewhere."""
    value = "" if text is None else str(text)
    value = value.lower()
    value = _URL.sub(" ", value)
    return _WHITESPACE.sub(" ", value).strip()


def load_sentiment_pipeline(path: Path = SENTIMENT_MODEL_PATH):
    """Load the trained overall-sentiment pipeline."""
    if not path.is_file():
        raise FileNotFoundError(f"Sentiment pipeline not found: {path}")
    return joblib.load(path)


def load_absa_model():
    """Load the published CPU-compatible ABSA classifier and tokenizer."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(ABSA_MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(ABSA_MODEL_ID)
    labels = {
        str(label).lower()
        for label in model.config.id2label.values()
    }
    if labels != {"negative", "neutral", "positive"}:
        raise ValueError(f"Unexpected ABSA model labels: {labels}")
    model.to(torch.device("cpu"))
    model.eval()
    return model, tokenizer


def load_reviews(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the supplied source records without replacing their provenance."""
    frame = pd.read_csv(path)
    if "review_text" not in frame.columns:
        raise ValueError("Review data must contain a review_text column.")
    frame["review_text"] = frame["review_text"].fillna("").astype(str)
    frame = frame.loc[frame["review_text"].str.strip().ne("")].copy()
    frame["product_id"] = frame.get("product_id", pd.Series("", index=frame.index))
    frame["product_id"] = frame["product_id"].fillna("").astype(str)
    return frame.reset_index(drop=True)


def load_product_metadata(path: Path = PRODUCT_METADATA_PATH) -> pd.DataFrame:
    required = {
        "product_id",
        "product_name",
        "brand",
        "category",
        "metadata_source",
        "verification_status",
    }
    if not path.is_file():
        return pd.DataFrame(columns=sorted(required))
    mapping = pd.read_csv(path, dtype=str).fillna("")
    missing = required - set(mapping.columns)
    if missing:
        raise ValueError(
            f"Product metadata mapping is missing columns: {sorted(missing)}"
        )
    mapping["product_id"] = mapping["product_id"].astype(str).str.strip()
    mapping["verification_status"] = (
        mapping["verification_status"].astype(str).str.strip().str.lower()
    )
    return mapping


def predict_overall(reviews: Iterable[object], model) -> list[str]:
    """Predict overall sentiment using the saved TF-IDF + Logistic Regression model."""
    texts = [clean_text(text) for text in reviews]
    if not texts:
        return []
    predictions = [str(label).lower() for label in model.predict(texts)]
    allowed = {"positive", "negative", "neutral"}
    unexpected = set(predictions) - allowed
    if unexpected:
        raise ValueError(f"Unexpected overall sentiment labels: {unexpected}")
    return predictions


def detect_aspects(text: str) -> list[str]:
    lowered = text.lower()
    found = []
    for aspect, keywords in ASPECT_KEYWORDS.items():
        if any(
            re.search(r"\b" + re.escape(keyword) + r"\b", lowered)
            for keyword in keywords
        ):
            found.append(aspect)
    return found


def aspect_opinion_units(text: str) -> list[str]:
    """Split at sentence and contrast boundaries while retaining evidence wording."""
    units = []
    for sentence in _SENTENCE_SPLIT.split(text):
        units.extend(_CONTRAST_SPLIT.split(sentence))
    return [unit.strip(" \t,") for unit in units if unit.strip(" \t,")]


def has_opinion_cue(text: str) -> bool:
    lowered = clean_text(text)
    return any(
        re.search(
            r"(?<!\w)" + re.escape(cue.strip()) + r"(?!\w)",
            lowered,
        )
        for cue in _OPINION_CUES
    )


def is_possible_other_product_comparison(text: str) -> bool:
    return bool(_OTHER_PRODUCT_CUE.search(text))


def _absa_predictions(contexts: list[str], aspects: list[str], model, tokenizer):
    import torch

    results = []
    id_to_label = {
        int(index): str(label).lower()
        for index, label in model.config.id2label.items()
    }
    if set(id_to_label.values()) != {"negative", "neutral", "positive"}:
        raise ValueError(f"Unexpected ABSA model labels: {id_to_label}")

    for start in range(0, len(contexts), 16):
        batch_contexts = contexts[start : start + 16]
        batch_aspects = aspects[start : start + 16]
        inputs = tokenizer(
            batch_contexts,
            batch_aspects,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with torch.inference_mode():
            probabilities = torch.softmax(model(**inputs).logits, dim=-1)
        for row in probabilities.tolist():
            class_index = max(range(len(row)), key=row.__getitem__)
            results.append((id_to_label[class_index], float(row[class_index])))
    return results


def analyze_aspect_evidence(
    reviews: Iterable[dict],
    overall_model,
    absa_model=None,
    tokenizer=None,
) -> list[dict]:
    """Classify aspect-bearing evidence, preserving neutral and uncertain outcomes."""
    evidence = []
    tasks = []
    for review in reviews:
        text = str(review.get("review_text", ""))
        for unit in aspect_opinion_units(text):
            aspects = detect_aspects(unit)
            for aspect in aspects:
                record = {
                    "review_id": str(review.get("review_id", "")),
                    "review_text": unit,
                    "aspect": aspect,
                    "source_url": str(review.get("source_url", "") or ""),
                    "sentiment": "",
                    "confidence": None,
                    "method": "",
                }
                if is_possible_other_product_comparison(unit):
                    record.update(
                        sentiment="uncertain",
                        method="comparison_context_requires_review",
                    )
                    evidence.append(record)
                elif not has_opinion_cue(unit):
                    record.update(sentiment="no_opinion", method="opinion_cue_check")
                    evidence.append(record)
                elif absa_model is not None and tokenizer is not None:
                    record["method"] = "deberta-v3-base-absa-v1.1"
                    evidence.append(record)
                    tasks.append((len(evidence) - 1, unit, aspect))
                else:
                    label = predict_overall([unit], overall_model)[0]
                    record.update(
                        sentiment=label,
                        method="sentence-level overall model fallback",
                    )
                    evidence.append(record)

    if tasks:
        predictions = _absa_predictions(
            [task[1] for task in tasks],
            [task[2] for task in tasks],
            absa_model,
            tokenizer,
        )
        for (index, _, _), (label, confidence) in zip(tasks, predictions):
            evidence[index]["confidence"] = confidence
            if confidence < ABSA_CONFIDENCE_THRESHOLD:
                evidence[index]["sentiment"] = "uncertain"
            else:
                evidence[index]["sentiment"] = label
    return evidence


def verified_product_catalog(
    reviews: pd.DataFrame,
    metadata: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Use source metadata only; otherwise identify a product solely by its ID."""
    verified_mapping = {}
    if metadata is not None and not metadata.empty:
        approved = metadata.loc[
            metadata["verification_status"].eq("verified")
            & metadata["metadata_source"].astype(str).str.strip().ne("")
            & metadata["product_id"].astype(str).str.strip().ne("")
        ]
        for product_id, group in approved.groupby("product_id"):
            verified_mapping[str(product_id)] = {
                key: (
                    values[0] if len(values := sorted({
                        str(value).strip()
                        for value in group[column].fillna("")
                        if str(value).strip()
                    })) == 1 else ""
                )
                for key, column in (
                    ("display_name", "product_name"),
                    ("brand", "brand"),
                    ("verified_category", "category"),
                )
            }

    rows = []
    for product_id, group in reviews.groupby("product_id", dropna=False, sort=False):
        product_id = str(product_id)
        verified = (
            group.get("category_method", pd.Series("", index=group.index))
            .eq("source_product_metadata")
        )
        names = set()
        if "product_name" in group:
            names = {
                value
                for value in group.loc[verified, "product_name"].dropna().astype(str).str.strip()
                if value
            }
        mapping = verified_mapping.get(product_id, {})
        name = mapping.get("display_name") or (next(iter(names)) if len(names) == 1 else "")
        category = mapping.get("verified_category", "")
        rows.append(
            {
                "product_id": product_id,
                "display_name": name or f"Product ID {product_id}",
                "has_verified_name": bool(name),
                "brand": mapping.get("brand", ""),
                "verified_category": category,
                "category_is_verified": bool(category),
                "review_count": len(group),
            }
        )
    return pd.DataFrame(rows)


def filter_reviews(
    reviews: pd.DataFrame,
    *,
    query: str = "",
    product_id: str = "",
    sentiment: str = "",
    aspect: str = "",
    year: str = "",
    date_precision: str = "",
) -> pd.DataFrame:
    filtered = reviews
    if product_id:
        filtered = filtered.loc[filtered["product_id"].astype(str).eq(product_id)]
    if sentiment:
        filtered = filtered.loc[filtered["predicted_sentiment"].eq(sentiment)]
    if query:
        filtered = filtered.loc[
            filtered["review_text"].str.contains(query, case=False, regex=False, na=False)
        ]
    if aspect:
        filtered = filtered.loc[
            filtered["review_text"].map(lambda text: aspect in detect_aspects(str(text)))
        ]
    if year and "review_date" in filtered:
        filtered = filtered.loc[
            filtered["review_date"].fillna("").astype(str).str.startswith(year)
        ]
    if date_precision and "date_precision" in filtered:
        filtered = filtered.loc[
            filtered["date_precision"].fillna("").astype(str).eq(date_precision)
        ]
    return filtered.copy()


def sentiment_distribution(labels: Iterable[str]) -> dict[str, dict[str, float | int]]:
    values = [str(label).lower() for label in labels]
    unknown = set(values) - {"positive", "negative", "neutral"}
    if unknown:
        raise ValueError(f"Unexpected overall sentiment labels: {unknown}")
    total = len(values)
    return {
        label: {
            "count": values.count(label),
            "percent": (values.count(label) / total * 100) if total else 0.0,
        }
        for label in ("positive", "negative", "neutral")
    }


def validate_review_csv(
    payload: bytes,
    *,
    max_bytes: int = MAX_UPLOAD_BYTES,
) -> pd.DataFrame:
    if not payload:
        raise UploadValidationError("The uploaded CSV is empty.")
    if len(payload) > max_bytes:
        raise UploadValidationError(
            f"CSV exceeds the {max_bytes // (1024 * 1024)} MB upload limit."
        )
    try:
        frame = pd.read_csv(io.BytesIO(payload))
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise UploadValidationError(f"Could not read this CSV: {exc}") from exc
    if "review_text" not in frame.columns:
        raise UploadValidationError(
            "CSV must include a review_text column. Other metadata columns are optional."
        )
    frame["review_text"] = frame["review_text"].fillna("").astype(str)
    frame = frame.loc[frame["review_text"].str.strip().ne("")].copy()
    if frame.empty:
        raise UploadValidationError("No non-empty reviews were found in review_text.")
    if "product_id" not in frame:
        frame["product_id"] = ""
    frame["product_id"] = frame["product_id"].fillna("").astype(str)
    if "review_id" not in frame:
        frame["review_id"] = [f"upload-{index + 1}" for index in range(len(frame))]
    frame = frame.reset_index(drop=True)
    return frame
