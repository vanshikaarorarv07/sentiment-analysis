from __future__ import annotations

import json
from functools import lru_cache

import pandas as pd
import requests
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from inference import (
    ABSA_MODEL_ID,
    DATA_PATH,
    analyze_aspect_evidence,
    detect_aspects,
    filter_reviews,
    load_absa_model,
    load_product_metadata,
    load_reviews,
    load_sentiment_pipeline,
    predict_overall,
    sentiment_distribution,
    validate_review_csv,
    verified_product_catalog,
)

app = FastAPI(title="Consumer Tech Review Intelligence API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_overall_model():
    return load_sentiment_pipeline()


@lru_cache(maxsize=1)
def get_absa_model():
    return load_absa_model()


@lru_cache(maxsize=1)
def get_source_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    reviews = load_reviews(DATA_PATH)
    reviews["predicted_sentiment"] = predict_overall(
        reviews["review_text"].tolist(),
        get_overall_model(),
    )
    metadata = load_product_metadata()
    return reviews, verified_product_catalog(reviews, metadata)


def frame_records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records", force_ascii=False))


def get_product(product_id: str) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    reviews, catalog = get_source_data()
    match = catalog.loc[catalog["product_id"].astype(str).eq(product_id)]
    if match.empty:
        raise HTTPException(status_code=404, detail="No reviews available for this product ID.")
    product = match.iloc[0]
    product_reviews = reviews.loc[reviews["product_id"].astype(str).eq(product_id)].copy()
    return product, product_reviews, catalog


def distribution_for(frame: pd.DataFrame) -> dict:
    return sentiment_distribution(frame["predicted_sentiment"].tolist())


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/overview")
def overview() -> dict:
    try:
        reviews, catalog = get_source_data()
    except (FileNotFoundError, OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=f"Could not load review data or model: {exc}") from exc

    sources = (
        reviews["platform"].fillna("Unspecified").astype(str).value_counts().to_dict()
        if "platform" in reviews
        else {}
    )
    dates = (
        reviews["review_date"].dropna().astype(str)
        if "review_date" in reviews
        else pd.Series(dtype=str)
    )
    source_links = []
    if {"platform", "source_dataset"}.issubset(reviews.columns):
        source_links = frame_records(
            reviews[["platform", "source_dataset"]].dropna().drop_duplicates()
        )
    return {
        "review_count": len(reviews),
        "product_id_count": int(catalog["product_id"].nunique()),
        "named_product_count": int(catalog["has_verified_name"].sum()),
        "source_count": len(sources),
        "sources": sources,
        "source_links": source_links,
        "date_start": dates.min() if not dates.empty else None,
        "date_end": dates.max() if not dates.empty else None,
        "sentiment": distribution_for(reviews),
        "date_precision_counts": (
            reviews["date_precision"].fillna("unavailable").astype(str).value_counts().to_dict()
            if "date_precision" in reviews
            else {}
        ),
        "sample_notice": (
            "These are predictions on a sampled review dataset, not a representative "
            "market survey. Source sentiment labels were derived from star ratings."
        ),
        "coverage_notice": (
            "Amazon and Amazon UK only; no live social feeds or 2025–2026 data."
        ),
    }


@app.get("/api/products")
def products(
    query: str = "",
    brand: str = "",
    category: str = "",
    limit: int = Query(default=100, ge=1, le=10000),
) -> dict:
    try:
        _, catalog = get_source_data()
    except (FileNotFoundError, OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=f"Could not load product catalogue: {exc}") from exc
    matches = catalog
    if query.strip():
        term = query.strip()
        matches = matches.loc[
            matches["display_name"].str.contains(term, case=False, regex=False)
            | matches["brand"].fillna("").str.contains(term, case=False, regex=False)
            | matches["product_id"].str.contains(term, case=False, regex=False)
        ]
    if brand:
        matches = matches.loc[matches["brand"].eq(brand)]
    if category:
        matches = matches.loc[matches["verified_category"].eq(category)]
    matches = matches.sort_values(
        ["review_count", "display_name"], ascending=[False, True]
    ).head(limit)
    return {
        "items": frame_records(matches),
        "verified_brands": sorted(
            value for value in catalog["brand"].dropna().astype(str).unique() if value
        ),
        "verified_categories": sorted(
            value
            for value in catalog.loc[
                catalog["category_is_verified"], "verified_category"
            ].dropna().astype(str).unique()
            if value
        ),
    }


@app.get("/api/products/{product_id}")
def product_detail(product_id: str) -> dict:
    try:
        product, reviews, _ = get_product(product_id)
    except (FileNotFoundError, OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=f"Could not load product reviews: {exc}") from exc
    source_counts = (
        reviews["platform"].fillna("Unspecified").astype(str).value_counts().to_dict()
        if "platform" in reviews
        else {}
    )
    dates = (
        reviews["review_date"].dropna().astype(str)
        if "review_date" in reviews
        else pd.Series(dtype=str)
    )
    return {
        "product": json.loads(product.to_json()),
        "review_count": len(reviews),
        "sources": source_counts,
        "date_start": dates.min() if not dates.empty else None,
        "date_end": dates.max() if not dates.empty else None,
        "sentiment": distribution_for(reviews),
        "reviews": frame_records(reviews),
    }


@app.post("/api/products/{product_id}/aspects")
def product_aspects(product_id: str) -> dict:
    try:
        product, reviews, _ = get_product(product_id)
    except (FileNotFoundError, OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=f"Could not load product reviews: {exc}") from exc

    model = tokenizer = None
    warning = None
    try:
        model, tokenizer = get_absa_model()
    except (
        ImportError,
        OSError,
        RuntimeError,
        ValueError,
        ConnectionError,
        TimeoutError,
        requests.exceptions.RequestException,
    ) as exc:
        warning = (
            f"The pretrained aspect model is unavailable ({exc}). "
            "Sentence-level overall sentiment is used as a weaker fallback."
        )

    columns = [
        column for column in ("review_id", "review_text", "source_url")
        if column in reviews
    ]
    evidence = analyze_aspect_evidence(
        reviews[columns].fillna("").to_dict("records"),
        get_overall_model(),
        model,
        tokenizer,
    )
    return {
        "product_id": str(product["product_id"]),
        "analyzed_review_count": len(reviews),
        "model": ABSA_MODEL_ID if model is not None else "sentence-level fallback",
        "warning": warning,
        "evidence": evidence,
    }


@app.get("/api/compare")
def compare(product_a: str, product_b: str) -> dict:
    try:
        first, first_reviews, _ = get_product(product_a)
        second, second_reviews, _ = get_product(product_b)
    except (FileNotFoundError, OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=f"Could not load comparison data: {exc}") from exc
    same_verified_category = (
        first["category_is_verified"]
        and second["category_is_verified"]
        and first["verified_category"] == second["verified_category"]
    )
    comparison_scope = (
        "Different verified categories"
        if first["category_is_verified"] and second["category_is_verified"]
        else "Category not verified"
    )
    return {
        "category": first["verified_category"] if same_verified_category else None,
        "comparison_scope": "Same verified category" if same_verified_category else comparison_scope,
        "products": [
            {
                "product": json.loads(first.to_json()),
                "review_count": len(first_reviews),
                "sentiment": distribution_for(first_reviews),
            },
            {
                "product": json.loads(second.to_json()),
                "review_count": len(second_reviews),
                "sentiment": distribution_for(second_reviews),
            },
        ],
    }


@app.get("/api/reviews")
def review_search(
    query: str = "",
    product_id: str = "",
    sentiment: str = "",
    aspect: str = "",
    year: str = "",
    date_precision: str = "",
    limit: int = Query(default=1000, ge=1, le=9000),
) -> dict:
    try:
        reviews, _ = get_source_data()
    except (FileNotFoundError, OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=f"Could not load reviews: {exc}") from exc
    filtered = filter_reviews(
        reviews,
        query=query,
        product_id=product_id,
        sentiment=sentiment,
        aspect=aspect,
        year=year,
        date_precision=date_precision,
    )
    return {
        "total": len(filtered),
        "items": frame_records(filtered.head(limit)),
        "limit": limit,
    }


@app.post("/api/upload")
def analyze_upload(file: UploadFile = File(...)) -> dict:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=415, detail="Upload a CSV file.")
    try:
        frame = validate_review_csv(file.file.read())
        frame["predicted_sentiment"] = predict_overall(
            frame["review_text"].tolist(),
            get_overall_model(),
        )
    except (ValueError, OSError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if "sentiment" in frame.columns:
        frame = frame.rename(columns={"sentiment": "source_rating_sentiment_proxy"})
    return {
        "review_count": len(frame),
        "sentiment": distribution_for(frame),
        "items": frame_records(frame),
        "notice": "Predictions describe only this uploaded file; rating and sentiment columns were not model inputs.",
    }
