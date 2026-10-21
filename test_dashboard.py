import unittest
from unittest.mock import patch

import pandas as pd

import api
from inference import (
    UploadValidationError,
    analyze_aspect_evidence,
    clean_text,
    filter_reviews,
    load_sentiment_pipeline,
    predict_overall,
    sentiment_distribution,
    validate_review_csv,
    verified_product_catalog,
)


class FakeSentimentModel:
    def predict(self, texts):
        return [
            "positive" if "great" in text.lower() or "better" in text.lower()
            else "negative" if "drains" in text.lower()
            else "neutral"
            for text in texts
        ]


class DashboardInferenceTests(unittest.TestCase):
    def test_comparison_works_without_verified_categories(self):
        first = pd.Series(
            {
                "product_id": "A",
                "display_name": "Product A",
                "category_is_verified": False,
                "verified_category": "",
            }
        )
        second = pd.Series(
            {
                "product_id": "B",
                "display_name": "Product B",
                "category_is_verified": False,
                "verified_category": "",
            }
        )
        first_reviews = pd.DataFrame({"predicted_sentiment": ["positive"]})
        second_reviews = pd.DataFrame({"predicted_sentiment": ["negative"]})
        with patch(
            "api.get_product",
            side_effect=[
                (first, first_reviews, pd.DataFrame()),
                (second, second_reviews, pd.DataFrame()),
            ],
        ):
            result = api.compare("A", "B")

        self.assertIsNone(result["category"])
        self.assertEqual(result["comparison_scope"], "Category not verified")
        self.assertEqual(
            [item["product"]["product_id"] for item in result["products"]],
            ["A", "B"],
        )

    def test_saved_overall_pipeline_loads_and_predicts(self):
        model = load_sentiment_pipeline()
        labels = predict_overall(
            ["The sound is excellent.", "The device does not work."],
            model,
        )
        self.assertEqual(len(labels), 2)
        self.assertTrue(set(labels).issubset({"positive", "negative", "neutral"}))

    def test_cleaning_preserves_negation_and_removes_url(self):
        self.assertEqual(
            clean_text("DO NOT buy https://example.test/review  now"),
            "do not buy now",
        )

    def test_sentiment_percentages_include_counts_and_empty_case(self):
        summary = sentiment_distribution(["positive", "negative", "positive"])
        self.assertEqual(summary["positive"]["count"], 2)
        self.assertAlmostEqual(summary["positive"]["percent"], 200 / 3)
        self.assertEqual(summary["neutral"], {"count": 0, "percent": 0})
        self.assertTrue(all(item["percent"] == 0 for item in sentiment_distribution([]).values()))

    def test_filtering_combines_text_product_sentiment_aspect_and_year(self):
        frame = pd.DataFrame(
            [
                {
                    "review_text": "Battery is great",
                    "product_id": "A",
                    "predicted_sentiment": "positive",
                    "review_date": "2024-04-01",
                    "date_precision": "day",
                },
                {
                    "review_text": "Battery drains",
                    "product_id": "B",
                    "predicted_sentiment": "negative",
                    "review_date": "2023",
                    "date_precision": "year_only",
                },
            ]
        )
        result = filter_reviews(
            frame,
            query="great",
            product_id="A",
            sentiment="positive",
            aspect="battery",
            year="2024",
            date_precision="day",
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["product_id"], "A")

    def test_catalog_uses_ids_for_unverified_names_and_hints(self):
        frame = pd.DataFrame(
            [
                {
                    "product_id": "X",
                    "product_name": "Invented Name",
                    "category_hint": "phones",
                    "category_method": "keyword_mention_not_verified_product_category",
                },
                {
                    "product_id": "Y",
                    "product_name": "Source Name",
                    "category_hint": "audio",
                    "category_method": "source_product_metadata",
                },
            ]
        )
        catalog = verified_product_catalog(frame).set_index("product_id")
        self.assertEqual(catalog.loc["X", "display_name"], "Product ID X")
        self.assertFalse(catalog.loc["X", "category_is_verified"])
        self.assertEqual(catalog.loc["Y", "display_name"], "Source Name")
        self.assertEqual(catalog.loc["Y", "verified_category"], "")
        self.assertFalse(catalog.loc["Y", "category_is_verified"])

    def test_metadata_mapping_exposes_only_verified_fields(self):
        reviews = pd.DataFrame(
            [{"product_id": "X"}, {"product_id": "Y"}]
        )
        mapping = pd.DataFrame(
            [
                {
                    "product_id": "X",
                    "product_name": "Verified Earbuds",
                    "brand": "Example",
                    "category": "audio",
                    "metadata_source": "manufacturer catalogue",
                    "verification_status": "verified",
                },
                {
                    "product_id": "Y",
                    "product_name": "Unverified Phone",
                    "brand": "Unverified",
                    "category": "phones",
                    "metadata_source": "",
                    "verification_status": "unverified",
                },
            ]
        )
        catalog = verified_product_catalog(reviews, mapping).set_index("product_id")
        self.assertEqual(catalog.loc["X", "brand"], "Example")
        self.assertEqual(catalog.loc["X", "verified_category"], "audio")
        self.assertEqual(catalog.loc["Y", "display_name"], "Product ID Y")
        self.assertEqual(catalog.loc["Y", "brand"], "")

    def test_csv_validation_requires_text_and_rejects_oversized_upload(self):
        frame = validate_review_csv(b"review_text\nBattery lasts well\n")
        self.assertEqual(frame.iloc[0]["review_text"], "Battery lasts well")
        self.assertEqual(frame.iloc[0]["product_id"], "")
        with self.assertRaises(UploadValidationError):
            validate_review_csv(b"rating\n5\n")
        with self.assertRaises(UploadValidationError):
            validate_review_csv(b"review_text\nok\n", max_bytes=4)

    def test_aspect_evidence_handles_mixed_opinions_and_no_opinion(self):
        reviews = [
            {
                "review_id": "mixed",
                "review_text": "The battery drains quickly, but sound quality is great.",
                "source_url": "https://example.test/review",
            },
            {
                "review_id": "fact",
                "review_text": "Before connecting the WiFi extender.",
            },
            {
                "review_id": "comparison",
                "review_text": "Compared to my old phone, the battery is better.",
            },
        ]
        evidence = analyze_aspect_evidence(reviews, FakeSentimentModel())
        mixed = {
            (item["aspect"], item["sentiment"])
            for item in evidence
            if item["review_id"] == "mixed"
        }
        self.assertIn(("battery", "negative"), mixed)
        self.assertIn(("sound", "positive"), mixed)
        self.assertTrue(
            any(
                item["review_id"] == "fact"
                and item["sentiment"] == "no_opinion"
                for item in evidence
            )
        )
        self.assertTrue(
            any(
                item["review_id"] == "comparison"
                and item["sentiment"] == "uncertain"
                for item in evidence
            )
        )
        self.assertEqual(
            next(item for item in evidence if item["review_id"] == "mixed" and item["aspect"] == "battery")["review_text"],
            "The battery drains quickly",
        )


if __name__ == "__main__":
    unittest.main()
