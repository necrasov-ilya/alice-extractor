from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from alice_extractor.dataset_validation import export_huggingface_jsonl, validate_dataset


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPOSITORY_ROOT / "benchmarks" / "data" / "v1"
SCHEMA_PATH = REPOSITORY_ROOT / "schemas" / "invoice-v1.schema.json"


class DatasetValidationTests(unittest.TestCase):
    def test_v1_release_passes_validation(self) -> None:
        report = validate_dataset(DATASET_ROOT, SCHEMA_PATH)

        self.assertTrue(report.ok, report.errors)
        self.assertEqual(report.record_count, 50)
        self.assertEqual(
            report.split_counts,
            {"development": 30, "test": 10, "validation": 10},
        )
        self.assertEqual(
            [warning.code for warning in report.warnings],
            ["test_only_currencies"],
        )

    def test_release_checksums_cover_all_records(self) -> None:
        checksums = (DATASET_ROOT / "checksums.sha256").read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(checksums), 101)
        self.assertTrue(any(line.endswith("  manifest.jsonl") for line in checksums))

    def test_huggingface_export_preserves_splits_and_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "data"
            counts = export_huggingface_jsonl(DATASET_ROOT, output_dir)

            self.assertEqual(counts, {"development": 30, "validation": 10, "test": 10})
            development_rows = [
                json.loads(line)
                for line in (output_dir / "development.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(development_rows[0]["id"], "invoice-001")
            self.assertEqual(development_rows[0]["expected"]["document_type"], "invoice")
            self.assertIn("СЧЁТ НА ОПЛАТУ", development_rows[0]["text"])

    def test_checksum_tampering_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            copied_dataset = Path(temporary_directory) / "v1"
            shutil.copytree(DATASET_ROOT, copied_dataset)
            target = copied_dataset / "test" / "invoice-050.txt"
            target.write_text(target.read_text(encoding="utf-8") + "\n", encoding="utf-8")

            report = validate_dataset(copied_dataset, SCHEMA_PATH)

            self.assertFalse(report.ok)
            self.assertIn("checksum_mismatch", {issue.code for issue in report.errors})


if __name__ == "__main__":
    unittest.main()
