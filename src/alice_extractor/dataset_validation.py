"""Validation and release helpers for the invoice benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import SchemaError


EXPECTED_SPLITS = {"development": 30, "validation": 10, "test": 10}
MANIFEST_FIELDS = {
    "id",
    "split",
    "source_file",
    "expected_file",
    "currency",
    "vat_amount",
    "item_count",
    "template_family",
    "vat_mode",
}
MONEY_QUANTUM = Decimal("0.01")
RUSSIAN_MONTHS = (
    "января",
    "февраля",
    "марта",
    "апреля",
    "мая",
    "июня",
    "июля",
    "августа",
    "сентября",
    "октября",
    "ноября",
    "декабря",
)


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    record_id: str | None = None
    path: str | None = None


@dataclass
class ValidationReport:
    dataset_root: str
    record_count: int = 0
    split_counts: dict[str, int] = field(default_factory=dict)
    currency_counts: dict[str, int] = field(default_factory=dict)
    vat_mode_counts: dict[str, int] = field(default_factory=dict)
    template_counts: dict[str, int] = field(default_factory=dict)
    item_count_counts: dict[str, int] = field(default_factory=dict)
    valid_tax_id_checksums: int = 0
    invalid_tax_id_checksums: int = 0
    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["ok"] = self.ok
        return result


def _issue(
    target: list[ValidationIssue],
    code: str,
    message: str,
    *,
    record_id: str | None = None,
    path: Path | None = None,
) -> None:
    target.append(
        ValidationIssue(
            code=code,
            message=message,
            record_id=record_id,
            path=str(path) if path is not None else None,
        )
    )


def _normalized_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"\s+", " ", value).strip()


def _digits(value: str | None) -> str:
    return re.sub(r"\D", "", value or "")


def _money(value: Decimal | int | float | str) -> Decimal:
    return Decimal(str(value)).quantize(MONEY_QUANTUM, rounding=ROUND_HALF_UP)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_inn_checksum(value: str) -> bool:
    if not value.isdigit():
        return False
    if len(value) == 10:
        coefficients = (2, 4, 10, 3, 5, 9, 4, 6, 8)
        check = sum(int(value[index]) * coefficient for index, coefficient in enumerate(coefficients)) % 11 % 10
        return check == int(value[9])
    if len(value) == 12:
        first = (7, 2, 4, 10, 3, 5, 9, 4, 6, 8)
        second = (3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8)
        first_check = sum(int(value[index]) * coefficient for index, coefficient in enumerate(first)) % 11 % 10
        second_check = sum(int(value[index]) * coefficient for index, coefficient in enumerate(second)) % 11 % 10
        return first_check == int(value[10]) and second_check == int(value[11])
    return False


def _safe_manifest_path(dataset_root: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        return None
    path = dataset_root.joinpath(*relative.parts)
    try:
        path.resolve().relative_to(dataset_root.resolve())
    except ValueError:
        return None
    return path


def _date_in_text(iso_date: str, text: str) -> bool:
    try:
        year, month, day = (int(part) for part in iso_date.split("-"))
        month_name = RUSSIAN_MONTHS[month - 1]
    except (AttributeError, IndexError, ValueError):
        return False
    variants = {
        iso_date,
        f"{day:02d}.{month:02d}.{year}",
        f"{day}.{month}.{year}",
        f"{day:02d}/{month:02d}/{year}",
        f"{day}/{month}/{year}",
        f"{year}/{month:02d}/{day:02d}",
        f"{year}/{month}/{day}",
        f"{day:02d}-{month:02d}-{year}",
        f"{day}-{month}-{year}",
        f"{day} {month_name} {year}",
        f"{day:02d} {month_name} {year}",
    }
    normalized = _normalized_text(text)
    return any(_normalized_text(variant) in normalized for variant in variants)


def _load_manifest(path: Path, report: ValidationReport) -> list[dict[str, Any]]:
    if not path.is_file():
        _issue(report.errors, "missing_manifest", "Manifest file is missing", path=path)
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            _issue(report.errors, "invalid_manifest_json", f"Line {line_number}: {exc.msg}", path=path)
            continue
        if not isinstance(record, dict):
            _issue(report.errors, "invalid_manifest_record", f"Line {line_number} is not an object", path=path)
            continue
        records.append(record)
    return records


def _validate_checksums(dataset_root: Path, report: ValidationReport) -> None:
    checksum_path = dataset_root / "checksums.sha256"
    if not checksum_path.exists():
        _issue(
            report.warnings,
            "missing_checksums",
            "checksums.sha256 is absent, so the fixed release cannot be verified byte-for-byte",
            path=checksum_path,
        )
        return
    checked: set[Path] = set()
    for line_number, line in enumerate(checksum_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            _issue(
                report.errors,
                "invalid_checksum_line",
                f"Line {line_number} has an invalid format",
                path=checksum_path,
            )
            continue
        expected_digest, relative_name = match.groups()
        target = _safe_manifest_path(dataset_root, relative_name)
        if target is None or not target.is_file():
            _issue(
                report.errors,
                "missing_checksummed_file",
                f"Checksummed file is missing or unsafe: {relative_name}",
                path=checksum_path,
            )
            continue
        checked.add(target.resolve())
        if _sha256(target) != expected_digest:
            _issue(report.errors, "checksum_mismatch", f"SHA-256 mismatch for {relative_name}", path=target)
    required = {
        path.resolve()
        for pattern in ("manifest.jsonl", "*/*.txt", "*/*.expected.json")
        for path in dataset_root.glob(pattern)
    }
    for path in sorted(required - checked):
        _issue(report.errors, "unchecked_release_file", "Release file is absent from checksums.sha256", path=path)


def validate_dataset(dataset_root: Path, schema_path: Path) -> ValidationReport:
    dataset_root = dataset_root.resolve()
    schema_path = schema_path.resolve()
    report = ValidationReport(dataset_root=str(dataset_root))
    records = _load_manifest(dataset_root / "manifest.jsonl", report)
    report.record_count = len(records)
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
    except (OSError, json.JSONDecodeError, SchemaError) as exc:
        _issue(report.errors, "invalid_schema", str(exc), path=schema_path)
        return report
    validator = Draft202012Validator(schema, format_checker=FormatChecker())

    ids: set[str] = set()
    source_files: set[Path] = set()
    expected_files: set[Path] = set()
    source_hashes: defaultdict[str, list[str]] = defaultdict(list)
    normalized_source_hashes: defaultdict[str, list[str]] = defaultdict(list)
    expected_hashes: defaultdict[str, list[str]] = defaultdict(list)
    tax_to_names: defaultdict[str, set[str]] = defaultdict(set)
    name_to_tax_ids: defaultdict[str, set[str]] = defaultdict(set)
    tax_id_splits: defaultdict[str, set[str]] = defaultdict(set)

    for row_number, record in enumerate(records, start=1):
        record_id = record.get("id") if isinstance(record.get("id"), str) else None
        missing_fields = sorted(MANIFEST_FIELDS - record.keys())
        extra_fields = sorted(record.keys() - MANIFEST_FIELDS)
        if missing_fields:
            _issue(report.errors, "manifest_fields_missing", f"Missing fields: {', '.join(missing_fields)}", record_id=record_id)
        if extra_fields:
            _issue(report.errors, "manifest_fields_extra", f"Unexpected fields: {', '.join(extra_fields)}", record_id=record_id)
        if record_id is None:
            _issue(report.errors, "invalid_record_id", f"Manifest row {row_number} has no string id")
            continue
        if record_id in ids:
            _issue(report.errors, "duplicate_record_id", "Record id is duplicated", record_id=record_id)
        ids.add(record_id)

        split = record.get("split")
        if split not in EXPECTED_SPLITS:
            _issue(report.errors, "invalid_split", f"Unknown split: {split!r}", record_id=record_id)
        source_path = _safe_manifest_path(dataset_root, record.get("source_file"))
        expected_path = _safe_manifest_path(dataset_root, record.get("expected_file"))
        if source_path is None:
            _issue(report.errors, "unsafe_source_path", "Source path is invalid", record_id=record_id)
            continue
        if expected_path is None:
            _issue(report.errors, "unsafe_expected_path", "Expected path is invalid", record_id=record_id)
            continue
        source_files.add(source_path.resolve())
        expected_files.add(expected_path.resolve())
        if not source_path.is_file():
            _issue(report.errors, "missing_source_file", "Source file is missing", record_id=record_id, path=source_path)
            continue
        if not expected_path.is_file():
            _issue(report.errors, "missing_expected_file", "Expected JSON is missing", record_id=record_id, path=expected_path)
            continue
        if source_path.stem != record_id or expected_path.name != f"{record_id}.expected.json":
            _issue(report.errors, "record_file_mismatch", "Record id and filenames do not match", record_id=record_id)
        if source_path.parent.name != split or expected_path.parent.name != split:
            _issue(report.errors, "record_split_mismatch", "Manifest split and directory do not match", record_id=record_id)

        try:
            source_text = source_path.read_text(encoding="utf-8")
            expected_content = expected_path.read_text(encoding="utf-8")
            expected_native = json.loads(expected_content)
            expected_decimal = json.loads(expected_content, parse_float=Decimal, parse_int=Decimal)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            _issue(report.errors, "invalid_record_file", str(exc), record_id=record_id)
            continue
        for error in sorted(validator.iter_errors(expected_native), key=lambda item: list(item.path)):
            location = ".".join(str(part) for part in error.path) or "<root>"
            _issue(
                report.errors,
                "schema_validation_error",
                f"{location}: {error.message}",
                record_id=record_id,
                path=expected_path,
            )

        if record.get("currency") != expected_native.get("currency"):
            _issue(report.errors, "manifest_currency_mismatch", "Manifest currency differs from expected JSON", record_id=record_id)
        if record.get("vat_amount") != expected_native.get("vat_amount"):
            _issue(report.errors, "manifest_vat_mismatch", "Manifest VAT differs from expected JSON", record_id=record_id)
        if record.get("item_count") != len(expected_native.get("line_items", [])):
            _issue(report.errors, "manifest_item_count_mismatch", "Manifest item count differs from expected JSON", record_id=record_id)

        try:
            line_total = sum(
                (item["line_total"] for item in expected_decimal["line_items"] if item["line_total"] is not None),
                Decimal(0),
            )
            if _money(line_total) != _money(expected_decimal["subtotal"]):
                _issue(report.errors, "subtotal_mismatch", "Subtotal does not equal the sum of line totals", record_id=record_id)
            if _money(expected_decimal["subtotal"] + expected_decimal["vat_amount"]) != _money(expected_decimal["total_amount"]):
                _issue(report.errors, "total_mismatch", "Total does not equal subtotal plus VAT", record_id=record_id)
            for item_index, item in enumerate(expected_decimal["line_items"], start=1):
                if None not in (item["quantity"], item["unit_price"], item["line_total"]):
                    calculated = _money(item["quantity"] * item["unit_price"])
                    if calculated != _money(item["line_total"]):
                        _issue(
                            report.errors,
                            "line_item_total_mismatch",
                            f"Line item {item_index} total does not equal quantity times unit price",
                            record_id=record_id,
                        )
        except (InvalidOperation, KeyError, TypeError):
            pass

        normalized_source = _normalized_text(source_text)
        for label, value in (
            ("invoice number", expected_native.get("invoice_number")),
            ("supplier name", expected_native.get("supplier", {}).get("name")),
            ("buyer name", expected_native.get("buyer", {}).get("name")),
        ):
            if value is not None and _normalized_text(value) not in normalized_source:
                _issue(report.errors, "value_absent_from_source", f"{label} is absent from source text", record_id=record_id)
        invoice_date = expected_native.get("invoice_date")
        if invoice_date is not None and not _date_in_text(invoice_date, source_text):
            _issue(report.errors, "date_absent_from_source", "Invoice date is absent from source text", record_id=record_id)
        source_digits = _digits(source_text)
        for party_name in ("supplier", "buyer"):
            party = expected_native.get(party_name, {})
            tax_id = party.get("tax_id")
            name = party.get("name")
            if tax_id is not None:
                if tax_id not in source_digits:
                    _issue(report.errors, "tax_id_absent_from_source", f"{party_name} tax id is absent from source text", record_id=record_id)
                if _valid_inn_checksum(tax_id):
                    report.valid_tax_id_checksums += 1
                else:
                    report.invalid_tax_id_checksums += 1
            if tax_id is not None and name is not None:
                tax_to_names[tax_id].add(name)
                name_to_tax_ids[_normalized_text(name)].add(tax_id)
                tax_id_splits[tax_id].add(str(split))
        for item_index, item in enumerate(expected_native.get("line_items", []), start=1):
            description = item.get("description")
            if description and _normalized_text(description) not in normalized_source:
                _issue(
                    report.errors,
                    "line_description_absent_from_source",
                    f"Line item {item_index} description is absent from source text",
                    record_id=record_id,
                )

        source_hashes[_sha256(source_path)].append(record_id)
        normalized_source_hashes[hashlib.sha256(normalized_source.encode("utf-8")).hexdigest()].append(record_id)
        expected_hashes[_sha256(expected_path)].append(record_id)

    split_counts = Counter(record.get("split") for record in records if isinstance(record.get("split"), str))
    report.split_counts = dict(sorted(split_counts.items()))
    report.currency_counts = dict(sorted(Counter(record.get("currency") for record in records).items()))
    report.vat_mode_counts = dict(sorted(Counter(record.get("vat_mode") for record in records).items()))
    report.template_counts = dict(sorted(Counter(record.get("template_family") for record in records).items()))
    report.item_count_counts = {
        str(key): value for key, value in sorted(Counter(record.get("item_count") for record in records).items())
    }
    for split, expected_count in EXPECTED_SPLITS.items():
        if split_counts.get(split, 0) != expected_count:
            _issue(
                report.errors,
                "split_count_mismatch",
                f"{split} contains {split_counts.get(split, 0)} records, expected {expected_count}",
            )

    actual_source_files = {path.resolve() for path in dataset_root.glob("*/*.txt")}
    actual_expected_files = {path.resolve() for path in dataset_root.glob("*/*.expected.json")}
    for path in sorted(actual_source_files - source_files):
        _issue(report.errors, "unlisted_source_file", "Source file is absent from manifest", path=path)
    for path in sorted(actual_expected_files - expected_files):
        _issue(report.errors, "unlisted_expected_file", "Expected file is absent from manifest", path=path)

    for code, groups in (
        ("duplicate_source", source_hashes),
        ("duplicate_normalized_source", normalized_source_hashes),
        ("duplicate_expected_json", expected_hashes),
    ):
        for record_ids in groups.values():
            if len(record_ids) > 1:
                _issue(report.errors, code, f"Duplicate records: {', '.join(sorted(record_ids))}")
    for tax_id, names in tax_to_names.items():
        if len(names) > 1:
            _issue(report.errors, "tax_id_identity_conflict", f"Tax id {tax_id} maps to multiple organization names")
    for name, tax_ids in name_to_tax_ids.items():
        if len(tax_ids) > 1:
            _issue(report.errors, "organization_identity_conflict", f"Organization {name!r} maps to multiple tax ids")
    for tax_id, splits in tax_id_splits.items():
        if len(splits) > 1:
            _issue(
                report.warnings,
                "tax_id_cross_split_overlap",
                f"Tax id {tax_id} occurs across splits: {', '.join(sorted(splits))}",
            )

    development_currencies = {record.get("currency") for record in records if record.get("split") == "development"}
    test_only_currencies = sorted(
        {record.get("currency") for record in records if record.get("split") == "test"} - development_currencies
    )
    if test_only_currencies:
        _issue(
            report.warnings,
            "test_only_currencies",
            "Currencies found only in test: " + ", ".join(str(value) for value in test_only_currencies),
        )

    _validate_checksums(dataset_root, report)
    return report


def write_checksums(dataset_root: Path) -> Path:
    dataset_root = dataset_root.resolve()
    files = [dataset_root / "manifest.jsonl"]
    files.extend(sorted(dataset_root.glob("*/*.txt")))
    files.extend(sorted(dataset_root.glob("*/*.expected.json")))
    output_path = dataset_root / "checksums.sha256"
    lines = [f"{_sha256(path)}  {path.relative_to(dataset_root).as_posix()}" for path in files]
    temporary_path = output_path.with_suffix(".sha256.tmp")
    temporary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary_path.replace(output_path)
    return output_path


def export_huggingface_jsonl(dataset_root: Path, output_dir: Path) -> dict[str, int]:
    dataset_root = dataset_root.resolve()
    output_dir = output_dir.resolve()
    records = [
        json.loads(line)
        for line in (dataset_root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    counts: dict[str, int] = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in EXPECTED_SPLITS:
        split_records = [record for record in records if record["split"] == split]
        target_path = output_dir / f"{split}.jsonl"
        temporary_path = target_path.with_suffix(".jsonl.tmp")
        with temporary_path.open("w", encoding="utf-8") as stream:
            for record in split_records:
                source_path = dataset_root / record["source_file"]
                expected_path = dataset_root / record["expected_file"]
                row = {
                    "id": record["id"],
                    "text": source_path.read_text(encoding="utf-8"),
                    "expected": json.loads(expected_path.read_text(encoding="utf-8")),
                    "currency": record["currency"],
                    "vat_amount": record["vat_amount"],
                    "item_count": record["item_count"],
                    "template_family": record["template_family"],
                    "vat_mode": record["vat_mode"],
                }
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        temporary_path.replace(target_path)
        counts[split] = len(split_records)
    return counts


def _default_repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    repository_root = _default_repository_root()
    parser = argparse.ArgumentParser(description="Validate and optionally export the invoice benchmark")
    parser.add_argument("--dataset-root", type=Path, default=repository_root / "benchmarks" / "data" / "v1")
    parser.add_argument("--schema", type=Path, default=repository_root / "schemas" / "invoice-v1.schema.json")
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--write-checksums", action="store_true")
    parser.add_argument("--export-hf-dir", type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.write_checksums:
        write_checksums(args.dataset_root)
    report = validate_dataset(args.dataset_root, args.schema)
    if args.export_hf_dir is not None:
        if not report.ok:
            print("Dataset export refused because validation failed", file=sys.stderr)
            return 1
        export_huggingface_jsonl(args.dataset_root, args.export_hf_dir)
    if args.as_json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        status = "OK" if report.ok else "FAILED"
        print(f"{status}: {report.record_count} records, {len(report.errors)} errors, {len(report.warnings)} warnings")
        for issue in (*report.errors, *report.warnings):
            location = f" [{issue.record_id}]" if issue.record_id else ""
            print(f"- {issue.code}{location}: {issue.message}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
