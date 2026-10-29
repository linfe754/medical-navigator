import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from pypdf import PdfReader

from app.ingestion.prepare_documents import load_sources, save_json


BENCHMARK_SOURCES = [
    "bulk-billing-en",
    "mental-health-navigation-en",
    "urgent-care-selection-en",
]


class PypdfParser:
    name = "pypdf"

    def parse(self, source: Path, output_dir: Path) -> dict:
        reader = PdfReader(source)

        pages = [
            {
                "page_number": number,
                "text": page.extract_text() or "",
            }
            for number, page in enumerate(reader.pages, start=1)
        ]

        text = "\n\n".join(page["text"] for page in pages)

        save_json(output_dir / "document.json", {"pages": pages})
        (output_dir / "document.txt").write_text(
            text, encoding="utf-8"
        )

        return {
            "page_count": len(pages),
            "text": text,
        }


class DoclingParser:
    name = "docling"

    def __init__(self) -> None:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import (
            DocumentConverter,
            PdfFormatOption,
        )

        options = PdfPipelineOptions()
        options.do_ocr = False
        options.do_table_structure = False

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=options
                )
            }
        )

    def parse(self, source: Path, output_dir: Path) -> dict:
        result = self.converter.convert(source)
        document = result.document
        text = document.export_to_markdown()

        document.save_as_json(output_dir / "document.json")
        (output_dir / "document.md").write_text(
            text, encoding="utf-8"
        )

        return {
            "page_count": len(document.pages),
            "text": text,
            "conversion_status": str(result.status),
        }


def main() -> None:
    sources = {
        source["source_id"]: source
        for source in load_sources()
    }

    for source_id in BENCHMARK_SOURCES:
        if source_id not in sources:
            raise ValueError(f"Source missing from manifest: {source_id}")

        if not sources[source_id]["prepare"]:
            raise ValueError(f"Source is disabled: {source_id}")

    run_id = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    run_dir = Path("data/experiments/benchmarks") / run_id

    report = {
        "schema_version": 1,
        "run_id": run_id,
        "python_version": platform.python_version(),
        "parser_settings": {
            "pypdf": {"version": version("pypdf")},
            "docling": {
                "version": version("docling"),
                "ocr": False,
                "table_structure": False,
            },
        },
        "results": [],
    }

    for parser_class in (PypdfParser, DoclingParser):
        parser_name = parser_class.name
        start = time.perf_counter()

        try:
            parser = parser_class()
        except Exception as error:
            report["results"].append({
                "parser": parser_name,
                "status": "initialization_failed",
                "error": f"{type(error).__name__}: {error}",
            })
            continue

        report["parser_settings"][parser_name][
            "initialization_seconds"
        ] = round(time.perf_counter() - start, 3)

        for source_id in BENCHMARK_SOURCES:
            source = sources[source_id]
            path = Path("data/raw") / source["filename"]
            output_dir = run_dir / source_id / parser_name
            output_dir.mkdir(parents=True, exist_ok=True)

            print(f"{parser_name}: {source_id}", flush=True)
            start = time.perf_counter()

            record = {
                "source_id": source_id,
                "parser": parser_name,
                "status": "failed",
            }

            try:
                record["content_hash"] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()

                result = parser.parse(path, output_dir)
                text = result.pop("text")

                record.update({
                    **result,
                    "status": "completed",
                    "visible_characters": sum(
                        not character.isspace()
                        for character in text
                    ),
                    "replacement_characters": text.count("\ufffd"),
                    "output_directory": str(output_dir),
                    "structural_review": "pending",
                })

            except Exception as error:
                record["error"] = (
                    f"{type(error).__name__}: {error}"
                )

            record["elapsed_seconds"] = round(
                time.perf_counter() - start, 3
            )
            report["results"].append(record)

            # Save progress after every document.
            save_json(run_dir / "report.json", report)

    save_json(run_dir / "report.json", report)

    print("\n--- Benchmark summary ---")

    for result in report["results"]:
        print(
            f"{result['parser']} | "
            f"{result.get('source_id', 'initialization')} | "
            f"{result['status']} | "
            f"{result.get('elapsed_seconds', '-')} seconds"
        )

    print(f"\nReport: {run_dir / 'report.json'}")

    if any(
        result["status"] != "completed"
        for result in report["results"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
