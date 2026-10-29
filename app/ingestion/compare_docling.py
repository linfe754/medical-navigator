import time
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)


def main() -> None:
    source = Path(
    	"data/raw/when-to-visit-a-medicare-urgent-care-clinic_0.pdf"
	)

    output_dir = Path("data/experiments/docling")
    output_dir.mkdir(parents=True, exist_ok=True)

    # First experiment: native text plus layout analysis.
    # OCR and table reconstruction are not needed for this factsheet.
    options = PdfPipelineOptions()
    options.do_ocr = False
    options.do_table_structure = False

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=options
            )
        }
    )

    start = time.perf_counter()
    result = converter.convert(source)

    markdown = result.document.export_to_markdown()

    markdown_path = output_dir / f"{source.stem}.md"
    json_path = output_dir / f"{source.stem}.json"

    markdown_path.write_text(markdown, encoding="utf-8")
    result.document.save_as_json(json_path)

    print(f"Conversion status: {result.status}")
    print(f"Elapsed: {time.perf_counter() - start:.1f} seconds")
    print(f"Markdown: {markdown_path}")
    print(f"Structured document: {json_path}")
    print("\n--- Markdown output ---\n")
    print(markdown)


if __name__ == "__main__":
    main()
