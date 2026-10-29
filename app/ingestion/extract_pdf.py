import argparse
import json
from pathlib import Path

from pypdf import PdfReader


def extract_pdf(source: Path) -> dict:
    reader = PdfReader(source)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        pages.append({
            "page_number": page_number,
            "text": text,
        })

    return {
        "source_filename": source.name,
        "extraction_method": "pypdf",
        "page_count": len(pages),
        "pages": pages,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract a PDF into page-level JSON."
    )
    parser.add_argument("source", type=Path)
    args = parser.parse_args()

    document = extract_pdf(args.source)

    output_dir = Path("data/extracted")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{args.source.stem}.json"
    output_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Source: {document['source_filename']}")
    print(f"Pages: {document['page_count']}")

    for page in document["pages"]:
        print(
            f"Page {page['page_number']}: "
            f"{len(page['text'])} characters"
        )

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
