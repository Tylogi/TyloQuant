"""Bounded server-side extraction for user-provided documents."""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

MAX_EXTRACTED_CHARACTERS = 2_000_000


class DocumentExtractionError(ValueError):
    pass


@dataclass(frozen=True)
class ExtractedDocument:
    text: str
    page_count: int | None
    extractor: str


def extract_document(path: Path, mime_type: str, name: str) -> ExtractedDocument:
    suffix = Path(name).suffix.lower()
    if mime_type == "application/pdf" or suffix == ".pdf":
        return _extract_pdf(path)
    if (
        mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        or suffix == ".docx"
    ):
        return _extract_docx(path)
    if (
        mime_type.startswith("text/")
        or suffix
        in {
            ".csv",
            ".css",
            ".h",
            ".hpp",
            ".html",
            ".ini",
            ".java",
            ".js",
            ".json",
            ".jsonl",
            ".jsx",
            ".log",
            ".md",
            ".markdown",
            ".py",
            ".rs",
            ".sh",
            ".toml",
            ".ts",
            ".tsx",
            ".tsv",
            ".xml",
            ".yaml",
            ".yml",
        }
        or mime_type
        in {
            "application/json",
            "application/xml",
            "application/yaml",
        }
    ):
        return _extract_text(path)
    raise DocumentExtractionError(f"unsupported document type: {mime_type or suffix}")


def _bounded(text: str) -> str:
    normalized = text.replace("\x00", "").strip()
    if not normalized:
        raise DocumentExtractionError("document contains no extractable text")
    if len(normalized) > MAX_EXTRACTED_CHARACTERS:
        raise DocumentExtractionError(
            f"extracted text exceeds {MAX_EXTRACTED_CHARACTERS} characters"
        )
    return normalized


def _extract_text(path: Path) -> ExtractedDocument:
    data = path.read_bytes()
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        text = data.decode("utf-16")
    else:
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError as error:
            raise DocumentExtractionError("text document is not UTF-8 or UTF-16") from error
    return ExtractedDocument(_bounded(text), None, "plain-text-v1")


def _extract_docx(path: Path) -> ExtractedDocument:
    try:
        with zipfile.ZipFile(path) as archive:
            document = archive.read("word/document.xml")
    except (KeyError, OSError, zipfile.BadZipFile) as error:
        raise DocumentExtractionError("invalid DOCX document") from error
    try:
        root = ElementTree.fromstring(document)
    except ElementTree.ParseError as error:
        raise DocumentExtractionError("invalid DOCX XML") from error
    namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    paragraphs: list[str] = []
    for paragraph in root.iter(f"{namespace}p"):
        fragments: list[str] = []
        for node in paragraph.iter():
            if node.tag == f"{namespace}t" and node.text:
                fragments.append(node.text)
            elif node.tag == f"{namespace}tab":
                fragments.append("\t")
            elif node.tag == f"{namespace}br":
                fragments.append("\n")
        line = "".join(fragments).strip()
        if line:
            paragraphs.append(line)
    return ExtractedDocument(_bounded("\n\n".join(paragraphs)), None, "docx-xml-v1")


def _extract_pdf(path: Path) -> ExtractedDocument:
    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise DocumentExtractionError(
            "PDF extraction requires the mfq daemon document dependency"
        ) from error
    try:
        reader = PdfReader(path)
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as error:
        raise DocumentExtractionError("invalid or unsupported PDF document") from error
    text = re.sub(r"[ \t]+\n", "\n", "\n\n".join(pages))
    return ExtractedDocument(_bounded(text), len(reader.pages), "pypdf-v1")
