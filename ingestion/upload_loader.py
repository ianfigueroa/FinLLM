from __future__ import annotations

from io import BytesIO

from pypdf import PdfReader

from ingestion.document_cleaner import clean_text

_MAX_EMPTY_TEXT_MESSAGE = "PDF did not contain extractable text; scanned PDFs need OCR first"
_PDF_CONTENT_TYPES = {"application/pdf", "application/x-pdf"}
_TEXT_CONTENT_TYPES = {"text/plain", "application/octet-stream"}


class UploadedFileError(ValueError):
    def __init__(self, message: str, *, status_code: int = 422) -> None:
        super().__init__(message)
        self.status_code = status_code


def extract_upload_text(
    *,
    filename: str | None,
    content_type: str | None,
    content: bytes,
) -> str:
    normalized_content_type = (content_type or "").lower()
    normalized_filename = (filename or "").lower()

    if _is_pdf_upload(normalized_filename, normalized_content_type, content):
        return _extract_pdf_text(content)
    if _is_text_upload(normalized_filename, normalized_content_type):
        return _extract_text(content)

    raise UploadedFileError(
        "Only text and PDF uploads are supported",
        status_code=415,
    )


def _is_pdf_upload(filename: str, content_type: str, content: bytes) -> bool:
    return (
        content_type in _PDF_CONTENT_TYPES
        or filename.endswith(".pdf")
        or content.startswith(b"%PDF-")
    )


def _is_text_upload(filename: str, content_type: str) -> bool:
    return content_type in _TEXT_CONTENT_TYPES or filename.endswith((".txt", ".text"))


def _extract_text(content: bytes) -> str:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise UploadedFileError("Text upload must be UTF-8 encoded") from exc
    cleaned = clean_text(text)
    if not cleaned:
        raise UploadedFileError("Text upload did not contain extractable text")
    return cleaned


def _extract_pdf_text(content: bytes) -> str:
    if not content.startswith(b"%PDF-"):
        raise UploadedFileError("PDF upload is missing a valid PDF header")
    try:
        reader = PdfReader(BytesIO(content))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as exc:
        raise UploadedFileError("PDF upload could not be parsed") from exc

    cleaned = clean_text(text)
    if not cleaned:
        raise UploadedFileError(_MAX_EMPTY_TEXT_MESSAGE)
    return cleaned
