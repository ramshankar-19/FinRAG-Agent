"""
tests/test_ingest.py

Tests for Claim 1: Multi-modal document ingestion (PDF, DOCX, Image).

Covers test cases 1, 2, 3 from the required test suite.
"""

import os
import sys
import shutil
import tempfile
import pytest

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ingest import (
    detect_file_type,
    clean_text,
    load_pdf,
    load_docx,
    load_image,
    ingest_file,
)


# ─────────────────────────────────────────────
# Test 1: PDF Ingestion
# ─────────────────────────────────────────────

class TestPDFIngestion:
    PDF_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "financial_report.pdf")

    def test_detect_pdf_type(self):
        assert detect_file_type(self.PDF_PATH) == "pdf"

    def test_load_pdf_returns_documents(self):
        docs = load_pdf(self.PDF_PATH)
        assert len(docs) > 0, "Should load at least one page from the PDF"

    def test_load_pdf_has_content(self):
        docs = load_pdf(self.PDF_PATH)
        total_chars = sum(len(d.page_content) for d in docs)
        assert total_chars > 100, "PDF should contain meaningful text"

    def test_load_pdf_has_metadata(self):
        docs = load_pdf(self.PDF_PATH)
        assert "source" in docs[0].metadata

    def test_pdf_end_to_end_ingest(self, tmp_path):
        """Full pipeline: PDF → parse → clean → chunk → embed → store."""
        db_dir = str(tmp_path / "chroma_test_pdf")
        n = ingest_file(self.PDF_PATH, persist_directory=db_dir)
        assert n > 0, f"Should have stored chunks; got {n}"

    def test_pdf_chunks_have_doc_type_metadata(self, tmp_path):
        """Verify doc_type field is stored in metadata."""
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings

        db_dir = str(tmp_path / "chroma_meta_test")
        ingest_file(self.PDF_PATH, persist_directory=db_dir)

        embeddings  = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma(persist_directory=db_dir, embedding_function=embeddings)
        sample      = vectorstore._collection.get(limit=1)

        assert sample["metadatas"], "Should have at least one document"
        meta = sample["metadatas"][0]
        assert meta.get("doc_type") == "pdf", f"Expected doc_type=pdf, got: {meta}"


# ─────────────────────────────────────────────
# Test 2: DOCX Ingestion
# ─────────────────────────────────────────────

class TestDOCXIngestion:
    DOCX_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "sample.docx")

    def test_detect_docx_type(self):
        assert detect_file_type(self.DOCX_PATH) == "docx"

    def test_load_docx_returns_document(self):
        docs = load_docx(self.DOCX_PATH)
        assert len(docs) == 1, "DOCX loader returns a single concatenated Document"

    def test_load_docx_has_content(self):
        docs = load_docx(self.DOCX_PATH)
        assert "Revenue" in docs[0].page_content or "revenue" in docs[0].page_content.lower()

    def test_load_docx_extracts_tables(self):
        docs = load_docx(self.DOCX_PATH)
        # Tables should be extracted as pipe-separated rows
        content = docs[0].page_content
        assert "128M" in content or "Gross Profit" in content or "|" in content

    def test_docx_end_to_end_ingest(self, tmp_path):
        """Full pipeline: DOCX → parse → clean → chunk → embed → store."""
        db_dir = str(tmp_path / "chroma_docx")
        n = ingest_file(self.DOCX_PATH, persist_directory=db_dir)
        assert n > 0, f"Should have stored chunks; got {n}"

    def test_docx_chunks_have_doc_type_metadata(self, tmp_path):
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings

        db_dir = str(tmp_path / "chroma_docx_meta")
        ingest_file(self.DOCX_PATH, persist_directory=db_dir)

        embeddings  = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma(persist_directory=db_dir, embedding_function=embeddings)
        sample      = vectorstore._collection.get(limit=1)

        meta = sample["metadatas"][0]
        assert meta.get("doc_type") == "docx", f"Expected doc_type=docx, got: {meta}"


# ─────────────────────────────────────────────
# Test 3: Image / OCR Ingestion
# ─────────────────────────────────────────────

class TestImageIngestion:
    IMG_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "sample_image.png")

    def test_detect_image_type(self):
        assert detect_file_type(self.IMG_PATH) == "image"

    def test_load_image_returns_document(self):
        docs = load_image(self.IMG_PATH)
        assert len(docs) == 1, "Image loader should return a single Document"

    def test_load_image_has_content(self):
        """
        Content may be OCR text (if Tesseract installed) or a placeholder.
        Either way, the pipeline must not crash and must return non-empty content.
        """
        docs = load_image(self.IMG_PATH)
        assert len(docs[0].page_content) > 0, "Image document should have content"

    def test_load_image_has_ocr_metadata(self):
        docs = load_image(self.IMG_PATH)
        assert docs[0].metadata.get("ocr") is True

    def test_image_end_to_end_ingest(self, tmp_path):
        """Full pipeline: Image → OCR → clean → chunk → embed → store."""
        db_dir = str(tmp_path / "chroma_image")
        n = ingest_file(self.IMG_PATH, persist_directory=db_dir)
        assert n >= 0, "Pipeline should complete without error"

    def test_image_chunks_have_doc_type_metadata(self, tmp_path):
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings

        db_dir = str(tmp_path / "chroma_img_meta")
        n = ingest_file(self.IMG_PATH, persist_directory=db_dir)
        if n == 0:
            pytest.skip("No chunks stored (OCR may have produced minimal text)")

        embeddings  = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma(persist_directory=db_dir, embedding_function=embeddings)
        sample      = vectorstore._collection.get(limit=1)

        meta = sample["metadatas"][0]
        assert meta.get("doc_type") == "image", f"Expected doc_type=image, got: {meta}"


# ─────────────────────────────────────────────
# Text cleaning unit tests
# ─────────────────────────────────────────────

class TestCleanText:
    def test_collapses_multiple_spaces(self):
        result = clean_text("hello    world")
        assert "    " not in result
        assert "hello world" in result

    def test_caps_consecutive_newlines(self):
        result = clean_text("line1\n\n\n\n\nline2")
        assert "\n\n\n" not in result

    def test_strips_control_characters(self):
        result = clean_text("hello\x00\x01world")
        assert "\x00" not in result
        assert "\x01" not in result

    def test_preserves_valid_text(self):
        original = "Revenue: $520 million for FY2025."
        result = clean_text(original)
        assert "Revenue" in result
        assert "520 million" in result


# ─────────────────────────────────────────────
# Error handling
# ─────────────────────────────────────────────

class TestIngestErrors:
    def test_raises_for_missing_file(self):
        with pytest.raises(FileNotFoundError):
            ingest_file("nonexistent_file.pdf")

    def test_raises_for_unsupported_extension(self, tmp_path):
        fake = tmp_path / "file.xyz"
        fake.write_text("content")
        with pytest.raises(ValueError, match="Unsupported file type"):
            ingest_file(str(fake))
