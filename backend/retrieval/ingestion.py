"""
MindBridge 2.0 - Governed Knowledge Base Ingestion & Chunking
==============================================================

Parses, validates, chunks, and filters knowledge base documents in strict
accordance with the Phase 3 Knowledge Base Audit and architectural contracts.

Key Invariants:
1. Approved Documents (13): Ingested for open informational retrieval.
2. Restricted Documents (7): Ingested with 'restricted' eligibility, mandatory
   guardrails, and clinical disclaimers.
3. Excluded Documents (1): 'somatic_regulation/06_vagus_nerve_stimulation_tapping.md'
   is strictly blocked from index ingestion.
4. Unknown Documents: Skipped and reported; never ingested.
5. Chunking: Header-aware chunking on '##' and '###' headers targeting 300-500
   tokens with ~50-token overlap.
6. Long paragraphs & sentences: Sub-split safely using tokenizer-measured slices
   without silently dropping any content.
7. Content Hash: SHA-256 hash computed for each chunk to guarantee integrity
   and prevent duplicates.
8. Frontmatter: Safely parsed, but NEVER trusted to override governance registry.
9. Token Transparency: If tokenizer is unavailable, explicitly flags token counts
   as heuristic approximations.
"""

import os
import re
import glob
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

from backend.orchestrator.state import KnowledgeChunk, JurisdictionCode
from backend.retrieval.governance import (
    DOCUMENT_GOVERNANCE_REGISTRY,
    DocumentGovernance,
    get_document_governance,
    is_document_eligible_for_ingestion,
)


def load_compatible_tokenizer():
    """
    Load a BGE/BERT compatible tokenizer offline from local cache if present,
    or local RoBERTa model directory, or return (None, 'heuristic', False).
    """
    # 1. Check local models/embeddings/bge-small-en-v1.5 directory
    try:
        from transformers import AutoTokenizer
        bge_local_dir = (
            Path(__file__).resolve().parent.parent.parent
            / "models"
            / "embeddings"
            / "bge-small-en-v1.5"
        )
        if bge_local_dir.exists():
            tok = AutoTokenizer.from_pretrained(str(bge_local_dir), local_files_only=True)
            return tok, "bge_local", True
    except Exception:
        pass

    # 2. Check local BGE snapshot in cache if available
    try:
        from transformers import PreTrainedTokenizerFast
        bge_snaps = glob.glob(
            os.path.expanduser(
                "~/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/snapshots/*/tokenizer.json"
            )
        )
        if bge_snaps and os.path.exists(bge_snaps[0]):
            return PreTrainedTokenizerFast(tokenizer_file=bge_snaps[0]), "bge_snapshot", True
    except Exception:
        pass

    # 3. Check local bert snapshot (WordPiece, vocab=30522, identical to BGE)
    try:
        from transformers import PreTrainedTokenizerFast
        snaps = glob.glob(
            os.path.expanduser(
                "~/.cache/huggingface/hub/models--bert-base-uncased/snapshots/*/tokenizer.json"
            )
        )
        if snaps and os.path.exists(snaps[0]):
            return PreTrainedTokenizerFast(tokenizer_file=snaps[0]), "bert_snapshot", True
    except Exception:
        pass

    # 4. Check local roberta model directory
    try:
        from transformers import AutoTokenizer
        roberta_dir = Path(__file__).resolve().parent.parent.parent / "models" / "crisis_detector" / "roberta"
        if roberta_dir.exists():
            tok = AutoTokenizer.from_pretrained(str(roberta_dir), local_files_only=True)
            return tok, "roberta_local", True
    except Exception:
        pass

    # 5. Fallback to heuristic tokenizer
    return None, "heuristic", False


class MarkdownSectionChunker:
    """
    Header-aware markdown chunker designed for clinical and psychoeducational texts.
    Preserves document title and header context while adhering to token budgets.
    """

    def __init__(
        self,
        tokenizer: Optional[Any] = None,
        target_min_tokens: int = 250,
        target_max_tokens: int = 500,
        overlap_tokens: int = 50,
    ):
        self.target_min_tokens = target_min_tokens
        self.target_max_tokens = target_max_tokens
        self.overlap_tokens = overlap_tokens

        if tokenizer is not None:
            self.tokenizer = tokenizer
            self.tokenizer_source = "injected"
            self.is_exact_token_count = True
        else:
            self.tokenizer, self.tokenizer_source, self.is_exact_token_count = (
                load_compatible_tokenizer()
            )

    def count_tokens(self, text: str) -> int:
        """Count tokens using tokenizer if available, else approximate."""
        if not text:
            return 0
        if self.tokenizer is not None:
            try:
                if hasattr(self.tokenizer, "tokenize"):
                    return len(self.tokenizer.tokenize(text))
                if hasattr(self.tokenizer, "encode"):
                    return len(self.tokenizer.encode(text))
            except Exception:
                pass
        # Deterministic regex approximation (~1.3 tokens per word)
        tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
        return len(tokens)

    def chunk_document(
        self,
        markdown_body: str,
        doc_governance: DocumentGovernance,
    ) -> List[Dict[str, Any]]:
        """
        Split a markdown document into semantic chunks with metadata.
        """
        title = doc_governance.title
        body = markdown_body.strip()
        if not body:
            return []

        # Split on level 2 and level 3 headers: ## and ###
        raw_sections = re.split(r"\n(?=#{2,3}\s+)", body)

        parsed_sections: List[Tuple[str, str]] = []  # (header, section_text)
        for s in raw_sections:
            s_clean = s.strip()
            if not s_clean:
                continue
            if s_clean.startswith("# ") and not s_clean.startswith("## "):
                # Level 1 title line
                lines = s_clean.split("\n", 1)
                title_line = lines[0].replace("# ", "").strip()
                if title_line:
                    title = title_line
                if len(lines) > 1 and lines[1].strip():
                    parsed_sections.append(("Overview", lines[1].strip()))
            else:
                header_match = re.search(r"^(#{2,3}\s+[^\n]+)", s_clean)
                header_title = (
                    header_match.group(1).lstrip("#").strip()
                    if header_match
                    else "General"
                )
                parsed_sections.append((header_title, s_clean))

        if not parsed_sections:
            return []

        # Build chunks from parsed sections
        raw_chunks: List[Tuple[str, str]] = []
        current_text = ""
        current_header = ""

        for header, sec_text in parsed_sections:
            sec_tokens = self.count_tokens(sec_text)

            # If section exceeds max tokens, split across paragraphs / sentences
            if sec_tokens > self.target_max_tokens:
                # Flush existing accumulated chunk first
                if current_text:
                    raw_chunks.append((current_header, current_text))
                    current_text = ""
                    current_header = ""

                # Sub-split oversized section along paragraphs/sentences
                sub_chunks = self._split_oversized_section(sec_text, header)
                raw_chunks.extend(sub_chunks)
                continue

            current_tok_count = self.count_tokens(current_text)
            if current_tok_count + sec_tokens <= self.target_max_tokens:
                if current_text:
                    current_text += "\n\n" + sec_text
                else:
                    current_text = sec_text
                    current_header = header
            else:
                # Flush current and start new with token overlap
                if current_text:
                    raw_chunks.append((current_header, current_text))

                    overlap_snippet = self._extract_token_overlap(
                        current_text, self.overlap_tokens
                    )
                    if overlap_snippet:
                        current_text = overlap_snippet + "\n\n" + sec_text
                    else:
                        current_text = sec_text
                    current_header = header
                else:
                    raw_chunks.append((header, sec_text))
                    current_text = ""
                    current_header = ""

        if current_text:
            raw_chunks.append((current_header, current_text))

        # Format into structured chunk records
        formatted_chunks: List[Dict[str, Any]] = []
        for idx, (primary_header, text) in enumerate(raw_chunks, start=1):
            formatted_text = text.strip()
            # Ensure document title header is present at the top of chunk
            if not formatted_text.startswith(f"# {title}"):
                formatted_text = f"# {title}\n\n{formatted_text}"

            # Verify chunk is not empty or header-only
            lines_content = [
                line.strip()
                for line in formatted_text.splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]
            if not lines_content:
                continue

            content_hash = hashlib.sha256(
                formatted_text.encode("utf-8")
            ).hexdigest()

            chunk_id = f"{doc_governance.document_id}__chunk_{idx:02d}"

            chunk_record: Dict[str, Any] = {
                "chunk_id": chunk_id,
                "document_id": doc_governance.document_id,
                "source": doc_governance.relative_path,
                "title": doc_governance.title,
                "category": doc_governance.category,
                "jurisdiction": doc_governance.jurisdiction,
                "verification_status": doc_governance.verification_status,
                "retrieval_tier": doc_governance.retrieval_tier,
                "safety_level": doc_governance.clinical_safety_level,
                "disclaimer": doc_governance.disclaimer,
                "content_hash": content_hash,
                "chunk_index": idx - 1,
                "content": formatted_text,
                "score": 0.0,
                "eligibility": "restricted"
                if doc_governance.retrieval_tier == "restricted"
                else "approved",
                "section_header": primary_header,
                "guardrails": list(doc_governance.guardrails),
                "disclaimer_text": doc_governance.disclaimer,
                "clinical_safety_level": doc_governance.clinical_safety_level,
                "token_count": self.count_tokens(formatted_text),
                "is_exact_token_count": self.is_exact_token_count,
                "token_count_method": "tokenizer"
                if self.is_exact_token_count
                else "heuristic_approximation",
            }
            formatted_chunks.append(chunk_record)

        return formatted_chunks

    def _split_oversized_section(
        self, section_text: str, section_header: str
    ) -> List[Tuple[str, str]]:
        """
        Split a large section across paragraph boundaries.
        If any single paragraph still exceeds target_max_tokens, sub-split it
        along sentences/token windows so that no text is silently dropped.
        """
        raw_paragraphs = [p.strip() for p in section_text.split("\n\n") if p.strip()]
        paragraphs: List[str] = []

        # Ensure no individual paragraph exceeds target_max_tokens
        for p in raw_paragraphs:
            if self.count_tokens(p) > self.target_max_tokens:
                sub_p = self._split_oversized_paragraph(p)
                paragraphs.extend(sub_p)
            else:
                paragraphs.append(p)

        chunks: List[Tuple[str, str]] = []
        cur_text = ""

        for p in paragraphs:
            p_toks = self.count_tokens(p)
            cur_toks = self.count_tokens(cur_text)

            if cur_toks + p_toks <= self.target_max_tokens:
                cur_text = f"{cur_text}\n\n{p}" if cur_text else p
            else:
                if cur_text:
                    chunks.append((section_header, cur_text))
                    overlap = self._extract_token_overlap(
                        cur_text, self.overlap_tokens
                    )
                    cur_text = f"{overlap}\n\n{p}" if overlap else p
                else:
                    chunks.append((section_header, p))
                    cur_text = ""

        if cur_text:
            chunks.append((section_header, cur_text))

        return chunks

    def _split_oversized_paragraph(self, paragraph: str) -> List[str]:
        """
        Split an oversized paragraph along sentences.
        If an individual sentence still exceeds target_max_tokens, sub-split it
        using token-measured word slices so that no text is dropped.
        """
        raw_sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        sentences: List[str] = []
        for s in raw_sentences:
            s_clean = s.strip()
            if not s_clean:
                continue
            if self.count_tokens(s_clean) > self.target_max_tokens:
                sentences.extend(self._split_oversized_sentence(s_clean))
            else:
                sentences.append(s_clean)

        sub_paras: List[str] = []
        cur_sent_group = ""

        for s in sentences:
            s_toks = self.count_tokens(s)
            cur_toks = self.count_tokens(cur_sent_group)

            if cur_toks + s_toks <= self.target_max_tokens:
                cur_sent_group = f"{cur_sent_group} {s}" if cur_sent_group else s
            else:
                if cur_sent_group:
                    sub_paras.append(cur_sent_group)
                    cur_sent_group = s
                else:
                    sub_paras.append(s)
                    cur_sent_group = ""

        if cur_sent_group:
            sub_paras.append(cur_sent_group)

        return sub_paras

    def _split_oversized_sentence(self, sentence: str) -> List[str]:
        """
        Split a single sentence that exceeds target_max_tokens.
        Uses actual token counting (self.count_tokens) to build slices,
        preserving every single word without dropping content.
        """
        words = sentence.split()
        slices: List[str] = []
        cur_slice_words: List[str] = []

        for word in words:
            cur_slice_words.append(word)
            candidate = " ".join(cur_slice_words)
            if self.count_tokens(candidate) >= self.target_max_tokens:
                slices.append(candidate)
                # Next slice starts with overlap if configured
                if self.overlap_tokens > 0:
                    overlap_snippet = self._extract_token_overlap(
                        candidate, self.overlap_tokens
                    )
                    cur_slice_words = overlap_snippet.split() if overlap_snippet else []
                else:
                    cur_slice_words = []

        if cur_slice_words:
            remainder = " ".join(cur_slice_words)
            if not slices or remainder != slices[-1]:
                slices.append(remainder)

        return slices

    def _extract_token_overlap(
        self, text: str, target_overlap_tokens: int
    ) -> str:
        """
        Extract trailing text from previous chunk to form approximately
        target_overlap_tokens overlap. Preserves whole word boundaries and verifies
        token counts using self.count_tokens().
        """
        if not text or target_overlap_tokens <= 0:
            return ""

        words = text.split()
        if not words:
            return ""

        selected_words: List[str] = []
        for word in reversed(words):
            selected_words.append(word)
            candidate = " ".join(reversed(selected_words))
            if self.count_tokens(candidate) >= target_overlap_tokens:
                break

        # If overlap is the entire text, return empty to avoid duplicate looping
        if len(selected_words) >= len(words):
            return ""

        return " ".join(reversed(selected_words))


def parse_frontmatter_safe(content: str) -> Tuple[Dict[str, str], str]:
    """
    Safely extract YAML frontmatter without external yaml dependencies.
    Never trusted to override governance registry decisions.
    """
    frontmatter: Dict[str, str] = {}
    match = re.match(r"^---\n(.*?)\n---\n*", content, re.DOTALL)
    if not match:
        return frontmatter, content

    raw_yaml = match.group(1)
    body = content[match.end() :]

    for line in raw_yaml.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if val.lower() == "null":
                val = ""
            frontmatter[key] = val

    return frontmatter, body


class KnowledgeBaseIngestor:
    """
    Governed knowledge base ingestor.
    Discovers, validates, chunks, and filters documents strictly according to governance.
    """

    def __init__(
        self,
        base_dir: Optional[Union[str, Path]] = None,
        chunker: Optional[MarkdownSectionChunker] = None,
    ):
        if base_dir is None:
            self.base_dir = (
                Path(__file__).resolve().parent.parent.parent
                / "data"
                / "knowledge_base"
            )
        else:
            self.base_dir = Path(base_dir)

        self.chunker = chunker or MarkdownSectionChunker()

    def ingest(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Execute end-to-end ingestion of the knowledge base.

        Returns:
            Tuple of (chunks, ingestion_summary)
        """
        if not self.base_dir.exists():
            raise FileNotFoundError(
                f"Knowledge base directory does not exist: {self.base_dir}"
            )

        markdown_files = sorted(self.base_dir.glob("**/*.md"))
        chunks: List[Dict[str, Any]] = []
        seen_content_hashes = set()

        summary: Dict[str, Any] = {
            "total_files_scanned": len(markdown_files),
            "approved_files_ingested": 0,
            "restricted_files_ingested": 0,
            "excluded_files_blocked": 0,
            "unregistered_files_skipped": 0,
            "total_chunks_produced": 0,
            "duplicates_pruned": 0,
            "blocked_documents": [],
            "unregistered_documents": [],
            "ingested_documents": [],
            "tokenizer_info": {
                "source": self.chunker.tokenizer_source,
                "is_exact_token_count": self.chunker.is_exact_token_count,
                "token_count_method": "tokenizer"
                if self.chunker.is_exact_token_count
                else "heuristic_approximation",
            },
        }

        for file_path in markdown_files:
            rel_path = file_path.relative_to(self.base_dir).as_posix()
            gov = get_document_governance(rel_path)

            # Skip unregistered / unknown documents
            if gov is None:
                summary["unregistered_files_skipped"] += 1
                summary["unregistered_documents"].append(rel_path)
                continue

            # Strict exclusion check
            if gov.retrieval_tier == "excluded_pending_review":
                summary["excluded_files_blocked"] += 1
                summary["blocked_documents"].append(
                    {
                        "document_id": gov.document_id,
                        "relative_path": rel_path,
                        "rejection_reason": gov.rejection_reason,
                    }
                )
                continue

            # Ingest approved or restricted document
            with open(file_path, "r", encoding="utf-8") as f:
                raw_content = f.read()

            _frontmatter, markdown_body = parse_frontmatter_safe(raw_content)

            # Chunk document
            doc_chunks = self.chunker.chunk_document(markdown_body, gov)

            # Deduplication by content hash
            ingested_doc_chunks = 0
            for chunk in doc_chunks:
                c_hash = chunk["content_hash"]
                if c_hash in seen_content_hashes:
                    summary["duplicates_pruned"] += 1
                    continue
                seen_content_hashes.add(c_hash)
                chunks.append(chunk)
                ingested_doc_chunks += 1

            if gov.retrieval_tier == "approved":
                summary["approved_files_ingested"] += 1
            elif gov.retrieval_tier == "restricted":
                summary["restricted_files_ingested"] += 1

            summary["ingested_documents"].append(
                {
                    "document_id": gov.document_id,
                    "relative_path": rel_path,
                    "retrieval_tier": gov.retrieval_tier,
                    "chunk_count": ingested_doc_chunks,
                }
            )

        summary["total_chunks_produced"] = len(chunks)
        return chunks, summary
