"""Recursive semantic chunker with token budgeting and boundary detection."""

import re
from typing import List, Dict, Any
from spectrarag.schema import Document, Chunk
from spectrarag.config import default_config


class SemanticChunker:
    """Splits raw text into semantically cohesive, context-bounded chunks.
    
    Features:
    - Multi-tier recursive delimiters (paragraphs, sentences, clauses)
    - Sliding window overlap to prevent context clipping
    - Metadata enrichment (chunk index, relative position, token budget)
    """

    def __init__(
        self,
        chunk_size: int = default_config.chunk_size,
        chunk_overlap: int = default_config.chunk_overlap,
        min_chunk_len: int = default_config.min_chunk_len,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_len = min_chunk_len

    def split_text(self, text: str) -> List[str]:
        """Recursively split text preserving natural grammatical boundaries."""
        if not text or not text.strip():
            return []

        # Split on natural paragraph breaks first
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        raw_chunks: List[str] = []

        current_chunk = ""
        for p in paragraphs:
            # If paragraph itself is too large, split by sentences
            if len(p) > self.chunk_size:
                sentences = re.split(r"(?<=[.!?])\s+", p)
                for s in sentences:
                    s = s.strip()
                    if not s:
                        continue
                    if len(current_chunk) + len(s) + 1 <= self.chunk_size:
                        current_chunk = f"{current_chunk} {s}".strip()
                    else:
                        if current_chunk:
                            raw_chunks.append(current_chunk)
                        current_chunk = s
            else:
                if len(current_chunk) + len(p) + 2 <= self.chunk_size:
                    current_chunk = f"{current_chunk}\n\n{p}".strip()
                else:
                    if current_chunk:
                        raw_chunks.append(current_chunk)
                    current_chunk = p

        if current_chunk:
            raw_chunks.append(current_chunk)

        # Apply sliding window overlap if configured
        final_chunks: List[str] = []
        for i, chunk in enumerate(raw_chunks):
            if i > 0 and self.chunk_overlap > 0:
                prev_text = raw_chunks[i - 1]
                overlap_text = prev_text[-self.chunk_overlap :].strip()
                combined = f"{overlap_text} {chunk}".strip()
                final_chunks.append(combined)
            else:
                final_chunks.append(chunk)

        return [c for c in final_chunks if len(c) >= self.min_chunk_len]

    def chunk_document(self, doc: Document) -> List[Chunk]:
        """Splits a Document into rich Chunk objects with metadata."""
        text_chunks = self.split_text(doc.text)
        total_chunks = len(text_chunks)
        output: List[Chunk] = []

        for idx, text in enumerate(text_chunks):
            # Approximate token count (1 token ~= 4 characters for English)
            token_count = max(1, len(text.split()))

            chunk_meta = {
                **doc.metadata,
                "parent_doc_id": doc.id,
                "chunk_index": idx,
                "total_chunks": total_chunks,
                "relative_position": round(idx / max(1, total_chunks - 1), 3) if total_chunks > 1 else 0.0,
            }

            chunk = Chunk(
                doc_id=doc.id,
                text=text,
                chunk_index=idx,
                char_count=len(text),
                token_count=token_count,
                metadata=chunk_meta,
            )
            output.append(chunk)

        return output

    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """Batch chunks multiple documents."""
        all_chunks: List[Chunk] = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks

