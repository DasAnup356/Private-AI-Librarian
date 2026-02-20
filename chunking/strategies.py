# chunk docs (sliding window + by headings)
import re
from dataclasses import dataclass
from typing import List, Sequence

from config import CHUNK_OVERLAP, CHUNK_SIZE, MIN_CHUNK_CHARS


@dataclass
class Chunk:
    text: str
    doc_id: str
    chunk_index: int
    start_char: int
    end_char: int
    heading: str = ""


def _approx_tokens(text: str) -> int:
    # rough token estimate (~4 chars per token)
    return max(1, len(text) // 4)


def _split_into_sentences(text: str) -> List[str]:
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if p.strip()]


def sliding_window_chunks(
    text: str,
    doc_id: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    min_chunk_chars: int = MIN_CHUNK_CHARS,
) -> List[Chunk]:
    # sliding window chunking with overlap, tries to keep sentences intact
    sentences = _split_into_sentences(text)
    if not sentences:
        if text.strip():
            return [Chunk(text.strip(), doc_id, 0, 0, len(text))]
        return []

    target_tokens = chunk_size
    overlap_tokens = overlap
    chunks: List[Chunk] = []
    current: List[str] = []
    current_tokens = 0
    start_char = 0
    chunk_index = 0

    for sent in sentences:
        sent_tokens = _approx_tokens(sent)
        if current_tokens + sent_tokens > target_tokens and current:
            chunk_text = " ".join(current)
            if len(chunk_text) >= min_chunk_chars:
                end_char = start_char + len(chunk_text)
                chunks.append(
                    Chunk(chunk_text, doc_id, chunk_index, start_char, end_char)
                )
                chunk_index += 1
                start_char = end_char

            # keep overlap sentences
            overlap_accum = 0
            new_current = []
            for s in reversed(current):
                t = _approx_tokens(s)
                if overlap_accum + t <= overlap_tokens:
                    new_current.insert(0, s)
                    overlap_accum += t
                else:
                    break
            current = new_current
            current_tokens = overlap_accum

        current.append(sent)
        current_tokens += sent_tokens

    if current:
        chunk_text = " ".join(current)
        if len(chunk_text) >= min_chunk_chars:
            chunks.append(
                Chunk(
                    chunk_text,
                    doc_id,
                    chunk_index,
                    start_char,
                    start_char + len(chunk_text),
                )
            )
    return chunks


def _extract_sections_by_headings(text: str) -> List[tuple]:
    # looks for markdown headings, ALL CAPS lines, or numbered sections
    lines = text.split("\n")
    sections: List[tuple] = []
    current_heading = ""
    current_content: List[str] = []

    heading_pattern = re.compile(
        r"^(#{1,6}\s+.+|\d+\.\s+[A-Z][^.]+|[A-Z][A-Z\s]{2,50})$"
    )

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current_content:
                content = "\n".join(current_content)
                if content.strip():
                    sections.append((current_heading, content.strip()))
                current_content = []
            continue

        if heading_pattern.match(stripped) and len(stripped) < 200:
            if current_content:
                content = "\n".join(current_content)
                if content.strip():
                    sections.append((current_heading, content.strip()))
                current_content = []
            current_heading = stripped
            continue

        current_content.append(line)

    if current_content:
        content = "\n".join(current_content)
        if content.strip():
            sections.append((current_heading, content.strip()))
    return sections


def chunk_by_headings_and_sliding(
    text: str,
    doc_id: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    min_chunk_chars: int = MIN_CHUNK_CHARS,
) -> List[Chunk]:
    # split by headings first, then sliding window within each section
    sections = _extract_sections_by_headings(text)
    if not sections:
        return sliding_window_chunks(
            text, doc_id, chunk_size, overlap, min_chunk_chars
        )

    all_chunks: List[Chunk] = []
    global_offset = 0
    chunk_index = 0

    for heading, content in sections:
        sub_chunks = sliding_window_chunks(
            content, doc_id, chunk_size, overlap, min_chunk_chars
        )
        for c in sub_chunks:
            all_chunks.append(
                Chunk(
                    text=c.text,
                    doc_id=c.doc_id,
                    chunk_index=chunk_index,
                    start_char=c.start_char + global_offset,
                    end_char=c.end_char + global_offset,
                    heading=heading,
                )
            )
            chunk_index += 1
        if content:
            global_offset += len(content) + 2  # approximate

    return all_chunks


def chunks_to_texts(chunks: Sequence[Chunk]) -> List[str]:
    return [c.text for c in chunks]
