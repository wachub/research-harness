"""Conservative source identities and fingerprints; never infer semantic equivalence."""

import hashlib
import json
import re
import unicodedata
from urllib.parse import unquote, urlsplit


def normalized_text(text: str) -> str:
    # Preserve case, punctuation, operators and accents in mathematical statements.
    return " ".join(unicodedata.normalize("NFC", text).split())


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def source_identifier(url: str | None) -> str | None:
    """Recognize DOI/arXiv identifiers, retaining an explicit arXiv revision."""
    if not url:
        return None
    try:
        parsed = urlsplit(url)
    except ValueError:
        return None
    path = unquote(parsed.path).strip("/")
    host = (parsed.hostname or "").lower()
    if host in {"doi.org", "dx.doi.org"} and re.fullmatch(r"10\.\d{4,9}/\S+", path):
        return "doi:" + path.casefold()
    if host in {"arxiv.org", "www.arxiv.org", "export.arxiv.org"}:
        match = re.fullmatch(r"(?:abs|pdf)/(.+?)(?:\.pdf)?", path)
        if match and re.fullmatch(r"(?:\d{4}\.\d{4,5}|[A-Za-z.-]+/\d{7})(?:v\d+)?", match[1]):
            return "arxiv:" + match[1]
    return None


def same_paper(left, right) -> bool:
    a, b = source_identifier(left.url), source_identifier(right.url)
    if a and b:
        # Different versions or different DOIs must not collapse on a title match.
        return a == b
    return (normalized_text(left.title).casefold() == normalized_text(right.title).casefold()
            and left.year == right.year
            and sorted(normalized_text(x).casefold() for x in left.authors)
            == sorted(normalized_text(x).casefold() for x in right.authors))
