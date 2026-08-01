"""Small, conservative redaction helpers for LLM and application-log boundaries."""
from __future__ import annotations

import re


_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN [^-]{0,80}PRIVATE KEY-----.*?-----END [^-]{0,80}PRIVATE KEY-----",
    re.IGNORECASE | re.DOTALL,
)
_AUTHORIZATION_RE = re.compile(
    r"\b(authorization\s*:\s*(?:bearer|basic)\s+)[^\s,;]+",
    re.IGNORECASE,
)
_SECRET_ASSIGNMENT_RE = re.compile(
    r"\b(password|passwd|token|secret|api[_-]?key|access[_-]?key|private[_-]?key)"
    r"(\s*[:=]\s*)(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)",
    re.IGNORECASE,
)
_URL_CREDENTIALS_RE = re.compile(
    r"\b([a-z][a-z0-9+.-]*://)([^\s:/@]+):([^\s@/]+)@",
    re.IGNORECASE,
)


def redact_sensitive_text(value: object) -> str:
    """Remove common credentials while preserving enough context for RCA."""
    text = str(value or "")
    text = _PRIVATE_KEY_RE.sub("[REDACTED PRIVATE KEY]", text)
    text = _AUTHORIZATION_RE.sub(r"\1[REDACTED]", text)
    text = _SECRET_ASSIGNMENT_RE.sub(r"\1\2[REDACTED]", text)
    return _URL_CREDENTIALS_RE.sub(r"\1[REDACTED]@", text)
