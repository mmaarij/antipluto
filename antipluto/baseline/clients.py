"""
antipluto.baseline.clients
========================
Lightweight, zero-dependency client implementations for querying:
1. Apache SpamAssassin (spamd daemon on port 783 via SPAMC/1.5 protocol)
2. Rspamd (HTTP REST API on port 11333 via /checkv2)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import re
import socket
from typing import Any, Dict, List, Optional
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)


@dataclass
class BaselineFilterResult:
    engine: str
    record_id: str
    is_phish_or_spam: bool
    score: float
    threshold: float
    action: str
    symbols: List[str] = field(default_factory=list)
    raw_response: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0


class SpamAssassinClient:
    """Client for querying the Apache SpamAssassin daemon (spamd) via socket."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 783,
        timeout: float = 10.0,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout

    def ping(self) -> bool:
        """Check if the spamd daemon is reachable and responding."""
        try:
            with socket.create_connection((self.host, self.port), timeout=self.timeout) as s:
                s.sendall(b"PING SPAMC/1.5\r\n\r\n")
                resp = s.recv(1024)
                return b"PONG" in resp
        except (socket.error, socket.timeout, ConnectionRefusedError):
            return False

    def check(
        self,
        raw_email: bytes,
        record_id: str = "unknown",
        command: str = "CHECK",
    ) -> BaselineFilterResult:
        """Query spamd to evaluate an email.

        Parameters
        ----------
        raw_email : bytes
            RFC 5322 MIME message bytes.
        record_id : str
            Identifier for logging.
        command : str
            SPAMC command: 'CHECK' (fastest) or 'SYMBOLS'.

        Returns
        -------
        BaselineFilterResult
        """
        import time

        t0 = time.perf_counter()
        header = f"{command} SPAMC/1.5\r\nContent-length: {len(raw_email)}\r\n\r\n".encode("utf-8")
        payload = header + raw_email

        try:
            with socket.create_connection((self.host, self.port), timeout=self.timeout) as s:
                s.sendall(payload)
                s.shutdown(socket.SHUT_WR)

                response_chunks = []
                while True:
                    chunk = s.recv(4096)
                    if not chunk:
                        break
                    response_chunks.append(chunk)

                resp_str = b"".join(response_chunks).decode("utf-8", errors="replace")
        except Exception as exc:
            logger.error("SpamAssassin socket error for record %s: %s", record_id, exc)
            return BaselineFilterResult(
                engine="SpamAssassin",
                record_id=record_id,
                is_phish_or_spam=False,
                score=0.0,
                threshold=5.0,
                action="error",
                raw_response={"error": str(exc)},
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        latency = (time.perf_counter() - t0) * 1000.0

        # Parse SPAMD response header
        # Example: Spam: True ; 8.2 / 5.0
        # Or: Spam: False ; 0.5 / 5.0
        is_spam = False
        score = 0.0
        threshold = 5.0
        symbols = []

        match = re.search(r"Spam:\s*(True|False|Yes|No)\s*;\s*([-\d\.]+)\s*/\s*([-\d\.]+)", resp_str, re.IGNORECASE)
        if match:
            is_spam = match.group(1).lower() in ("true", "yes")
            score = float(match.group(2))
            threshold = float(match.group(3))

        # Check if symbols were returned
        lines = resp_str.splitlines()
        for line in lines:
            line = line.strip()
            if line and not line.startswith("SPAMD") and not line.startswith("Content-length") and not line.startswith("Spam:"):
                # Potential symbol line
                symbols.extend([s.strip() for s in line.split(",") if s.strip()])

        return BaselineFilterResult(
            engine="SpamAssassin",
            record_id=record_id,
            is_phish_or_spam=is_spam,
            score=score,
            threshold=threshold,
            action="spam" if is_spam else "ham",
            symbols=symbols,
            raw_response={"raw": resp_str},
            latency_ms=latency,
        )


class RspamdClient:
    """Client for querying the Rspamd HTTP REST API (port 11333 /checkv2)."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11333",
        timeout: float = 10.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def ping(self) -> bool:
        """Check if Rspamd HTTP service is reachable."""
        try:
            req = urllib.request.Request(f"{self.base_url}/ping")
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.read().decode("utf-8").strip().lower() == "pong"
        except Exception:
            return False

    def check(
        self,
        raw_email: bytes,
        record_id: str = "unknown",
    ) -> BaselineFilterResult:
        """Query Rspamd /checkv2 endpoint with an email.

        Parameters
        ----------
        raw_email : bytes
            RFC 5322 MIME message bytes.
        record_id : str
            Identifier for logging.

        Returns
        -------
        BaselineFilterResult
        """
        import time

        t0 = time.perf_counter()
        req = urllib.request.Request(
            url=f"{self.base_url}/checkv2",
            data=raw_email,
            headers={
                "Content-Type": "message/rfc822",
                "Queue-Id": record_id,
                "Pass": "all",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                resp_json = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            logger.error("Rspamd HTTP error for record %s: %s", record_id, exc)
            return BaselineFilterResult(
                engine="Rspamd",
                record_id=record_id,
                is_phish_or_spam=False,
                score=0.0,
                threshold=15.0,
                action="error",
                raw_response={"error": str(exc)},
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        latency = (time.perf_counter() - t0) * 1000.0

        score = float(resp_json.get("score", 0.0))
        required_score = float(resp_json.get("required_score", 15.0))
        action = resp_json.get("action", "no action")
        is_spam = resp_json.get("is_spam", False) or action in ("reject", "add header")

        symbols_dict = resp_json.get("symbols", {})
        symbols = list(symbols_dict.keys()) if isinstance(symbols_dict, dict) else []

        return BaselineFilterResult(
            engine="Rspamd",
            record_id=record_id,
            is_phish_or_spam=is_spam,
            score=score,
            threshold=required_score,
            action=action,
            symbols=symbols,
            raw_response=resp_json,
            latency_ms=latency,
        )
