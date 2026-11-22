"""Load the frozen constants embedded in PROTOCOL_v2.md."""

from __future__ import annotations

import hashlib
import json
import re

from conn_assoc.paths import MANIFEST_PATH, PROTOCOL_PATH

_FENCE = re.compile(r"```frozen-json\n(.*?)\n```", re.DOTALL)


def protocol_text() -> str:
    return PROTOCOL_PATH.read_text(encoding="utf-8")


def protocol_sha256() -> str:
    return hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()


def frozen_constants() -> dict:
    matches = _FENCE.findall(protocol_text())
    if len(matches) != 1:
        raise RuntimeError(f"Expected one frozen-json block, found {len(matches)}")
    payload = json.loads(matches[0])
    if payload.get("protocol_version") != "v2":
        raise RuntimeError("Frozen constants are not protocol version v2")
    return payload


def manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
