#!/usr/bin/env python3
"""Convenience launcher for the local web UI."""

from __future__ import annotations

import os
import sys

from search_server import main


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if len(sys.argv) == 1:
        sys.argv.extend(["--host", "127.0.0.1", "--port", "8000"])
    main()
