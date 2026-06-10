#!/usr/bin/env python
"""Backward-compatible concat training entrypoint."""

from __future__ import annotations

import sys

from train_expression_head import main


if __name__ == "__main__":
    if "--head" not in sys.argv and "--head-type" not in sys.argv:
        sys.argv[1:1] = ["--head", "concat"]
    main()
