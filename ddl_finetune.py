#!/usr/bin/env python3
"""Compatibility entrypoint.

This wrapper preserves the previous command entry while routing all
implementation to the package CLI.
"""

from detectanyllm.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
