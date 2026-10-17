#!/usr/bin/env python3
"""Launch Local Chat on Windows, Linux or macOS: python start.py."""

import sys


def main():
    if sys.version_info < (3, 10):
        print("Local Chat requires Python 3.10 or newer.", file=sys.stderr)
        return 1

    from local_chat.__main__ import main as run

    return run()


if __name__ == "__main__":
    raise SystemExit(main())
