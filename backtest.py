#!/usr/bin/env python3
"""
Backward compatible entry point.
Run: python backtest.py or python backtest.py --optimize
"""

import sys
from backtest.__main__ import main

if __name__ == "__main__":
    main()