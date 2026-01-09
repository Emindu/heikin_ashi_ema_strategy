# backtest/__main__.py
"""Entry point for running backtest as a module: python -m backtest"""

import argparse

from .strategy import run_backtest_and_plot
from .optimizer import run_optimization


def main():
    parser = argparse.ArgumentParser(description='Heikin Ashi EMA Strategy Backtester')
    parser.add_argument('--optimize', action='store_true', help='Run parameter optimization')
    args = parser.parse_args()
    
    if args.optimize:
        run_optimization()
    else:
        run_backtest_and_plot()


if __name__ == "__main__":
    main()
