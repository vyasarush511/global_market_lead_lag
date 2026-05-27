# Global Market Lead-Lag Analysis

Research pipeline for testing whether U.S. equity-market session returns contain tradable information for the next Japanese equity-market session.

The project started with a simple question: because Japan and the U.S. trade at different times, can one market help predict the other? The current evidence points more strongly in the opposite direction from the original guess: U.S. market moves appear to lead Japan more than Japan leads the U.S.

## Research Question

Does the S&P 500 open-to-close return from the latest completed U.S. session help predict the Nikkei 225 open-to-close return during the next Japanese session?

Main setup:

- Signal: S&P 500 open-to-close return
- Target: Nikkei 225 next-session open-to-close return
- Data: Yahoo Finance via `yfinance`
- Start date: 2010-01-01
- Transaction cost assumption: 5 bps per unit of turnover
- Validation: fixed train/validation/test split and expanding walk-forward backtest

## Key Findings

Latest generated results are in [reports/summary.md](reports/summary.md).

Headline results:

- S&P previous session vs Nikkei next session correlation: `0.4140`
- Baseline regression R-squared: `1.32%`
- Fixed train/validation/test selected strategy: `Threshold Long Short 5bps`
- Fixed test Sharpe: `1.71` vs buy-and-hold Sharpe `0.44`
- Walk-forward Sharpe: `0.56` vs buy-and-hold Sharpe `0.12`
- Walk-forward max drawdown: `-14.55%` vs buy-and-hold `-28.86%`

Interpretation: the signal has modest but measurable value. The realistic walk-forward result is much weaker than the best fixed-split result, which is exactly why walk-forward validation is included. The edge appears to come mostly from selective exposure and drawdown reduction rather than high daily accuracy.

## Repository Structure

```text
global-market-lead-lag/
  notebooks/
    01_data_exploration.ipynb
    02_strategy_backtest.ipynb
  reports/
    figures/
    tables/
    summary.md
  scripts/
    run_research.py
  src/
    global_market_lead_lag/
      backtest.py
      config.py
      data.py
      metrics.py
      models.py
      pipeline.py
      plots.py
      signals.py
  tests/
  pyproject.toml
  requirements.txt
```

## Reproduce The Study

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run tests:

```powershell
python -m pytest
```

Regenerate all tables, charts, and the research summary:

```powershell
python scripts\run_research.py
```

The pipeline writes outputs to:

```text
reports/tables/
reports/figures/
reports/summary.md
```

## Methodology

The project uses open-to-close returns for both markets to isolate regular-session moves:

- U.S. open-to-close: captures the S&P 500 regular trading session.
- Japan open-to-close: captures the tradable Nikkei 225 session after the U.S. signal is known.

The main dataset uses `merge_asof(..., allow_exact_matches=False)` to match each Japanese session with the most recent prior U.S. session. This avoids using information that would not have been available at trade time.

Strategies tested:

- Long only
- Long short
- Threshold long only
- Threshold long short

Validation layers:

- Same-date and lagged correlation checks
- OLS regression with p-values and R-squared
- Fixed train/validation/test model selection
- Expanding walk-forward model selection and testing
- Transaction cost sensitivity
- Exposure, turnover, and invested-day diagnostics

## Limitations

This is a research project, not financial advice. Results depend on Yahoo Finance data quality, ETF/futures implementation choices, shorting constraints, taxes, fees, slippage, exchange holidays, and execution at the Japanese open. A production strategy would need richer data, explicit instrument selection, liquidity checks, and live paper trading.
