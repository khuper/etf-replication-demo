import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")


def _dedupe_legend(ax, **legend_kwargs):
    """Extract unique legend entries, preserving order."""
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax.legend(by_label.values(), by_label.keys(), **legend_kwargs)


def main():
    # Setup directories
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)

    # 1. Fetch Data
    assets = ["SPY", "QQQ", "IWM", "EFA", "VWO", "AGG", "TLT", "GLD", "VNQ", "HYG"]
    target = "PSP"
    tickers = assets + [target]

    # Using 2018 to capture all requested stress windows
    start_date = "2018-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")

    print(f"Fetching daily adjusted prices from {start_date} to {end_date}...")
    df = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)

    if df.empty:
        raise ValueError("No data downloaded.")

    # Flatten columns if multi-level
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.levels[0]:
            prices = df["Close"]
        else:
            raise ValueError(f"Could not find 'Close' in yfinance output columns: {df.columns.levels[0]}")
    else:
        prices = df

    prices = prices.dropna()

    if prices.empty:
        raise ValueError("Price dataframe is empty after dropping NAs. Some tickers might not have overlap.")

    print("Calculating log returns...")
    log_returns = np.log(prices / prices.shift(1)).dropna()

    # 2. Core Computations
    print("Calculating rolling correlations...")
    target_returns = log_returns[target]
    asset_returns = log_returns[assets]

    # Calculate 60-day and 120-day rolling correlation
    rolling_corr_60 = asset_returns.rolling(window=60).corr(target_returns)
    rolling_corr_120 = asset_returns.rolling(window=120).corr(target_returns)

    rolling_mean_corr_60 = rolling_corr_60.mean(axis=1)
    rolling_mean_corr_120 = rolling_corr_120.mean(axis=1)

    # Stress periods
    stress_periods = {
        "COVID Crash": ("2020-02-19", "2020-03-23"),
        "2022 Rate Shock": ("2022-01-03", "2022-06-16"),
        "2018 Vol Shock": ("2018-01-26", "2018-02-08"),
    }

    # Verify which periods are actually in our data
    valid_stress_periods = {}
    for name, (start, end) in stress_periods.items():
        if pd.to_datetime(start) >= log_returns.index[0] and pd.to_datetime(end) <= log_returns.index[-1]:
            valid_stress_periods[name] = (start, end)

    # 3. Correlation Summary Statistics Table
    print("Computing summary statistics...")
    stats_list = []

    for asset in assets:
        asset_corr = rolling_corr_60[asset].dropna()
        full_avg = asset_corr.mean()

        asset_stats = {"ETF": asset, "Full_Period_Avg_Corr": full_avg}

        for name, (start, end) in valid_stress_periods.items():
            stress_mask = (asset_corr.index >= start) & (asset_corr.index <= end)
            stress_corr = asset_corr[stress_mask]

            if not stress_corr.empty:
                stress_avg = stress_corr.mean()
                stress_max = stress_corr.max()
                delta = stress_avg - full_avg
            else:
                stress_avg = np.nan
                stress_max = np.nan
                delta = np.nan

            # Rename for compactness in print
            short_name = name.split()[0]
            asset_stats[f"{short_name}_Avg"] = stress_avg
            asset_stats[f"{short_name}_Max"] = stress_max
            asset_stats[f"{short_name}_Delta"] = delta

        stats_list.append(asset_stats)

    stats_df = pd.DataFrame(stats_list).set_index("ETF")
    stats_df.to_csv(os.path.join(out_dir, "correlation_stats.csv"))
    print("\n--- Correlation Statistics ---")
    print(stats_df.round(3).to_string())

    # Plotting Helper
    def add_stress_shading(ax):
        colors = ["red", "orange", "purple"]
        for i, (name, (start, end)) in enumerate(valid_stress_periods.items()):
            start_dt = pd.to_datetime(start)
            end_dt = pd.to_datetime(end)
            ax.axvspan(start_dt, end_dt, color=colors[i % len(colors)], alpha=0.2, label=name)

    # 4. Visualizations
    print("\nGenerating visualizations...")

    # Plot 1: Individual Subplots
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, asset in enumerate(assets):
        ax = axes[i]
        ax.plot(rolling_corr_60.index, rolling_corr_60[asset], color="b", linewidth=1.5)
        ax.axhline(1.0, color="k", linestyle="--", linewidth=1)
        ax.set_title(f"{asset} vs {target}")
        ax.set_ylim(-1, 1.1)
        add_stress_shading(ax)
        ax.grid(True, alpha=0.3)
        if i == 0:
            _dedupe_legend(ax, loc="lower left", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "rolling_correlation_individual.png"), dpi=300)
    plt.close()

    # Plot 2: Overlay
    plt.figure(figsize=(14, 7))
    for asset in assets:
        plt.plot(rolling_corr_60.index, rolling_corr_60[asset], label=asset, alpha=0.7)

    plt.axhline(1.0, color="k", linestyle="--", linewidth=1)
    ax = plt.gca()
    add_stress_shading(ax)
    _dedupe_legend(ax, loc="center left", bbox_to_anchor=(1.01, 0.5), borderaxespad=0.0)
    plt.title(f"60-Day Rolling Pairwise Correlations (All ETFs vs {target})")
    plt.xlabel("Date")
    plt.ylabel("Pearson Correlation")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "rolling_correlation_overlay.png"), dpi=300)
    plt.close()

    # Plot 3: Mean Correlation
    plt.figure(figsize=(14, 7))
    plt.plot(
        rolling_mean_corr_60.index,
        rolling_mean_corr_60,
        label="60-Day Mean Cross-Correlation",
        color="navy",
        linewidth=2,
    )
    plt.plot(
        rolling_mean_corr_120.index,
        rolling_mean_corr_120,
        label="120-Day Mean Cross-Correlation",
        color="darkorange",
        linewidth=2,
    )

    ax = plt.gca()
    add_stress_shading(ax)
    _dedupe_legend(ax, loc="upper left")

    plt.title(f"Average Liquid Basket Correlation to {target} Over Time")
    plt.xlabel("Date")
    plt.ylabel("Mean Pearson Correlation")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "rolling_mean_correlation.png"), dpi=300)
    plt.close()

    print("\n--- Key Findings & Interpretation Guidance ---")
    print("1. Do correlations spike toward 1.0 during stress?")
    print("   Look at the 'rolling_mean_correlation.png' and the 'COVID Crash' delta in the stats table.")
    print("   If correlations spike, the diversification benefits of the replicator diminish right when needed.")
    print("2. Which ETFs are most/least stable?")
    print("   Look at the variance in the individual subplots and the 'Full_Period_Avg_Corr'. Assets like GLD or TLT")
    print("   often provide true diversification (lower correlation), while broad equities (SPY, QQQ) track tightly.")
    print("3. Does the 2022 rate shock differ from COVID?")
    print("   COVID was a sharp liquidity event driving everything down together (correlations jump).")
    print("   2022 was an inflation/rate shock, which can cause bonds (AGG, TLT) to drop alongside equities,")
    print("   drastically changing the standard correlation regimes.")
    print("\nAnalysis complete. Visualizations and stats saved to 'outputs/'.")


if __name__ == "__main__":
    main()
