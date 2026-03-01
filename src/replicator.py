import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple
import cvxpy as cp
from datetime import datetime, timedelta


class SyntheticLiabilityReplicator:
    def __init__(self, assets: List[str], target: str, start_date: str, end_date: str):
        self.assets = assets
        self.target = target
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.returns = None
        self.weights_history = None

    def fetch_data(self):
        """Fetches historical adjusted close prices for assets and target."""
        all_tickers = self.assets + [self.target]
        print(f"Fetching data for: {', '.join(all_tickers)}")
        df = yf.download(all_tickers, start=self.start_date, end=self.end_date, auto_adjust=True)

        if df.empty:
            raise ValueError("No data downloaded. Check your tickers and network connection.")
        if isinstance(df.columns, pd.MultiIndex) and "Close" in df.columns.levels[0]:
            self.data = df["Close"].copy()
        elif not isinstance(df.columns, pd.MultiIndex):
            self.data = df.copy()
        else:
            raise KeyError(f"Could not find 'Close' in columns: {df.columns}")

        self.data = self.data.dropna()
        if self.data.empty:
            raise ValueError("Dataframe is empty after dropping NAs. Tickers might have non-overlapping history.")

        self.returns = self.data.pct_change().dropna()
        return self.returns

    def get_asset_target_split(self, returns_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Splits the returns into assets and target."""
        asset_returns = returns_df[self.assets]
        target_returns = returns_df[self.target]
        return asset_returns, target_returns

    def optimize_tracking_error(
        self,
        asset_returns: pd.DataFrame,
        target_returns: pd.Series,
        cvar_constraint_ratio: float = 1.0,
        w_prev: np.ndarray = None,
        max_weight: float = 0.25,
        max_turnover: float = 0.20,
    ) -> pd.DataFrame:
        """
        Minimizes Tracking Error using CVXPY subject to a hard CVaR constraint,
        Position limits (max_weight), and Turnover constraints (max_turnover).
        """
        R = asset_returns.values
        R_target = target_returns.values
        T, n_assets = R.shape
        alpha = 0.05

        # 1. Calculate Target CVaR (historical)
        k = int(alpha * T)
        if k == 0:
            k = 1
        sorted_target = np.sort(R_target)
        target_cvar = -np.mean(sorted_target[:k])
        limit_cvar = target_cvar * cvar_constraint_ratio

        # 2. Setup CVXPY variables
        w = cp.Variable(n_assets)
        v = cp.Variable()  # VaR
        z = cp.Variable(T)  # Auxiliary variables for CVaR

        # 3. Objective: Minimize Sum of Squared Tracking Error
        tracking_error = R @ w - R_target
        objective = cp.Minimize(cp.sum_squares(tracking_error))

        # 4. Constraints (Including Concentration / Position Limits)
        constraints = [
            cp.sum(w) == 1,
            w >= 0,
            w <= max_weight,
            z >= 0,
            z >= -(R @ w) - v,
            v + (1.0 / (T * alpha)) * cp.sum(z) <= limit_cvar,
        ]

        # Turnover Constraint
        if w_prev is not None:
            constraints.append(cp.norm(w - w_prev, 1) <= max_turnover)

        # 5. Solve
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve()  # Let CVXPY choose the solver automatically

            if w.value is None:
                # If constrained problem fails, fallback to unconstrained TE minimization (with pos/turnover limits)
                print("CVaR Constrained Optimization failed, falling back to unconstrained TE minimization.")
                fallback_constraints = [cp.sum(w) == 1, w >= 0, w <= max_weight]
                if w_prev is not None:
                    fallback_constraints.append(cp.norm(w - w_prev, 1) <= max_turnover)
                prob_unconstrained = cp.Problem(objective, fallback_constraints)
                prob_unconstrained.solve()

            w_val = np.where(w.value < 1e-4, 0.0, w.value)
            w_val /= w_val.sum()
            return pd.DataFrame(w_val, index=asset_returns.columns, columns=["weights"])
        except Exception as e:
            print(f"CVXPY Optimization Exception: {e}")
            return None

    def backtest_expanding_window(
        self, initial_train_size: int = 504, step: int = 126, max_weight: float = 0.25, max_turnover: float = 0.20
    ):
        """
        Backtests the strategy using an expanding window to avoid look-ahead bias.
        Incorporates turnover constraints and calculates weight drift between periods.
        """
        print(f"Starting expanding window backtest (initial={initial_train_size}, step={step})...")
        print(f"Constraints: Max Single Asset Weight={max_weight*100}%, Max Turnover={max_turnover*100}%")
        results_weights = []
        dates = []

        w_prev = None
        last_i = None
        n_obs = len(self.returns)

        for i in range(initial_train_size, n_obs, step):
            train_returns = self.returns.iloc[:i]
            asset_train, target_train = self.get_asset_target_split(train_returns)

            # Calculate drifted weights if w_prev exists
            if w_prev is not None and last_i is not None:
                period_asset_returns = self.returns.iloc[last_i:i][self.assets]
                compounded_returns = (1 + period_asset_returns).prod() - 1
                drifted = w_prev * (1 + compounded_returns.values)
                w_prev_drifted = drifted / drifted.sum()
            else:
                w_prev_drifted = None

            try:
                w_df = self.optimize_tracking_error(
                    asset_train, target_train, w_prev=w_prev_drifted, max_weight=max_weight, max_turnover=max_turnover
                )

                if w_df is not None:
                    w_prev = w_df["weights"].values
                    last_i = i
                    results_weights.append(w_df.T)
                    dates.append(self.returns.index[i])
                    print(f"Processed date: {self.returns.index[i].date()}")
                else:
                    print(f"Optimization returned None at {self.returns.index[i].date()}")
            except Exception as e:
                print(f"Optimization failed at {self.returns.index[i].date()}: {e}")

        if results_weights:
            self.weights_history = pd.concat(results_weights)
            self.weights_history.index = dates
        return self.weights_history

    def stress_test(self, weights: pd.Series, shock: float = -0.20):
        """
        Simulates a regime shift (e.g., 20% drop in target).
        """
        asset_returns, target_returns = self.get_asset_target_split(self.returns)
        combined = pd.concat([asset_returns, target_returns], axis=1)
        cov_matrix = np.cov(combined.values, rowvar=False, ddof=1)
        target_var = cov_matrix[-1, -1]
        betas = pd.Series(cov_matrix[:-1, -1] / target_var, index=self.assets)
        portfolio_beta = (weights * betas).sum()

        portfolio_impact = portfolio_beta * shock
        target_impact = shock

        return {
            "Shock Size": shock,
            "Portfolio Impact": portfolio_impact,
            "Target Impact": target_impact,
            "Relative Performance": portfolio_impact - target_impact,
        }

    def plot_results(self, final_weights: pd.Series):
        """Generates plots for the replicator."""
        # Correlation Heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.returns.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Asset-Target Correlation Heatmap")
        plt.savefig("correlation_heatmap.png")
        plt.close()

        # Cumulative Returns
        asset_returns, target_returns = self.get_asset_target_split(self.returns)
        portfolio_returns = (asset_returns * final_weights).sum(axis=1)

        cum_portfolio = (1 + portfolio_returns).cumprod()
        cum_target = (1 + target_returns).cumprod()

        plt.figure(figsize=(12, 6))
        cum_portfolio.plot(label="Synthetic Replicator (Optimized)")
        cum_target.plot(label=f"Target Index ({self.target})")

        if self.weights_history is not None:
            bt_returns = []
            for i in range(len(self.weights_history) - 1):
                start_date = self.weights_history.index[i]
                end_date = self.weights_history.index[i + 1]
                w = self.weights_history.iloc[i]
                period_slice = asset_returns.loc[start_date:end_date]
                if i > 0:
                    period_slice = period_slice.iloc[1:]
                if not period_slice.empty:
                    period_returns = (period_slice * w).sum(axis=1)
                    bt_returns.append(period_returns)
            if bt_returns:
                bt_returns_series = pd.concat(bt_returns)
                cum_bt = (1 + bt_returns_series).cumprod()
                # Align starting point for visual comparison
                offset = cum_target.loc[bt_returns_series.index[0]] / cum_bt.iloc[0]
                (cum_bt * offset).plot(label="Replicator (Backtested)", linestyle="--")

        plt.title("Cumulative Returns: Replicator vs Target")
        plt.legend()
        plt.grid(True)
        plt.savefig("cumulative_returns.png")
        plt.close()
        print("Plots saved to 'correlation_heatmap.png' and 'cumulative_returns.png'.")


def main():
    # Diversified asset list to avoid multicollinearity (removed IVV and VTI which redundant with SPY)
    assets = ["SPY", "QQQ", "VEA", "VWO", "BND", "AGG", "LQD", "TIP", "GLD", "VNQ"]
    target = "PSP"

    start_date = (datetime.now() - timedelta(days=5 * 365)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")

    replicator = SyntheticLiabilityReplicator(assets, target, start_date, end_date)
    replicator.fetch_data()

    asset_returns, target_returns = replicator.get_asset_target_split(replicator.returns)

    print("\n--- Running Expanding Window Backtest ---")
    # We run the backtest first now so we can see the realistic historical performance with turnover limits
    replicator.backtest_expanding_window(max_weight=0.25, max_turnover=0.20)

    print("\n--- Running Final Optimization ---")
    # For the final static optimization, we'll apply position limits, but skip turnover
    # since we just want to see the current optimal "ideal" portfolio if starting today.
    final_weights_df = replicator.optimize_tracking_error(asset_returns, target_returns, max_weight=0.25)

    if final_weights_df is None or final_weights_df.empty:
        print("Optimization failed! Check parameters or data.")
        return

    print("\nFinal Allocation (Max 25% per asset):")
    print(final_weights_df)

    print("\n--- Stress Test (Regime Shift: -20%) ---")
    stress_results = replicator.stress_test(final_weights_df["weights"], shock=-0.20)
    for k, v in stress_results.items():
        print(f"{k}: {v:.4f}")

    replicator.plot_results(final_weights_df["weights"])


if __name__ == "__main__":
    main()
