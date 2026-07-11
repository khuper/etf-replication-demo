import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import List, Optional, Tuple
import cvxpy as cp
from datetime import datetime, timedelta


@dataclass(frozen=True)
class BacktestResult:
    """Complete, out-of-sample output from a walk-forward experiment."""

    weights: pd.DataFrame
    returns: pd.DataFrame
    turnover: pd.Series
    metrics: dict[str, float]


def _annualized_return(returns: pd.Series) -> float:
    if returns.empty:
        return float("nan")
    return float((1 + returns).prod() ** (252 / len(returns)) - 1)


def _max_drawdown(returns: pd.Series) -> float:
    wealth = (1 + returns).cumprod()
    return float((wealth / wealth.cummax() - 1).min())


def calculate_metrics(returns: pd.DataFrame, turnover: pd.Series) -> dict[str, float]:
    """Calculate the small set of metrics shown by the terminal report."""
    replicator = returns["replicator"]
    target = returns["target"]
    active = replicator - target
    return {
        "tracking_error": float(active.std(ddof=1) * np.sqrt(252)),
        "correlation": float(replicator.corr(target)),
        "replicator_return": _annualized_return(replicator),
        "target_return": _annualized_return(target),
        "replicator_volatility": float(replicator.std(ddof=1) * np.sqrt(252)),
        "max_drawdown": _max_drawdown(replicator),
        "average_turnover": float(turnover.mean()),
        "total_cost": float(returns["cost"].sum()),
    }


class SyntheticLiabilityReplicator:
    def __init__(self, assets: List[str], target: str, start_date: str, end_date: str):
        self.assets = assets
        self.target = target
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.returns = None
        self.weights_history = None
        self.backtest_result = None

    def fetch_data(self):
        """Fetches historical adjusted close prices for assets and target."""
        all_tickers = self.assets + [self.target]
        df = yf.download(all_tickers, start=self.start_date, end=self.end_date, auto_adjust=True, progress=False)

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
        cvar_constraint_ratio: Optional[float] = 1.0,
        w_prev: Optional[np.ndarray] = None,
        max_weight: float = 0.25,
        max_turnover: float = 0.20,
    ) -> pd.DataFrame:
        """
        Minimizes Tracking Error using CVXPY subject to a hard CVaR constraint,
        Position limits (max_weight), and Turnover constraints (max_turnover).
        """
        if asset_returns.empty or target_returns.empty:
            raise ValueError("Optimization requires non-empty return history.")
        if len(asset_returns) != len(target_returns):
            raise ValueError("Asset and target returns must have the same length.")
        if asset_returns.shape[1] * max_weight < 1 - 1e-9:
            raise ValueError("The max-weight constraint is infeasible for this number of assets.")

        R = asset_returns.to_numpy(dtype=float)
        R_target = target_returns.to_numpy(dtype=float)
        if not np.isfinite(R).all() or not np.isfinite(R_target).all():
            raise ValueError("Optimization inputs contain missing or infinite values.")
        T, n_assets = R.shape
        alpha = 0.05

        # 2. Setup CVXPY variables
        w = cp.Variable(n_assets)

        # 3. Objective: Minimize Sum of Squared Tracking Error
        tracking_error = R @ w - R_target
        objective = cp.Minimize(cp.sum_squares(tracking_error))

        # 4. Constraints (Including Concentration / Position Limits)
        constraints = [
            cp.sum(w) == 1,
            w >= 0,
            w <= max_weight,
        ]

        if cvar_constraint_ratio is not None:
            k = max(1, int(alpha * T))
            # CVaR is a loss measure. A sample whose worst observations are all
            # gains has zero observed loss, not a negative risk budget.
            target_cvar = max(0.0, float(-np.mean(np.sort(R_target)[:k])))
            limit_cvar = target_cvar * cvar_constraint_ratio
            v = cp.Variable()
            z = cp.Variable(T)
            constraints.extend(
                [
                    z >= 0,
                    z >= -(R @ w) - v,
                    v + (1.0 / (T * alpha)) * cp.sum(z) <= limit_cvar,
                ]
            )

        # Turnover Constraint
        if w_prev is not None:
            constraints.append(cp.norm(w - w_prev, 1) <= max_turnover)

        # 5. Solve
        prob = cp.Problem(objective, constraints)
        # Pin the solver so feasibility behavior is stable across environments.
        # CLARABEL ships with CVXPY and handles both the quadratic objective and
        # the optional CVaR/turnover cone constraints.
        prob.solve(solver=cp.CLARABEL)
        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or w.value is None:
            raise RuntimeError(f"Optimization failed with status: {prob.status}")

        w_val = np.asarray(w.value, dtype=float).reshape(-1)
        tolerance = 1e-5
        if not np.isfinite(w_val).all():
            raise RuntimeError("Optimizer returned non-finite weights.")
        if abs(w_val.sum() - 1) > tolerance:
            raise RuntimeError("Optimizer returned weights that do not sum to one.")
        if w_val.min() < -tolerance or w_val.max() > max_weight + tolerance:
            raise RuntimeError(
                "Optimizer returned weights outside the requested bounds "
                f"(min={w_val.min():.8f}, max={w_val.max():.8f})."
            )
        if w_prev is not None and np.abs(w_val - w_prev).sum() > max_turnover + tolerance:
            raise RuntimeError("Optimizer returned weights outside the turnover constraint.")
        w_val[np.abs(w_val) < 1e-10] = 0.0
        return pd.DataFrame(w_val, index=asset_returns.columns, columns=["weights"])

    def run_backtest(
        self,
        initial_train_size: int = 504,
        step: int = 126,
        max_weight: float = 0.25,
        max_turnover: float = 0.20,
        transaction_cost_bps: float = 5.0,
        cvar_constraint_ratio: Optional[float] = None,
    ) -> BacktestResult:
        """Run a genuine walk-forward simulation with drifting holdings and costs."""
        if self.returns is None:
            raise ValueError("Fetch or assign return data before running a backtest.")
        if len(self.returns) <= initial_train_size:
            raise ValueError(
                f"Need more than {initial_train_size} observations; only {len(self.returns)} are available."
            )
        if step < 1:
            raise ValueError("Rebalance step must be positive.")
        if transaction_cost_bps < 0:
            raise ValueError("Transaction costs cannot be negative.")

        weights_records = []
        turnover_records = []
        return_records = []
        drifted_weights = None

        for i in range(initial_train_size, len(self.returns), step):
            asset_train, target_train = self.get_asset_target_split(self.returns.iloc[:i])
            optimized = self.optimize_tracking_error(
                asset_train,
                target_train,
                cvar_constraint_ratio=cvar_constraint_ratio,
                w_prev=drifted_weights,
                max_weight=max_weight,
                max_turnover=max_turnover,
            )["weights"].to_numpy()

            rebalance_date = self.returns.index[i]
            one_way_turnover = 1.0 if drifted_weights is None else float(np.abs(optimized - drifted_weights).sum() / 2)
            weights_records.append(pd.Series(optimized, index=self.assets, name=rebalance_date))
            turnover_records.append((rebalance_date, one_way_turnover))

            current_weights = optimized.copy()
            period = self.returns.iloc[i : min(i + step, len(self.returns))]
            rebalance_cost = one_way_turnover * transaction_cost_bps / 10_000
            for day_number, (day, row) in enumerate(period.iterrows()):
                asset_day = row[self.assets].to_numpy(dtype=float)
                gross_return = float(current_weights @ asset_day)
                cost = rebalance_cost if day_number == 0 else 0.0
                net_return = gross_return - cost
                target_return = float(row[self.target])
                return_records.append(
                    {
                        "date": day,
                        "replicator": net_return,
                        "target": target_return,
                        "active": net_return - target_return,
                        "cost": cost,
                    }
                )
                denominator = 1 + gross_return
                if denominator <= 0:
                    raise RuntimeError(f"Portfolio value became non-positive on {day}.")
                current_weights = current_weights * (1 + asset_day) / denominator
            drifted_weights = current_weights

        weights = pd.DataFrame(weights_records)
        returns = pd.DataFrame(return_records).set_index("date")
        turnover = pd.Series(dict(turnover_records), dtype=float)
        turnover.index.name = "rebalance_date"
        result = BacktestResult(weights, returns, turnover, calculate_metrics(returns, turnover))
        self.weights_history = weights
        self.backtest_result = result
        return result

    def backtest_expanding_window(
        self, initial_train_size: int = 504, step: int = 126, max_weight: float = 0.25, max_turnover: float = 0.20
    ):
        """
        Backtests the strategy using an expanding window to avoid look-ahead bias.
        Incorporates turnover constraints and calculates weight drift between periods.
        """
        result = self.run_backtest(
            initial_train_size=initial_train_size,
            step=step,
            max_weight=max_weight,
            max_turnover=max_turnover,
            transaction_cost_bps=0,
        )
        return result.weights

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
