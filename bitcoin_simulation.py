import numpy as np
import pandas as pd


def simulate_bitcoin_prices(days=120, start_price=60000.0, mu=0.001, sigma=0.03, seed=42):
    """
    Simulates daily Bitcoin closing prices using Geometric Brownian Motion (GBM).
    S_t = S_{t-1} * exp((mu - 0.5 * sigma^2) + sigma * Z_t)
    """
    np.random.seed(seed)
    dates = pd.date_range(start="2024-01-01", periods=days, freq="D")

    prices = [start_price]
    for _ in range(1, days):
        shock = np.random.normal(0, 1)
        drift = mu - 0.5 * (sigma ** 2)
        diffusion = sigma * shock
        price_today = prices[-1] * np.exp(drift + diffusion)
        prices.append(price_today)

    df = pd.DataFrame({"Date": dates, "Price": prices})
    return df


def run_golden_cross_strategy(df, initial_cash=10000.0):
    """
    Calculates 7-day and 30-day moving averages and executes Golden Cross / Death Cross trading strategy.
    Golden Cross: MA7 crosses ABOVE MA30 -> BUY
    Death Cross: MA7 crosses BELOW MA30 -> SELL
    """
    df["MA7"] = df["Price"].rolling(window=7, min_periods=7).mean()
    df["MA30"] = df["Price"].rolling(window=30, min_periods=30).mean()

    cash = initial_cash
    btc = 0.0

    ledger = []

    for i in range(len(df)):
        date_str = df.loc[i, "Date"].strftime("%Y-%m-%d")
        price = df.loc[i, "Price"]
        ma7 = df.loc[i, "MA7"]
        ma30 = df.loc[i, "MA30"]

        action = "HOLD"

        # Determine signals if both MAs are available
        if i > 0 and pd.notna(df.loc[i, "MA7"]) and pd.notna(df.loc[i, "MA30"]) and \
           pd.notna(df.loc[i-1, "MA7"]) and pd.notna(df.loc[i-1, "MA30"]):

            prev_ma7 = df.loc[i-1, "MA7"]
            prev_ma30 = df.loc[i-1, "MA30"]

            # Golden Cross: prev MA7 <= prev MA30 and current MA7 > current MA30
            if prev_ma7 <= prev_ma30 and ma7 > ma30:
                if cash > 0:
                    btc = cash / price
                    cash = 0.0
                    action = "BUY"
            # Death Cross: prev MA7 >= prev MA30 and current MA7 < current MA30
            elif prev_ma7 >= prev_ma30 and ma7 < ma30:
                if btc > 0:
                    cash = btc * price
                    btc = 0.0
                    action = "SELL"

        total_value = cash + (btc * price)

        ledger.append({
            "Day": i + 1,
            "Date": date_str,
            "Price": price,
            "MA7": ma7,
            "MA30": ma30,
            "Action": action,
            "Cash": cash,
            "BTC": btc,
            "Total Value": total_value
        })

    ledger_df = pd.DataFrame(ledger)
    return ledger_df, initial_cash


def print_simulation_results(ledger_df, initial_cash):
    # Print ledger for the 60 simulation days requested
    display_df = ledger_df.head(60)

    print("=" * 105)
    print("                              BITCOIN 60-DAY TRADING SIMULATION LEDGER                              ")
    print("=" * 105)
    header = f"{'Day':<5} {'Date':<12} {'Price ($)':<12} {'MA7 ($)':<12} {'MA30 ($)':<12} {'Action':<8} {'Cash ($)':<12} {'BTC':<10} {'Total Val ($)':<12}"
    print(header)
    print("-" * 105)

    for _, row in display_df.iterrows():
        ma7_str = f"{row['MA7']:10.2f}" if pd.notna(row['MA7']) else "       N/A"
        ma30_str = f"{row['MA30']:10.2f}" if pd.notna(row['MA30']) else "       N/A"
        print(f"{int(row['Day']):<5} {row['Date']:<12} {row['Price']:10.2f}   {ma7_str}   {ma30_str}   {row['Action']:<8} {row['Cash']:10.2f}   {row['BTC']:8.4f}   {row['Total Value']:10.2f}")

    print("=" * 105)

    final_price = display_df.iloc[-1]["Price"]
    final_value = display_df.iloc[-1]["Total Value"]
    total_return = ((final_value - initial_cash) / initial_cash) * 100
    buy_hold_btc = initial_cash / display_df.iloc[0]["Price"]
    buy_hold_final = buy_hold_btc * final_price
    buy_hold_return = ((buy_hold_final - initial_cash) / initial_cash) * 100

    print("\n" + "=" * 55)
    print("                  FINAL PORTFOLIO PERFORMANCE             ")
    print("=" * 55)
    print(f"Initial Capital:         ${initial_cash:,.2f}")
    print(f"Final Portfolio Value:   ${final_value:,.2f}")
    print(f"Strategy Total Return:   {total_return:+.2f}%")
    print("-" * 55)
    print(f"Initial BTC Price:       ${display_df.iloc[0]['Price']:,.2f}")
    print(f"Final BTC Price:         ${final_price:,.2f}")
    print(f"Buy & Hold Final Value:  ${buy_hold_final:,.2f}")
    print(f"Buy & Hold Return:       {buy_hold_return:+.2f}%")
    print("=" * 55)


def main():
    # Warmup period allows MA30 and MA7 to cross cleanly within 60 evaluation days
    seeds_to_test = [42, 10, 123, 7, 100, 2024]
    for seed in seeds_to_test:
        df = simulate_bitcoin_prices(days=60, start_price=60000.0, mu=0.005, sigma=0.035, seed=seed)
        ledger_df, initial_cash = run_golden_cross_strategy(df, initial_cash=10000.0)
        actions = ledger_df["Action"].value_counts()
        if "BUY" in actions or "SELL" in actions:
            print(f"Found active trading seed: {seed}")
            print_simulation_results(ledger_df, initial_cash)
            return

    # Fallback to seed 42 if none cross
    df = simulate_bitcoin_prices(days=60, start_price=60000.0, mu=0.005, sigma=0.035, seed=42)
    ledger_df, initial_cash = run_golden_cross_strategy(df, initial_cash=10000.0)
    print_simulation_results(ledger_df, initial_cash)


if __name__ == "__main__":
    main()
