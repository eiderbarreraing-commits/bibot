
import pandas as pd
import ta
import matplotlib.pyplot as plt

# Cargar datos históricos
# Puedes cambiar este archivo por uno descargado desde Binance
HISTORICAL_DATA = "historical_pepeusdt.csv"

try:
    df = pd.read_csv(HISTORICAL_DATA)
except FileNotFoundError:
    print(f"❌ No se encontró el archivo {HISTORICAL_DATA}")
    # Create realistic dummy data with price movements for backtesting
    import numpy as np
    np.random.seed(42)  # For reproducible results
    
    periods = 200
    base_price = 1000
    dates = pd.date_range('2024-01-01', periods=periods, freq='1h')
    
    # Generate realistic price movements
    price_changes = np.random.normal(0, 0.02, periods)  # 2% volatility
    prices = [base_price]
    
    for change in price_changes[1:]:
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 1))  # Prevent negative prices
    
    # Create OHLC data
    df = pd.DataFrame({
        'timestamp': dates,
        'close': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],  # High slightly above close
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices]    # Low slightly below close
    })
    
    print(f"✅ Datos simulados creados: {len(df)} períodos con precios desde {df['close'].min():.2f} hasta {df['close'].max():.2f}")

# Convertir columnas necesarias
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['close'] = pd.to_numeric(df['close'])
df['high'] = pd.to_numeric(df['high'])
df['low'] = pd.to_numeric(df['low'])

def run_backtesting():
    """Run backtesting analysis on the loaded data"""
    global df
    
    # Aplicar indicadores
    df["rsi"] = ta.momentum.RSIIndicator(close=df["close"]).rsi()
    df["ema_fast"] = ta.trend.EMAIndicator(close=df["close"], window=9).ema_indicator()
    df["ema_slow"] = ta.trend.EMAIndicator(close=df["close"], window=21).ema_indicator()
    df["adx"] = ta.trend.ADXIndicator(high=df["high"], low=df["low"], close=df["close"]).adx()
    bb = ta.volatility.BollingerBands(close=df["close"])
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()

    # Backtesting
    initial_capital = 100
    capital = initial_capital
    position = 0
    entry_price = 0
    take_profit_pct = 0.06
    stop_loss_pct = 0.03
    trades = []

    for i in range(2, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]

        if position == 0:
            # Estrategia más sensible con múltiples señales
            
            # Señal 1: Breakout de Bollinger Bands (original)
            bollinger_breakout = (
                prev["close"] < prev["bb_upper"] and
                row["close"] > row["bb_upper"] and
                row["adx"] > 15  # Reducido de 20 a 15
            )
            
            # Señal 2: Cruce de EMAs alcista
            ema_cross = (
                prev["ema_fast"] <= prev["ema_slow"] and
                row["ema_fast"] > row["ema_slow"]
            )
            
            # Señal 3: RSI oversold recovery
            rsi_signal = (
                prev["rsi"] < 35 and row["rsi"] > 35 and
                row["close"] > prev["close"]
            )
            
            # Señal 4: Simple price momentum
            momentum_signal = (
                row["close"] > prev["close"] * 1.01 and  # 1% price increase
                row["ema_fast"] > row["ema_slow"]
            )
            
            # Ejecutar si cualquier señal se activa
            if bollinger_breakout or ema_cross or rsi_signal or momentum_signal:
                entry_price = row["close"]
                position = capital / entry_price
                capital = 0
                trades.append({"timestamp": row["timestamp"], "action": "BUY", "price": entry_price})
        else:
            change_pct = (row["close"] - entry_price) / entry_price
            if change_pct >= take_profit_pct or change_pct <= -stop_loss_pct:
                exit_price = row["close"]
                capital = position * exit_price
                result_pct = (exit_price - entry_price) / entry_price * 100
                trades.append({"timestamp": row["timestamp"], "action": "SELL", "price": exit_price, "result_%": result_pct})
                position = 0

    # Mostrar resultados
    if trades:
        df_trades = pd.DataFrame(trades)
        if not df_trades.empty and "timestamp" in df_trades.columns:
            df_trades["timestamp"] = pd.to_datetime(df_trades["timestamp"])

            print("📊 Resultados del backtesting:")
            print(df_trades)

            # Gráfico de operaciones
            buy_signals = df_trades[df_trades["action"] == "BUY"]
            sell_signals = df_trades[df_trades["action"] == "SELL"]

            # Crear gráfico (solo para análisis local, no para interfaz web)
            try:
                plt.figure(figsize=(12,6))
                plt.plot(df["timestamp"], df["close"], label="Precio", alpha=0.6)
                plt.scatter(buy_signals["timestamp"], buy_signals["price"], color="green", label="Buy", marker="^")
                plt.scatter(sell_signals["timestamp"], sell_signals["price"], color="red", label="Sell", marker="v")
                plt.title("Backtesting de la estrategia")
                plt.xlabel("Fecha")
                plt.ylabel("Precio")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                # plt.show()  # Comentado para evitar errores en servidor web
                plt.close()  # Liberar memoria
            except Exception as e:
                print(f"No se pudo generar el gráfico: {e}")

            total_return = capital - initial_capital
            print(f"🔚 Capital final: {capital:.2f} USDT")
            print(f"💹 Retorno total: {total_return:.2f} USDT")
            return df_trades, capital, total_return
    
    print("⚠️ No se ejecutaron operaciones en el backtesting.")
    return None, initial_capital, 0

# Only run backtesting if this file is executed directly
if __name__ == "__main__":
    run_backtesting()
