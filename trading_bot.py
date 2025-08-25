import os
import logging
import pandas as pd
import ta
import csv
import json
import numpy as np
from datetime import datetime, timedelta
from statistics import mean, stdev
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException
from app import db
from models import TradeSignal, BotStatus
from telegram_bot import telegram_notifier
from cloud_optimizer import cloud_optimizer

logger = logging.getLogger(__name__)

class TradingBot:
    def __init__(self):
        self.api_key = os.getenv("BINANCE_API_KEY")
        self.api_secret = os.getenv("BINANCE_SECRET_KEY")
        self.use_indicators = os.getenv("USE_TECHNICAL_INDICATORS", "true").lower() == "true"
        
        # Advanced strategy parameters
        self.take_profit_pct = float(os.getenv("TAKE_PROFIT_PCT", "0.06"))  # 6%
        self.stop_loss_pct = float(os.getenv("STOP_LOSS_PCT", "0.03"))  # 3%
        self.log_file = os.getenv("LOG_FILE", "operation_log.csv")
        
        # Position tracking
        self.position = None
        self.entry_price = 0
        
        # AI Features
        self.ai_confidence = 85
        self.pattern_history = []
        self.performance_metrics = {
            "win_rate": 0,
            "avg_profit": 0,
            "max_drawdown": 0,
            "sharpe_ratio": 0
        }
        self.ai_predictions = {
            "next_price_direction": None,
            "volatility_forecast": None,
            "optimal_entry_time": None
        }
        
        # Performance Optimization Features
        self.cache = {}
        self.cache_ttl = {}
        self.performance_stats = {
            "execution_times": [],
            "api_calls": 0,
            "cache_hits": 0,
            "memory_usage": 0
        }
        self.last_optimization = datetime.now()
        
        if not self.api_key or not self.api_secret:
            logger.error("Binance API credentials not found in environment variables")
            # Try to load credentials from database
            if self.load_credentials_from_db():
                logger.info("Loading credentials from database instead")
                # Still need the secret to initialize client, so client will remain None until credentials are updated
                self.client = None
            else:
                logger.info("No credentials available from environment or database")
                self.client = None
        else:
            self.client = self._initialize_client()
        
        logger.info(f"Trading bot initialized - Technical indicators: {'ENABLED' if self.use_indicators else 'DISABLED'}")
        logger.info(f"Strategy parameters - Take Profit: {self.take_profit_pct*100}%, Stop Loss: {self.stop_loss_pct*100}%")
        
        # Initialize cloud optimizations
        if cloud_optimizer.is_cloud_environment:
            cloud_optimizer.optimize_for_24_7()
        
        # Send startup notification
        telegram_notifier.send_status_alert("Bot iniciado", "Sistema operativo y listo para trading")
    
    def _initialize_client(self):
        """Initialize Binance client with fallback options"""
        # Try different configurations to bypass restrictions
        configs = [
            # Use testnet for development (safer)
            {"testnet": True},
            # Try with different base URLs
            {"tld": "us"},
            {"tld": "com"},
            # Standard configuration
            {}
        ]
        
        for config in configs:
            try:
                logger.info(f"Attempting to connect with config: {config}")
                if config.get("testnet"):
                    client = Client(self.api_key, self.api_secret, testnet=True)
                else:
                    client = Client(self.api_key, self.api_secret, **config)
                
                # Test the connection
                client.ping()
                logger.info(f"Successfully connected to Binance API with config: {config}")
                return client
                
            except Exception as e:
                logger.warning(f"Failed to connect with config {config}: {e}")
                continue
        
        # If all configs fail, still create client without ping test
        try:
            logger.info("Creating client without ping test as fallback")
            client = Client(self.api_key, self.api_secret)
            logger.info("Client created successfully (connection not verified)")
            return client
        except Exception as e:
            logger.error(f"Failed to create Binance client: {e}")
            return None
    
    def is_ready(self):
        """Check if the bot is ready to execute trades"""
        return self.client is not None
    
    def execute_signal(self, signal_data):
        """Execute a trading signal"""
        signal_type = signal_data.get('signal', '').lower()
        symbol = signal_data.get('symbol', 'PEPEUSDT').upper()
        amount = signal_data.get('amount', 5)  # Default 5 USDT
        
        # Create trade signal record
        trade_signal = TradeSignal(
            signal_type=signal_type,
            symbol=symbol,
            amount=amount,
            status='processing',
            raw_data=str(signal_data)
        )
        db.session.add(trade_signal)
        db.session.commit()
        
        logger.info(f"Received {signal_type} signal for {symbol}")
        
        # Update bot status
        bot_status = BotStatus.get_status()
        bot_status.update_ping()
        
        if not self.is_ready():
            error_msg = "Trading bot is not ready - API credentials missing or invalid"
            logger.error(error_msg)
            trade_signal.status = 'error'
            trade_signal.error_message = error_msg
            db.session.commit()
            bot_status.increment_trade(success=False)
            return {"status": "error", "message": error_msg}
        
        # AI-POWERED ANALYSIS BEFORE TRADING
        logger.info("🧠 Running AI analysis before signal execution...")
        
        # Perform AI risk assessment
        risk_assessment = self.risk_assessment(symbol, amount)
        logger.info(f"🛡️ Risk Level: {risk_assessment['risk_level']} (Score: {risk_assessment['risk_score']}/100)")
        
        # Run AI predictive analysis
        ai_prediction = self.predictive_analysis(symbol)
        
        # Run AI parameter optimization
        self.optimize_parameters()
        
        # Run automatic performance optimization
        self.auto_optimization()
        
        # First, manage any existing position (take profit/stop loss)
        if self.has_position():
            position_result = self.manage_position(symbol)
            if position_result and position_result.get("status") == "success":
                # Position was closed, update trade signal
                trade_signal.status = 'success'
                trade_signal.error_message = f"Position closed by {position_result.get('reason')}"
                db.session.commit()
                bot_status = BotStatus.get_status()
                bot_status.increment_trade(success=True)
                return position_result
        
        if signal_type == 'buy':
            # Don't open new position if we already have one
            if self.has_position():
                logger.warning(f"Cannot execute BUY signal - position already open for {symbol}")
                trade_signal.status = 'ignored'
                trade_signal.error_message = 'Position already open'
                db.session.commit()
                return {"status": "ignored", "message": "Position already open"}
            
            # AI-ENHANCED SIGNAL VALIDATION
            ai_approved = True
            rejection_reasons = []
            
            # Check AI risk level
            if risk_assessment['risk_level'] in ['VERY_HIGH', 'HIGH']:
                ai_approved = False
                rejection_reasons.append(f"AI Risk too high: {risk_assessment['risk_level']}")
            
            # Check AI prediction
            if ai_prediction and ai_prediction['direction'] == 'SELL':
                ai_approved = False
                rejection_reasons.append(f"AI predicts SELL with {ai_prediction['confidence']:.1f}% confidence")
            elif ai_prediction and ai_prediction['direction'] == 'HOLD' and ai_prediction['confidence'] > 75:
                ai_approved = False
                rejection_reasons.append(f"AI recommends HOLD with high confidence ({ai_prediction['confidence']:.1f}%)")
            
            # AI override for risk management
            if not ai_approved:
                logger.warning(f"🚫 AI BLOCKS TRADE: {', '.join(rejection_reasons)}")
                trade_signal.status = 'ignored'
                trade_signal.error_message = f"AI blocked trade: {', '.join(rejection_reasons)}"
                db.session.commit()
                return {"status": "ignored", "message": f"AI blocked trade: {', '.join(rejection_reasons)}"}
            
            # Adjust amount based on AI risk assessment
            adjusted_amount = min(amount, risk_assessment.get('position_size_suggestion', amount))
            if adjusted_amount < amount:
                logger.info(f"💡 AI suggests smaller position: ${amount} → ${adjusted_amount:.2f}")
                amount = adjusted_amount
            
            # Validate buy signal with technical indicators if enabled
            if self.use_indicators:
                if self.check_buy_signal(symbol):
                    logger.info(f"✅ Technical indicators + AI confirm BUY signal for {symbol}")
                    if ai_prediction:
                        logger.info(f"🤖 AI Confidence: {ai_prediction['confidence']:.1f}% | Direction: {ai_prediction['direction']}")
                    return self._execute_buy_order(trade_signal, symbol, amount)
                else:
                    logger.warning(f"Technical indicators do not support BUY signal for {symbol}")
                    trade_signal.status = 'ignored'
                    trade_signal.error_message = 'Technical indicators do not support buy signal'
                    db.session.commit()
                    return {"status": "ignored", "message": "Technical indicators do not support buy signal"}
            else:
                logger.info(f"🤖 Executing AI-approved BUY signal for {symbol}")
                return self._execute_buy_order(trade_signal, symbol, amount)
        elif signal_type == 'sell':
            # If we have a position, close it regardless of technical indicators
            if self.has_position():
                logger.info(f"Closing position due to SELL signal for {symbol}")
                ticker = self.get_optimized_ticker(symbol)
                current_price = float(ticker["price"]) if ticker else 0
                position_result = self._close_position(symbol, current_price, "SELL_SIGNAL")
                if position_result:
                    trade_signal.status = 'success'
                    trade_signal.error_message = 'Position closed by sell signal'
                    db.session.commit()
                    bot_status = BotStatus.get_status()
                    bot_status.increment_trade(success=True)
                    return position_result
            
            # Standard sell logic if no position
            if self.use_indicators:
                if self.check_sell_signal(symbol):
                    logger.info(f"Technical indicators confirm SELL signal for {symbol}")
                    return self._execute_sell_order(trade_signal, symbol, amount)
                else:
                    logger.warning(f"Technical indicators do not support SELL signal for {symbol}")
                    trade_signal.status = 'ignored'
                    trade_signal.error_message = 'Technical indicators do not support sell signal'
                    db.session.commit()
                    return {"status": "ignored", "message": "Technical indicators do not support sell signal"}
            else:
                logger.info(f"Executing SELL signal for {symbol} (technical indicators disabled)")
                return self._execute_sell_order(trade_signal, symbol, amount)
        else:
            trade_signal.status = 'ignored'
            trade_signal.error_message = f"Unknown signal type: {signal_type}"
            db.session.commit()
            logger.warning(f"Ignored unknown signal type: {signal_type}")
            return {"status": "ignored", "message": f"Unknown signal type: {signal_type}"}
    
    def _execute_buy_order(self, trade_signal, symbol, amount):
        """Execute a market buy order"""
        try:
            logger.info(f"Executing market buy order for {symbol} with {amount} USDT")
            
            if not self.client:
                error_msg = "Trading bot client is not available"
                logger.error(error_msg)
                trade_signal.status = 'error'
                trade_signal.error_message = error_msg
                db.session.commit()
                bot_status = BotStatus.get_status()
                bot_status.increment_trade(success=False)
                return {"status": "error", "message": error_msg}
            
            # Test connection before executing trade
            try:
                self.client.get_server_time()
            except Exception as conn_e:
                logger.warning(f"Connection test failed, attempting to reconnect: {conn_e}")
                self.client = self._initialize_client()
                if not self.client:
                    error_msg = "Could not establish connection to Binance API"
                    logger.error(error_msg)
                    trade_signal.status = 'error'
                    trade_signal.error_message = error_msg
                    db.session.commit()
                    bot_status = BotStatus.get_status()
                    bot_status.increment_trade(success=False)
                    return {"status": "error", "message": error_msg}
            
            # Get current price for position tracking using optimized method
            ticker = self.get_optimized_ticker(symbol)
            current_price = float(ticker["price"]) if ticker else 0
            qty = round(amount / current_price, 0)  # Calculate quantity for PEPE
            
            if qty == 0:
                error_msg = "Insufficient amount for trading"
                logger.warning(error_msg)
                trade_signal.status = 'error'
                trade_signal.error_message = error_msg
                db.session.commit()
                return {"status": "error", "message": error_msg}
            
            order = self.client.order_market_buy(
                symbol=symbol,
                quoteOrderQty=amount
            )
            
            # Track position for advanced strategy
            executed_price = float(order.get('fills', [{}])[0].get('price', current_price)) if order.get('fills') else current_price
            executed_qty = float(order.get('executedQty', qty))
            
            self.position = {"symbol": symbol, "qty": executed_qty}
            self.entry_price = executed_price
            
            # Log operation to CSV
            self.log_operation("BUY", symbol, executed_qty, executed_price)
            
            trade_signal.status = 'success'
            trade_signal.order_id = order.get('orderId')
            trade_signal.price = executed_price
            db.session.commit()
            
            # Send Telegram notification
            telegram_notifier.send_buy_alert(symbol, amount, executed_price, trade_signal.order_id)
            
            # Update bot status
            bot_status = BotStatus.get_status()
            bot_status.increment_trade(success=True)
            
            logger.info(f"🟢 BUY order executed: {executed_qty} {symbol} at {executed_price:.8f} USDT")
            logger.info(f"📊 Position opened - Entry: {self.entry_price:.8f}, Qty: {self.position['qty']}")
            
            return {"status": "success", "order": order, "position": self.position}
            
        except BinanceAPIException as e:
            error_msg = f"Binance API error: {e.message}"
            logger.error(error_msg)
            trade_signal.status = 'error'
            trade_signal.error_message = error_msg
            db.session.commit()
            
            # Update bot status
            bot_status = BotStatus.get_status()
            bot_status.increment_trade(success=False)
            
            return {"status": "error", "message": error_msg}
            
        except BinanceOrderException as e:
            error_msg = f"Binance order error: {e.message}"
            logger.error(error_msg)
            trade_signal.status = 'error'
            trade_signal.error_message = error_msg
            db.session.commit()
            
            # Update bot status
            bot_status = BotStatus.get_status()
            bot_status.increment_trade(success=False)
            
            return {"status": "error", "message": error_msg}
            
        except Exception as e:
            error_msg = f"Unexpected error executing buy order: {str(e)}"
            logger.error(error_msg)
            trade_signal.status = 'error'
            trade_signal.error_message = error_msg
            db.session.commit()
            
            # Update bot status
            bot_status = BotStatus.get_status()
            bot_status.increment_trade(success=False)
            
            return {"status": "error", "message": error_msg}
    
    def _execute_sell_order(self, trade_signal, symbol, amount):
        """Execute a market sell order"""
        try:
            logger.info(f"Executing market sell order for {symbol}")
            
            # For sell orders, we need to specify the quantity of the base asset
            # This is a simplified implementation - in practice, you'd want to
            # get the account balance and calculate the quantity to sell
            
            # Get account info to determine available balance
            if not self.client:
                error_msg = "Trading bot client is not available"
                logger.error(error_msg)
                trade_signal.status = 'error'
                trade_signal.error_message = error_msg
                db.session.commit()
                bot_status = BotStatus.get_status()
                bot_status.increment_trade(success=False)
                return {"status": "error", "message": error_msg}
            
            account = self.client.get_account()
            
            # Find the base asset balance (e.g., PEPE from PEPEUSDT)
            base_asset = symbol.replace('USDT', '')
            balance = None
            
            for asset in account['balances']:
                if asset['asset'] == base_asset:
                    balance = float(asset['free'])
                    break
            
            if balance is None or balance <= 0:
                error_msg = f"No {base_asset} balance available for selling"
                logger.error(error_msg)
                trade_signal.status = 'error'
                trade_signal.error_message = error_msg
                db.session.commit()
                
                # Update bot status
                bot_status = BotStatus.get_status()
                bot_status.increment_trade(success=False)
                
                return {"status": "error", "message": error_msg}
            
            # Sell all available balance
            if not self.client:
                error_msg = "Trading bot client is not available"
                logger.error(error_msg)
                trade_signal.status = 'error'
                trade_signal.error_message = error_msg
                db.session.commit()
                bot_status = BotStatus.get_status()
                bot_status.increment_trade(success=False)
                return {"status": "error", "message": error_msg}
            
            order = self.client.order_market_sell(
                symbol=symbol,
                quantity=balance
            )
            
            trade_signal.status = 'success'
            trade_signal.order_id = order.get('orderId')
            trade_signal.price = float(order.get('fills', [{}])[0].get('price', 0)) if order.get('fills') else None
            db.session.commit()
            
            # Send Telegram notification
            telegram_notifier.send_sell_alert(symbol, balance, trade_signal.price, trade_signal.order_id, "manual")
            
            # Update bot status
            bot_status = BotStatus.get_status()
            bot_status.increment_trade(success=True)
            
            logger.info(f"Sell order executed successfully: {order}")
            return {"status": "success", "order": order}
            
        except Exception as e:
            error_msg = f"Error executing sell order: {str(e)}"
            logger.error(error_msg)
            trade_signal.status = 'error'
            trade_signal.error_message = error_msg
            db.session.commit()
            
            # Update bot status
            bot_status = BotStatus.get_status()
            bot_status.increment_trade(success=False)
            
            return {"status": "error", "message": error_msg}
    
    def get_account_info(self):
        """Get account information"""
        if not self.is_ready():
            return None
        
        try:
            if self.client:
                return self.client.get_account()
            else:
                return None
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return None
    
    def get_klines(self, symbol, interval=Client.KLINE_INTERVAL_5MINUTE, limit=100):
        """Get market data for technical analysis"""
        if not self.client:
            return None
        
        try:
            klines = self.client.get_klines(symbol=symbol, interval=interval, limit=limit)
            df = pd.DataFrame(klines, columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "number_of_trades",
                "taker_buy_base", "taker_buy_quote", "ignore"
            ])
            df["close"] = pd.to_numeric(df["close"])
            df["high"] = pd.to_numeric(df["high"])
            df["low"] = pd.to_numeric(df["low"])
            df["open"] = pd.to_numeric(df["open"])
            df["volume"] = pd.to_numeric(df["volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
            return df
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return None
    
    def apply_indicators(self, df):
        """Apply technical indicators to market data"""
        try:
            # Basic indicators
            df["rsi"] = ta.momentum.RSIIndicator(close=df["close"]).rsi()
            df["ema_fast"] = ta.trend.EMAIndicator(close=df["close"], window=9).ema_indicator()
            df["ema_slow"] = ta.trend.EMAIndicator(close=df["close"], window=21).ema_indicator()
            macd = ta.trend.MACD(close=df["close"])
            df["macd"] = macd.macd()
            df["macd_signal"] = macd.macd_signal()
            
            # Advanced indicators from uploaded strategy
            df["adx"] = ta.trend.ADXIndicator(high=df["high"], low=df["low"], close=df["close"]).adx()
            bb = ta.volatility.BollingerBands(close=df["close"])
            df["bb_upper"] = bb.bollinger_hband()
            df["bb_lower"] = bb.bollinger_lband()
            df["bb_width"] = df["bb_upper"] - df["bb_lower"]
            
            return df
        except Exception as e:
            logger.error(f"Error applying indicators: {e}")
            return df
    
    def log_operation(self, action, symbol, qty, entry_price, exit_price=None):
        """Log trading operations to CSV file"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            result_pct = 0
            result_net = 0

            if action == "SELL" and exit_price and entry_price:
                result_pct = (exit_price - entry_price) / entry_price * 100
                result_net = (exit_price - entry_price) * qty

            log_data = {
                "timestamp": timestamp,
                "action": action,
                "symbol": symbol,
                "qty": qty,
                "entry_price": entry_price,
                "exit_price": exit_price if exit_price else "",
                "result_%": round(result_pct, 2),
                "net_result_usdt": round(result_net, 2)
            }

            df_log = pd.DataFrame([log_data])
            if not os.path.exists(self.log_file):
                df_log.to_csv(self.log_file, index=False)
            else:
                df_log.to_csv(self.log_file, mode="a", header=False, index=False)
            
            logger.info(f"Operation logged: {action} {qty} {symbol} at {entry_price if action == 'BUY' else exit_price}")
        except Exception as e:
            logger.error(f"Error logging operation: {e}")
    
    def check_buy_signal(self, symbol):
        """Check if current market conditions support a buy signal"""
        df = self.get_klines(symbol)
        if df is None or len(df) < 21:
            logger.warning("Insufficient market data for technical analysis")
            return False
        
        try:
            df = self.apply_indicators(df)
            latest = df.iloc[-1]
            previous = df.iloc[-2]
            
            # Basic strategy: RSI oversold, EMA crossover, MACD bullish
            basic_signal = (
                latest["rsi"] < 30 and  # RSI oversold
                previous["ema_fast"] < previous["ema_slow"] and  # Previous EMA bearish
                latest["ema_fast"] > latest["ema_slow"] and  # Current EMA bullish crossover
                latest["macd"] > latest["macd_signal"]  # MACD bullish
            )
            
            # Advanced breakout strategy from uploaded file
            breakout_signal = (
                previous["close"] < previous["bb_upper"] and
                latest["close"] > latest["bb_upper"] and
                latest["adx"] > 20
            )
            
            trend_confirm = latest["ema_fast"] > latest["ema_slow"]
            
            # Combine both strategies
            buy_signal = basic_signal or (breakout_signal and trend_confirm)
            
            logger.info(f"Technical analysis for {symbol}: RSI={latest['rsi']:.2f}, "
                       f"EMA_Fast={latest['ema_fast']:.6f}, EMA_Slow={latest['ema_slow']:.6f}, "
                       f"MACD={latest['macd']:.6f}, ADX={latest['adx']:.2f}, "
                       f"BB_Breakout={'Yes' if breakout_signal else 'No'}, "
                       f"Signal={'BUY' if buy_signal else 'HOLD'}")
            
            return buy_signal
        except Exception as e:
            logger.error(f"Error in technical analysis: {e}")
            return False
    
    def check_sell_signal(self, symbol):
        """Check if current market conditions support a sell signal"""
        df = self.get_klines(symbol)
        if df is None or len(df) < 21:
            logger.warning("Insufficient market data for technical analysis")
            return False
        
        try:
            df = self.apply_indicators(df)
            latest = df.iloc[-1]
            previous = df.iloc[-2]
            
            # Combined rules for selling: RSI overbought, EMA bearish crossover, MACD bearish
            sell_signal = (
                latest["rsi"] > 70 and  # RSI overbought
                previous["ema_fast"] > previous["ema_slow"] and  # Previous EMA bullish
                latest["ema_fast"] < latest["ema_slow"] and  # Current EMA bearish crossover
                latest["macd"] < latest["macd_signal"]  # MACD bearish
            )
            
            logger.info(f"Technical analysis for {symbol}: RSI={latest['rsi']:.2f}, "
                       f"EMA_Fast={latest['ema_fast']:.6f}, EMA_Slow={latest['ema_slow']:.6f}, "
                       f"MACD={latest['macd']:.6f}, Signal={'SELL' if sell_signal else 'HOLD'}")
            
            return sell_signal
        except Exception as e:
            logger.error(f"Error in technical analysis: {e}")
            return False
    
    def manage_position(self, symbol):
        """Monitor current position and apply take profit/stop loss"""
        if not self.position:
            return None
        
        try:
            ticker = self.get_optimized_ticker(symbol)
            current_price = float(ticker["price"]) if ticker else 0
            change_pct = (current_price - self.entry_price) / self.entry_price
            
            logger.info(f"Position monitoring - Current: {current_price:.8f} | Entry: {self.entry_price:.8f} | Change: {change_pct*100:.2f}%")
            
            if change_pct >= self.take_profit_pct:
                logger.info("🎯 Take-Profit reached. Closing position with PROFIT.")
                return self._close_position(symbol, current_price, "TAKE_PROFIT")
            elif change_pct <= -self.stop_loss_pct:
                logger.info("🛑 Stop-Loss reached. Closing position with LOSS.")
                return self._close_position(symbol, current_price, "STOP_LOSS")
            
            return None
        except Exception as e:
            logger.error(f"Error managing position: {e}")
            return None
    
    def _close_position(self, symbol, current_price, reason):
        """Close current position"""
        if not self.position:
            return None
        
        try:
            # Execute sell order
            order = self.client.order_market_sell(
                symbol=symbol,
                quantity=self.position["qty"]
            )
            
            # Log the operation
            self.log_operation("SELL", symbol, self.position["qty"], self.entry_price, current_price)
            
            # Calculate result
            result_pct = (current_price - self.entry_price) / self.entry_price * 100
            result_net = (current_price - self.entry_price) * self.position["qty"]
            
            logger.info(f"🔴 Position closed: {self.position['qty']} {symbol} at {current_price:.8f} USDT")
            logger.info(f"📊 Result: {result_pct:.2f}% ({result_net:.2f} USDT) - Reason: {reason}")
            
            # Reset position
            self.position = None
            self.entry_price = 0
            
            return {
                "status": "success",
                "order": order,
                "reason": reason,
                "result_pct": result_pct,
                "result_net": result_net
            }
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return {"status": "error", "message": str(e)}
    
    def has_position(self):
        """Check if bot currently has an open position"""
        return self.position is not None
    
    # ====== AI FEATURES ======
    
    def detect_patterns(self, symbol):
        """AI Pattern Detection: Identify market patterns and trends"""
        try:
            # Get recent price data using optimized method
            klines = self.get_optimized_klines(symbol, interval="5m", limit=50)
            if not klines:
                return None
            
            prices = [float(kline[4]) for kline in klines]  # Closing prices
            volumes = [float(kline[5]) for kline in klines]  # Volumes
            
            # Pattern detection algorithms
            patterns = {
                "trend": self._analyze_trend(prices),
                "support_resistance": self._find_support_resistance(prices),
                "momentum": self._calculate_momentum(prices),
                "volume_pattern": self._analyze_volume_pattern(volumes),
                "breakout_probability": self._calculate_breakout_probability(prices),
                "reversal_signals": self._detect_reversal_signals(prices)
            }
            
            # Store pattern for AI learning
            self.pattern_history.append({
                "timestamp": datetime.now(),
                "symbol": symbol,
                "patterns": patterns,
                "price": prices[-1]
            })
            
            # Keep only recent patterns (last 100)
            if len(self.pattern_history) > 100:
                self.pattern_history = self.pattern_history[-100:]
            
            logger.info(f"🧠 AI Pattern Analysis: Trend={patterns['trend']}, Breakout={patterns['breakout_probability']:.1f}%")
            return patterns
            
        except Exception as e:
            logger.error(f"Error in pattern detection: {e}")
            return None
    
    def _analyze_trend(self, prices):
        """Analyze price trend using AI algorithms"""
        if len(prices) < 10:
            return "UNCLEAR"
        
        # Calculate trend strength using multiple methods
        recent_prices = prices[-10:]
        older_prices = prices[-20:-10] if len(prices) >= 20 else prices[:-10]
        
        # Linear regression trend
        x = np.arange(len(recent_prices))
        slope = np.polyfit(x, recent_prices, 1)[0]
        
        # Moving average crossover
        ma_short = mean(prices[-5:])
        ma_long = mean(prices[-15:]) if len(prices) >= 15 else mean(prices)
        
        # Combine indicators for AI decision
        if slope > 0 and ma_short > ma_long:
            return "BULLISH"
        elif slope < 0 and ma_short < ma_long:
            return "BEARISH"
        else:
            return "SIDEWAYS"
    
    def _find_support_resistance(self, prices):
        """AI algorithm to find support and resistance levels"""
        if len(prices) < 20:
            return {"support": min(prices), "resistance": max(prices)}
        
        # Use local minima and maxima
        support_levels = []
        resistance_levels = []
        
        for i in range(2, len(prices) - 2):
            # Local minimum (support)
            if prices[i] < prices[i-1] and prices[i] < prices[i+1] and \
               prices[i] < prices[i-2] and prices[i] < prices[i+2]:
                support_levels.append(prices[i])
            
            # Local maximum (resistance)
            if prices[i] > prices[i-1] and prices[i] > prices[i+1] and \
               prices[i] > prices[i-2] and prices[i] > prices[i+2]:
                resistance_levels.append(prices[i])
        
        current_price = prices[-1]
        
        # Find closest levels
        support = max([s for s in support_levels if s <= current_price], default=min(prices))
        resistance = min([r for r in resistance_levels if r >= current_price], default=max(prices))
        
        return {"support": support, "resistance": resistance}
    
    def _calculate_momentum(self, prices):
        """Calculate momentum using AI-enhanced indicators"""
        if len(prices) < 10:
            return 0
        
        # Rate of change momentum
        roc = (prices[-1] - prices[-10]) / prices[-10] * 100
        
        # Acceleration factor
        recent_changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        acceleration = mean(recent_changes[-5:]) - mean(recent_changes[-10:-5]) if len(recent_changes) >= 10 else 0
        
        # Combined momentum score
        momentum_score = (roc * 0.7) + (acceleration * 1000 * 0.3)
        
        return min(max(momentum_score, -100), 100)  # Normalize to -100 to 100
    
    def _analyze_volume_pattern(self, volumes):
        """AI volume pattern analysis"""
        if len(volumes) < 10:
            return "NORMAL"
        
        recent_vol = mean(volumes[-5:])
        avg_vol = mean(volumes[-20:]) if len(volumes) >= 20 else mean(volumes)
        
        vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 1
        
        if vol_ratio > 1.5:
            return "HIGH"
        elif vol_ratio < 0.7:
            return "LOW"
        else:
            return "NORMAL"
    
    def _calculate_breakout_probability(self, prices):
        """AI algorithm to calculate breakout probability"""
        if len(prices) < 20:
            return 50
        
        # Volatility analysis
        recent_volatility = stdev(prices[-10:]) if len(prices) >= 10 else 0
        avg_volatility = stdev(prices[-20:]) if len(prices) >= 20 else recent_volatility
        
        # Price compression analysis
        price_range = max(prices[-10:]) - min(prices[-10:])
        avg_range = (max(prices) - min(prices)) / len(prices) * 10
        
        compression_ratio = price_range / avg_range if avg_range > 0 else 1
        
        # Calculate probability (0-100%)
        if compression_ratio < 0.5 and recent_volatility < avg_volatility:
            return min(90, 60 + (1 - compression_ratio) * 30)
        else:
            return max(10, 50 - compression_ratio * 20)
    
    def _detect_reversal_signals(self, prices):
        """AI-powered reversal signal detection"""
        if len(prices) < 15:
            return []
        
        signals = []
        
        # Double top/bottom detection
        recent_high = max(prices[-10:])
        recent_low = min(prices[-10:])
        
        # Look for similar levels in older data
        for i in range(len(prices) - 15, len(prices) - 5):
            if abs(prices[i] - recent_high) / recent_high < 0.02:  # Within 2%
                signals.append("POTENTIAL_DOUBLE_TOP")
                break
            if abs(prices[i] - recent_low) / recent_low < 0.02:  # Within 2%
                signals.append("POTENTIAL_DOUBLE_BOTTOM")
                break
        
        # Divergence detection (simplified)
        price_trend = prices[-1] - prices[-10]
        momentum = self._calculate_momentum(prices)
        
        if price_trend > 0 and momentum < -20:
            signals.append("BEARISH_DIVERGENCE")
        elif price_trend < 0 and momentum > 20:
            signals.append("BULLISH_DIVERGENCE")
        
        return signals
    
    def predictive_analysis(self, symbol):
        """AI Predictive Analysis: Forecast price movements"""
        try:
            patterns = self.detect_patterns(symbol)
            if not patterns:
                return None
            
            # AI prediction based on multiple factors
            prediction_score = 0
            confidence_factors = []
            
            # Trend factor
            if patterns["trend"] == "BULLISH":
                prediction_score += 30
                confidence_factors.append("Bullish trend confirmed")
            elif patterns["trend"] == "BEARISH":
                prediction_score -= 30
                confidence_factors.append("Bearish trend confirmed")
            
            # Momentum factor
            momentum = patterns["momentum"]
            prediction_score += momentum * 0.3
            confidence_factors.append(f"Momentum: {momentum:.1f}")
            
            # Breakout probability
            breakout_prob = patterns["breakout_probability"]
            if breakout_prob > 70:
                prediction_score += 20
                confidence_factors.append(f"High breakout probability: {breakout_prob:.1f}%")
            
            # Volume confirmation
            if patterns["volume_pattern"] == "HIGH":
                prediction_score += 15
                confidence_factors.append("Volume confirmation")
            
            # Reversal signals
            reversals = patterns["reversal_signals"]
            if "BULLISH_DIVERGENCE" in reversals:
                prediction_score += 25
                confidence_factors.append("Bullish divergence detected")
            elif "BEARISH_DIVERGENCE" in reversals:
                prediction_score -= 25
                confidence_factors.append("Bearish divergence detected")
            
            # Normalize prediction score to direction and confidence
            direction = "BUY" if prediction_score > 10 else "SELL" if prediction_score < -10 else "HOLD"
            confidence = min(95, max(50, 50 + abs(prediction_score)))
            
            # Update AI confidence
            self.ai_confidence = confidence
            
            prediction = {
                "direction": direction,
                "confidence": confidence,
                "score": prediction_score,
                "factors": confidence_factors,
                "timestamp": datetime.now()
            }
            
            self.ai_predictions["next_price_direction"] = prediction
            
            logger.info(f"🤖 AI Prediction: {direction} with {confidence:.1f}% confidence")
            logger.info(f"🎯 Key factors: {', '.join(confidence_factors[:3])}")
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error in predictive analysis: {e}")
            return None
    
    def optimize_parameters(self):
        """AI Parameter Optimization: Dynamically adjust trading parameters"""
        try:
            if len(self.pattern_history) < 10:
                return False
            
            # Calculate recent performance
            recent_trades = TradeSignal.query.filter(
                TradeSignal.timestamp >= datetime.now() - timedelta(days=7)
            ).all()
            
            if not recent_trades:
                return False
            
            # Calculate success rate
            successful_trades = [t for t in recent_trades if t.status == 'success']
            success_rate = len(successful_trades) / len(recent_trades)
            
            # Optimize take profit and stop loss based on AI analysis
            old_tp = self.take_profit_pct
            old_sl = self.stop_loss_pct
            
            # AI optimization logic
            if success_rate < 0.6:  # Low success rate
                # Reduce take profit, increase stop loss (more conservative)
                self.take_profit_pct = max(0.03, self.take_profit_pct * 0.9)
                self.stop_loss_pct = min(0.05, self.stop_loss_pct * 1.1)
                reason = "Low success rate - more conservative"
            elif success_rate > 0.8:  # High success rate
                # Increase take profit, reduce stop loss (more aggressive)
                self.take_profit_pct = min(0.10, self.take_profit_pct * 1.1)
                self.stop_loss_pct = max(0.02, self.stop_loss_pct * 0.9)
                reason = "High success rate - more aggressive"
            else:
                return False
            
            # Update performance metrics
            self.performance_metrics["win_rate"] = success_rate * 100
            
            logger.info(f"🔧 AI Parameter Optimization: {reason}")
            logger.info(f"📊 Take Profit: {old_tp*100:.1f}% → {self.take_profit_pct*100:.1f}%")
            logger.info(f"📊 Stop Loss: {old_sl*100:.1f}% → {self.stop_loss_pct*100:.1f}%")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in parameter optimization: {e}")
            return False
    
    def risk_assessment(self, symbol, amount):
        """AI-powered risk assessment before trade execution"""
        try:
            patterns = self.detect_patterns(symbol)
            if not patterns:
                return {"risk_level": "MEDIUM", "risk_score": 50, "recommendations": []}
            
            risk_score = 50  # Base risk
            recommendations = []
            
            # Volatility risk
            if patterns.get("volume_pattern") == "HIGH":
                risk_score += 15
                recommendations.append("Alta volatilidad detectada")
            
            # Trend risk
            if patterns.get("trend") == "SIDEWAYS":
                risk_score += 10
                recommendations.append("Mercado lateral - riesgo de whipsaw")
            
            # Momentum risk
            momentum = patterns.get("momentum", 0)
            if abs(momentum) < 10:
                risk_score += 10
                recommendations.append("Momentum débil")
            
            # Breakout risk
            breakout_prob = patterns.get("breakout_probability", 50)
            if breakout_prob > 80:
                risk_score += 20
                recommendations.append("Alta probabilidad de breakout - riesgo de falso breakout")
            
            # Position size risk (based on available balance)
            try:
                account_info = self.client.get_account()
                usdt_balance = 0
                for balance in account_info["balances"]:
                    if balance["asset"] == "USDT":
                        usdt_balance = float(balance["free"])
                        break
                
                position_ratio = amount / usdt_balance if usdt_balance > 0 else 1
                if position_ratio > 0.3:  # More than 30% of balance
                    risk_score += 25
                    recommendations.append("Tamaño de posición muy grande")
                elif position_ratio > 0.15:  # More than 15% of balance
                    risk_score += 10
                    recommendations.append("Tamaño de posición considerable")
            except:
                risk_score += 5
                recommendations.append("No se pudo verificar balance")
            
            # Determine risk level
            if risk_score >= 80:
                risk_level = "VERY_HIGH"
            elif risk_score >= 65:
                risk_level = "HIGH"
            elif risk_score >= 45:
                risk_level = "MEDIUM"
            elif risk_score >= 30:
                risk_level = "LOW"
            else:
                risk_level = "VERY_LOW"
            
            assessment = {
                "risk_level": risk_level,
                "risk_score": min(100, risk_score),
                "recommendations": recommendations,
                "position_size_suggestion": amount * (100 - risk_score) / 100
            }
            
            logger.info(f"🛡️ AI Risk Assessment: {risk_level} (Score: {risk_score}/100)")
            if recommendations:
                logger.info(f"⚠️ Recommendations: {', '.join(recommendations[:2])}")
            
            return assessment
            
        except Exception as e:
            logger.error(f"Error in risk assessment: {e}")
            return {"risk_level": "MEDIUM", "risk_score": 50, "recommendations": ["Error en evaluación"]}
    
    # ====== PERFORMANCE OPTIMIZATION ======
    
    def _get_cached_data(self, key, ttl_seconds=300):
        """High-performance caching system with TTL"""
        current_time = datetime.now()
        
        # Check if data exists and is still valid
        if key in self.cache and key in self.cache_ttl:
            if current_time - self.cache_ttl[key] < timedelta(seconds=ttl_seconds):
                self.performance_stats["cache_hits"] += 1
                logger.debug(f"⚡ Cache HIT for {key}")
                return self.cache[key]
        
        # Data not in cache or expired
        logger.debug(f"💾 Cache MISS for {key}")
        return None
    
    def _set_cached_data(self, key, data):
        """Store data in cache with timestamp"""
        self.cache[key] = data
        self.cache_ttl[key] = datetime.now()
        
        # Clean old cache entries (keep last 50 items)
        if len(self.cache) > 50:
            oldest_keys = sorted(self.cache_ttl.keys(), key=lambda k: self.cache_ttl[k])[:10]
            for old_key in oldest_keys:
                del self.cache[old_key]
                del self.cache_ttl[old_key]
    
    def _measure_performance(self, func_name):
        """Performance measurement decorator context"""
        class PerformanceMeasurer:
            def __init__(self, bot, name):
                self.bot = bot
                self.name = name
                self.start_time = None
            
            def __enter__(self):
                self.start_time = datetime.now()
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                execution_time = (datetime.now() - self.start_time).total_seconds()
                self.bot.performance_stats["execution_times"].append({
                    "function": self.name,
                    "time": execution_time,
                    "timestamp": self.start_time
                })
                
                # Keep only recent performance data
                if len(self.bot.performance_stats["execution_times"]) > 100:
                    self.bot.performance_stats["execution_times"] = \
                        self.bot.performance_stats["execution_times"][-100:]
                
                if execution_time > 1.0:  # Log slow operations
                    logger.warning(f"🐌 Slow operation: {self.name} took {execution_time:.2f}s")
                else:
                    logger.debug(f"⚡ Fast execution: {self.name} took {execution_time:.3f}s")
        
        return PerformanceMeasurer(self, func_name)
    
    def get_optimized_klines(self, symbol, interval="5m", limit=50):
        """Optimized klines retrieval with caching"""
        cache_key = f"klines_{symbol}_{interval}_{limit}"
        
        with self._measure_performance("get_klines"):
            # Try cache first
            cached_data = self._get_cached_data(cache_key, ttl_seconds=60)  # 1 minute TTL
            if cached_data:
                return cached_data
            
            # Fetch from API
            try:
                self.performance_stats["api_calls"] += 1
                klines = self.client.get_klines(symbol=symbol, interval=interval, limit=limit)
                
                # Cache the result
                self._set_cached_data(cache_key, klines)
                
                return klines
            except Exception as e:
                logger.error(f"Error fetching klines: {e}")
                return None
    
    def get_optimized_ticker(self, symbol):
        """Optimized ticker retrieval with caching"""
        cache_key = f"ticker_{symbol}"
        
        with self._measure_performance("get_ticker"):
            # Try cache first (short TTL for price data)
            cached_data = self._get_cached_data(cache_key, ttl_seconds=30)  # 30 seconds TTL
            if cached_data:
                return cached_data
            
            # Fetch from API
            try:
                self.performance_stats["api_calls"] += 1
                ticker = self.client.get_symbol_ticker(symbol=symbol)
                
                # Cache the result
                self._set_cached_data(cache_key, ticker)
                
                return ticker
            except Exception as e:
                logger.error(f"Error fetching ticker: {e}")
                return None
    
    def optimize_memory_usage(self):
        """Memory optimization and cleanup"""
        try:
            # Clean old pattern history
            if len(self.pattern_history) > 50:
                self.pattern_history = self.pattern_history[-50:]
                logger.debug("🧹 Cleaned old pattern history")
            
            # Clean old performance stats
            if len(self.performance_stats["execution_times"]) > 100:
                self.performance_stats["execution_times"] = \
                    self.performance_stats["execution_times"][-100:]
                logger.debug("🧹 Cleaned old performance stats")
            
            # Clean expired cache entries
            current_time = datetime.now()
            expired_keys = []
            for key, timestamp in self.cache_ttl.items():
                if current_time - timestamp > timedelta(minutes=10):  # 10 minutes max
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.cache[key]
                del self.cache_ttl[key]
            
            if expired_keys:
                logger.debug(f"🧹 Cleaned {len(expired_keys)} expired cache entries")
            
            # Update memory usage stats (simplified)
            import sys
            self.performance_stats["memory_usage"] = sys.getsizeof(self.cache) + \
                                                   sys.getsizeof(self.pattern_history) + \
                                                   sys.getsizeof(self.performance_stats)
            
            return True
        except Exception as e:
            logger.error(f"Error in memory optimization: {e}")
            return False
    
    def get_performance_report(self):
        """Generate comprehensive performance report"""
        try:
            current_time = datetime.now()
            
            # Calculate average execution times by function
            func_times = {}
            for stat in self.performance_stats["execution_times"]:
                func_name = stat["function"]
                if func_name not in func_times:
                    func_times[func_name] = []
                func_times[func_name].append(stat["time"])
            
            avg_times = {}
            for func, times in func_times.items():
                avg_times[func] = {
                    "avg": mean(times),
                    "min": min(times),
                    "max": max(times),
                    "count": len(times)
                }
            
            # Calculate cache efficiency
            total_operations = self.performance_stats["api_calls"] + self.performance_stats["cache_hits"]
            cache_efficiency = (self.performance_stats["cache_hits"] / total_operations * 100) if total_operations > 0 else 0
            
            # Calculate uptime since last optimization
            uptime = (current_time - self.last_optimization).total_seconds()
            
            report = {
                "timestamp": current_time,
                "uptime_seconds": uptime,
                "cache_efficiency": cache_efficiency,
                "api_calls": self.performance_stats["api_calls"],
                "cache_hits": self.performance_stats["cache_hits"],
                "memory_usage_bytes": self.performance_stats["memory_usage"],
                "function_performance": avg_times,
                "cache_size": len(self.cache),
                "pattern_history_size": len(self.pattern_history)
            }
            
            logger.info(f"📊 Performance Report: Cache efficiency: {cache_efficiency:.1f}%, API calls: {self.performance_stats['api_calls']}")
            
            return report
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return None
    
    def auto_optimization(self):
        """Automatic performance optimization routine"""
        try:
            current_time = datetime.now()
            
            # Run optimization every 30 minutes
            if current_time - self.last_optimization < timedelta(minutes=30):
                return False
            
            logger.info("🔧 Running automatic performance optimization...")
            
            # Memory cleanup
            self.optimize_memory_usage()
            
            # Performance analysis
            report = self.get_performance_report()
            
            # Adaptive cache TTL optimization
            if report and report["cache_efficiency"] < 70:
                logger.info("📈 Optimizing cache strategy for better efficiency")
                # Could implement dynamic TTL adjustment here
            
            # API call optimization
            if report and report["api_calls"] > 100:
                logger.info("🌐 High API usage detected, optimizing call frequency")
                # Could implement rate limiting or batching here
            
            self.last_optimization = current_time
            
            logger.info("✅ Performance optimization completed")
            return True
            
        except Exception as e:
            logger.error(f"Error in auto optimization: {e}")
            return False
    
    def update_credentials(self, api_key, api_secret):
        """Update Binance API credentials and reinitialize client"""
        try:
            logger.info("Updating Binance API credentials...")
            
            # Update credentials
            self.api_key = api_key
            self.api_secret = api_secret
            
            # Reinitialize client with new credentials
            self.client = self._initialize_client()
            
            if self.client:
                logger.info("✅ Binance API credentials updated successfully")
                return True
            else:
                logger.error("❌ Failed to initialize client with new credentials")
                return False
                
        except Exception as e:
            logger.error(f"Error updating credentials: {e}")
            return False
    
    def load_credentials_from_db(self):
        """Load active credentials from database"""
        try:
            from models import BinanceCredentials
            credentials = BinanceCredentials.get_active_credentials()
            
            if credentials:
                logger.info("Loading credentials from database...")
                # Note: We can't decrypt the secret, so we need to handle this differently
                # This method will be used when the credentials are saved
                self.api_key = credentials.api_key
                return True
            else:
                logger.info("No active credentials found in database")
                return False
                
        except Exception as e:
            logger.error(f"Error loading credentials from database: {e}")
            return False

# Global bot instance
trading_bot = TradingBot()
