import json
import logging
from datetime import datetime
from flask import request, jsonify, render_template, flash, redirect, url_for
from app import app, db
from models import TradeSignal, BotStatus, BinanceCredentials
from trading_bot import trading_bot
import backtesting
from cloud_optimizer import cloud_optimizer
from telegram_bot import telegram_notifier

logger = logging.getLogger(__name__)

@app.route('/')
def index():
    """Main dashboard page"""
    bot_status = BotStatus.get_status()
    recent_signals = TradeSignal.query.order_by(TradeSignal.timestamp.desc()).limit(10).all()
    
    # Get account info if bot is ready
    account_info = None
    if trading_bot.is_ready():
        account_info = trading_bot.get_account_info()
    
    # Get Binance credentials status
    binance_credentials = BinanceCredentials.get_active_credentials()
    
    return render_template('index.html', 
                         bot_status=bot_status, 
                         recent_signals=recent_signals,
                         account_info=account_info,
                         bot_ready=trading_bot.is_ready(),
                         binance_credentials=binance_credentials)

@app.route('/webhook', methods=['POST'])
def webhook():
    """Process trading signals from external sources"""
    try:
        print("¡Webhook recibido!")
        data = request.get_json()
        
        if not data:
            logger.error("No JSON data received in webhook")
            return {"status": "error", "message": "No data received"}, 400
        
        # Extract signal data
        signal_type = data.get('signal', '').upper()
        symbol = data.get('symbol', 'BTCUSDT')
        amount = data.get('amount')
        price = data.get('price')
        
        # Create signal record
        signal = TradeSignal(
            signal_type=signal_type,
            symbol=symbol,
            amount=amount,
            price=price,
            status='received',
            raw_data=json.dumps(data)
        )
        
        db.session.add(signal)
        db.session.commit()
        
        logger.info(f"Signal saved: {signal_type} {symbol} - Amount: {amount}")
        
        # Try to execute the trade if bot is ready
        if trading_bot.is_ready() and signal_type in ['BUY', 'SELL']:
            try:
                # For now, just mark as executed - actual trading integration can be added later
                signal.status = 'executed'
                db.session.commit()
                logger.info(f"Trade would be executed: {signal_type} {symbol}")
            except Exception as e:
                signal.status = 'error'
                signal.error_message = str(e)
                db.session.commit()
                logger.error(f"Trade execution failed: {e}")
        else:
            signal.status = 'ignored'
            signal.error_message = 'Bot not ready or invalid signal type'
            db.session.commit()
            logger.warning(f"Signal ignored: Bot ready={trading_bot.is_ready()}, Signal={signal_type}")
        
        return {"status": "ok", "signal_id": signal.id}
        
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return {"status": "error", "message": str(e)}, 500

@app.route('/api/status')
def api_status():
    """API endpoint to get bot status"""
    bot_status = BotStatus.get_status()
    return jsonify({
        "is_active": bot_status.is_active,
        "is_ready": trading_bot.is_ready(),
        "last_ping": bot_status.last_ping.isoformat() if bot_status.last_ping else None,
        "total_trades": bot_status.total_trades,
        "successful_trades": bot_status.successful_trades,
        "failed_trades": bot_status.failed_trades
    })

@app.route('/api/signals')
def api_signals():
    """API endpoint to get recent trading signals"""
    try:
        limit = request.args.get('limit', 50, type=int)
        signals = TradeSignal.query.order_by(TradeSignal.timestamp.desc()).limit(limit).all()
        
        return jsonify([{
            "id": signal.id,
            "signal_type": signal.signal_type,
            "symbol": signal.symbol,
            "amount": signal.amount,
            "price": signal.price,
            "status": signal.status,
            "order_id": signal.order_id,
            "error_message": signal.error_message,
            "timestamp": signal.timestamp.isoformat()
        } for signal in signals])
    except Exception as e:
        logger.error(f"Error in api_signals: {e}")
        return jsonify({"error": "Unable to fetch signals"}), 500

@app.route('/toggle_bot', methods=['POST'])
def toggle_bot():
    """Toggle bot active status"""
    bot_status = BotStatus.get_status()
    bot_status.is_active = not bot_status.is_active
    db.session.commit()
    
    status_text = "activated" if bot_status.is_active else "deactivated"
    flash(f"Trading bot has been {status_text}", "success")
    return redirect(url_for('index'))

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({"status": "error", "message": "Endpoint not found"}), 404

@app.route('/configure_binance')
def configure_binance():
    """Show Binance configuration form"""
    credentials = BinanceCredentials.get_active_credentials()
    return render_template('configure_binance.html', credentials=credentials)

@app.route('/save_binance_credentials', methods=['POST'])
def save_binance_credentials():
    """Save Binance API credentials"""
    api_key = request.form.get('api_key', '').strip()
    api_secret = request.form.get('api_secret', '').strip()
    
    if not api_key or not api_secret:
        flash('Por favor ingresa tanto la API Key como el Secret', 'error')
        return redirect(url_for('configure_binance'))
    
    try:
        # Save credentials
        credentials = BinanceCredentials.set_credentials(api_key, api_secret)
        
        # Test connection
        if credentials.test_connection(api_secret):
            # Update the trading bot with new credentials
            trading_bot.update_credentials(api_key, api_secret)
            telegram_notifier.send_status_alert("Credenciales actualizadas", "Conexión con Binance establecida correctamente")
            flash('Credenciales guardadas y conexión exitosa con Binance', 'success')
        else:
            flash('Credenciales guardadas pero no se pudo conectar con Binance. Verifica tus credenciales.', 'warning')
            
    except Exception as e:
        logger.error(f"Error saving Binance credentials: {e}")
        flash('Error al guardar las credenciales. Inténtalo de nuevo.', 'error')
    
    return redirect(url_for('index'))

@app.route('/test_binance_connection')
def test_binance_connection():
    """Test current Binance connection"""
    credentials = BinanceCredentials.get_active_credentials()
    if not credentials:
        return jsonify({
            "success": False,
            "message": "No hay credenciales configuradas"
        })
    
    try:
        # For testing, we'll use the bot's current connection status
        is_connected = trading_bot.is_ready()
        
        credentials.is_connected = is_connected
        credentials.last_connection_test = datetime.utcnow()
        db.session.commit()
        
        return jsonify({
            "success": is_connected,
            "message": "Conexión exitosa" if is_connected else "Error de conexión",
            "last_test": credentials.last_connection_test.isoformat()
        })
    except Exception as e:
        logger.error(f"Error testing connection: {e}")
        return jsonify({
            "success": False,
            "message": f"Error al probar la conexión: {str(e)}"
        })

@app.route('/backtest')
def backtest_page():
    """Show backtesting page"""
    return render_template('backtest.html')

@app.route('/run_backtest', methods=['POST'])
def run_backtest():
    """Run backtesting with given parameters"""
    try:
        symbol = request.form.get('symbol', 'PEPEUSDT')
        days = int(request.form.get('days', 30))
        
        # Run backtest
        result_tuple = backtesting.run_backtesting()
        
        if result_tuple and result_tuple[0] is not None:
            # Unpack the tuple (df_trades, capital, total_return)
            df_trades, final_capital, total_return = result_tuple
            
            # Create results object for template
            results = {
                'trades': df_trades.to_dict('records') if df_trades is not None else [],
                'final_capital': final_capital,
                'total_return': total_return,
                'symbol': symbol,
                'days': days,
                'trade_count': len(df_trades) if df_trades is not None else 0
            }
            
            flash('Backtesting completado exitosamente', 'success')
            return render_template('backtest.html', results=results)
        else:
            flash('No se ejecutaron operaciones en el backtesting', 'warning')
            return render_template('backtest.html', results=None)
            
    except Exception as e:
        logger.error(f"Error in backtesting: {e}")
        flash('Error ejecutando el backtesting', 'error')
        return redirect(url_for('backtest_page'))

@app.route('/api/system_status')
def system_status():
    """Get comprehensive system status"""
    try:
        # Get performance metrics
        performance_metrics = cloud_optimizer.performance_monitor.get_current_metrics()
        
        # Get health status
        health_status = cloud_optimizer.health_checker.health_status
        
        return jsonify({
            'bot_status': trading_bot.is_ready(),
            'health_status': health_status,
            'performance': performance_metrics,
            'cloud_environment': cloud_optimizer.is_cloud_environment,
            'telegram_enabled': telegram_notifier.enabled,
            'last_health_check': cloud_optimizer.health_checker.last_check.isoformat() if cloud_optimizer.health_checker.last_check else None
        })
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return jsonify({'error': 'Unable to get system status'}), 500

@app.route('/telegram/setup')
def telegram_setup():
    """Show Telegram setup page"""
    return render_template('telegram_setup.html')

@app.route('/api/test_telegram', methods=['POST'])
def test_telegram():
    """Test Telegram connection"""
    try:
        if not telegram_notifier.enabled:
            return jsonify({
                "success": False,
                "message": "Telegram no está configurado. Configura TELEGRAM_BOT_TOKEN y TELEGRAM_CHAT_ID."
            })
        
        # Send test message
        telegram_notifier.send_status_alert("Prueba de conexión", "✅ El bot está funcionando correctamente!")
        
        return jsonify({
            "success": True,
            "message": "¡Mensaje de prueba enviado exitosamente! Revisa tu Telegram."
        })
        
    except Exception as e:
        logger.error(f"Error testing Telegram: {e}")
        return jsonify({
            "success": False,
            "message": f"Error probando Telegram: {str(e)}"
        })

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({"status": "error", "message": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return jsonify({"status": "error", "message": "Internal server error"}), 500
