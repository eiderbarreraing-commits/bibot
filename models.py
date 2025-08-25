from app import db
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

class TradeSignal(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    signal_type = db.Column(db.String(20), nullable=False)  # 'buy', 'sell', etc.
    symbol = db.Column(db.String(20), nullable=False)
    amount = db.Column(db.Float, nullable=True)
    price = db.Column(db.Float, nullable=True)
    status = db.Column(db.String(20), nullable=False)  # 'success', 'error', 'ignored'
    order_id = db.Column(db.String(50), nullable=True)
    error_message = db.Column(db.Text, nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    raw_data = db.Column(db.Text, nullable=True)  # Store original webhook data

    def __repr__(self):
        return f'<TradeSignal {self.signal_type} {self.symbol} {self.status}>'

class BotStatus(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    is_active = db.Column(db.Boolean, default=True)
    last_ping = db.Column(db.DateTime, default=datetime.utcnow)
    total_trades = db.Column(db.Integer, default=0)
    successful_trades = db.Column(db.Integer, default=0)
    failed_trades = db.Column(db.Integer, default=0)
    
    @classmethod
    def get_status(cls):
        status = cls.query.first()
        if not status:
            status = cls()
            db.session.add(status)
            db.session.commit()
        return status
    
    def update_ping(self):
        self.last_ping = datetime.utcnow()
        db.session.commit()
    
    def increment_trade(self, success=True):
        self.total_trades += 1
        if success:
            self.successful_trades += 1
        else:
            self.failed_trades += 1
        db.session.commit()


class BinanceCredentials(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    api_key = db.Column(db.String(100), nullable=False)
    api_secret_hash = db.Column(db.String(200), nullable=False)
    is_active = db.Column(db.Boolean, default=True)
    is_connected = db.Column(db.Boolean, default=False)
    last_connection_test = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def set_api_secret(self, api_secret):
        """Store encrypted API secret"""
        self.api_secret_hash = generate_password_hash(api_secret)
    
    def check_api_secret(self, api_secret):
        """Verify API secret"""
        return check_password_hash(self.api_secret_hash, api_secret)
    
    @classmethod
    def get_active_credentials(cls):
        """Get the currently active credentials"""
        return cls.query.filter_by(is_active=True).first()
    
    @classmethod
    def set_credentials(cls, api_key, api_secret):
        """Set new credentials and deactivate others"""
        # Deactivate all existing credentials
        cls.query.update({'is_active': False})
        
        # Create new credentials
        credentials = cls(api_key=api_key)
        credentials.set_api_secret(api_secret)
        db.session.add(credentials)
        db.session.commit()
        return credentials
    
    def test_connection(self, api_secret):
        """Test the connection with these credentials"""
        try:
            from binance.client import Client
            client = Client(self.api_key, api_secret)
            # Try to ping the API
            client.ping()
            self.is_connected = True
            self.last_connection_test = datetime.utcnow()
            db.session.commit()
            return True
        except Exception as e:
            self.is_connected = False
            self.last_connection_test = datetime.utcnow()
            db.session.commit()
            return False
