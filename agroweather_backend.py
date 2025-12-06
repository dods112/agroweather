from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict
import sqlite3
import hashlib
import secrets
from datetime import datetime, timedelta
import httpx
from pathlib import Path
import logging
import os
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agroweather")

app = FastAPI(title="AgroWeather System", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow ALL devices (for testing)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Configuration
OPENWEATHER_API_KEY = "e9564520e982e273d2c242d296f0c433"
OPEN_METEO_BASE = "https://api.open-meteo.com/v1/forecast"
OPENWEATHER_BASE = "https://api.openweathermap.org/data/2.5/weather"

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "agroweather.db"

def init_db():
    logger.info(f"Initializing DB at: {DB_PATH}")
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS farmers
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT NOT NULL,
                  age INTEGER NOT NULL,
                  gender TEXT NOT NULL,
                  location TEXT NOT NULL,
                  username TEXT UNIQUE NOT NULL,
                  password_hash TEXT NOT NULL,
                  created_at TEXT NOT NULL)''')
                
    c.execute('''CREATE TABLE IF NOT EXISTS sessions
                 (token TEXT PRIMARY KEY,
                  user_id INTEGER NOT NULL,
                  created_at TEXT NOT NULL,
                  expires_at TEXT NOT NULL,
                  FOREIGN KEY(user_id) REFERENCES farmers(id))''')
    
    # Add notification tables
    c.execute('''CREATE TABLE IF NOT EXISTS notifications
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  title TEXT NOT NULL,
                  message TEXT NOT NULL,
                  type TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  created_by INTEGER,
                  is_active INTEGER DEFAULT 1,
                  FOREIGN KEY(created_by) REFERENCES farmers(id))''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS user_notifications
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER NOT NULL,
                  notification_id INTEGER NOT NULL,
                  is_read INTEGER DEFAULT 0,
                  read_at TEXT,
                  FOREIGN KEY(user_id) REFERENCES farmers(id),
                  FOREIGN KEY(notification_id) REFERENCES notifications(id))''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS historical_events
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  event_name TEXT NOT NULL,
                  event_type TEXT NOT NULL,
                  event_date TEXT NOT NULL,
                  latitude REAL NOT NULL,
                  longitude REAL NOT NULL,
                  severity TEXT NOT NULL,
                  description TEXT,
                  affected_areas TEXT,
                  casualties INTEGER DEFAULT 0,
                  damage_estimate TEXT,
                  source TEXT,
                  created_at TEXT NOT NULL)''')
    
    # Add is_admin column if it doesn't exist
    try:
        c.execute('ALTER TABLE farmers ADD COLUMN is_admin INTEGER DEFAULT 0')
    except sqlite3.OperationalError:
        pass  # Column already exists

    # ADD THIS: Add last_seen column for online/offline status
    try:
        c.execute('ALTER TABLE farmers ADD COLUMN last_seen TEXT')
    except sqlite3.OperationalError:
        pass  # Column already exists

    conn.commit()
    conn.close()

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def create_session_token() -> str:
    return secrets.token_urlsafe(32)

def get_db():
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def create_default_admin():
    """Create default admin user if none exists"""
    conn = get_db()
    c = conn.cursor()
    
    # Check if admin exists
    admin = c.execute('SELECT id FROM farmers WHERE username = ?', ('admin',)).fetchone()
    
    if not admin:
        # Create default admin
        password_hash = hash_password('admin123')
        created_at = datetime.now().isoformat()
        
        c.execute('''INSERT INTO farmers (name, age, gender, location, username, password_hash, created_at, is_admin)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                  ('Admin User', 30, 'Other', 'System', 'admin', password_hash, created_at, 1))
        
        conn.commit()
        logger.info("✅ Default admin created: username='admin', password='admin123'")
    
    conn.close()

def seed_philippine_historical_events():
    """Add some historical Philippine weather events as sample data"""
    conn = get_db()
    c = conn.cursor()
    
    count = c.execute('SELECT COUNT(*) as count FROM historical_events').fetchone()
    if count and count['count'] > 0:
        conn.close()
        logger.info("📜 Historical events already seeded")
        return
    
    sample_events = [
        {
            "event_name": "Typhoon Yolanda (Haiyan)",
            "event_type": "typhoon",
            "event_date": "2013-11-08",
            "latitude": 11.2500,
            "longitude": 125.0000,
            "severity": "extreme",
            "description": "One of the strongest tropical cyclones ever recorded. Devastating storm surge.",
            "affected_areas": "Eastern Visayas, Tacloban, Samar, Leyte",
            "casualties": 6300,
            "damage_estimate": "₱95.5 billion",
            "source": "PAGASA"
        },
        {
            "event_name": "Typhoon Odette (Rai)",
            "event_type": "typhoon",
            "event_date": "2021-12-16",
            "latitude": 9.8500,
            "longitude": 126.1000,
            "severity": "extreme",
            "description": "Super typhoon that devastated Visayas and Mindanao regions",
            "affected_areas": "Surigao del Norte, Bohol, Cebu, Palawan",
            "casualties": 410,
            "damage_estimate": "₱51.8 billion",
            "source": "PAGASA"
        },
        {
            "event_name": "Typhoon Pepito (Man-yi)",
            "event_type": "typhoon",
            "event_date": "2024-11-17",
            "latitude": 14.5995,
            "longitude": 120.9842,
            "severity": "high",
            "description": "Recent typhoon affecting Luzon areas",
            "affected_areas": "Luzon, Northern provinces",
            "casualties": 8,
            "damage_estimate": "Under assessment",
            "source": "PAGASA"
        }
    ]
    
    created_at = datetime.now().isoformat()
    
    for event in sample_events:
        c.execute('''INSERT INTO historical_events 
                     (event_name, event_type, event_date, latitude, longitude, 
                      severity, description, affected_areas, casualties, 
                      damage_estimate, source, created_at)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                  (event["event_name"], event["event_type"], event["event_date"],
                   event["latitude"], event["longitude"], event["severity"],
                   event["description"], event["affected_areas"], event["casualties"],
                   event["damage_estimate"], event["source"], created_at))
    
    conn.commit()
    logger.info(f"✅ Seeded {len(sample_events)} historical events")
    conn.close()

init_db()
create_default_admin()
seed_philippine_historical_events() 


class SignUpRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    age: int = Field(..., ge=18, le=120)
    gender: str = Field(..., pattern="^(Male|Female|Other)$")
    location: str = Field(..., min_length=3, max_length=200)
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)
    confirm_password: str

    @validator('confirm_password')
    def passwords_match(cls, v, values):
        if 'password' in values and v != values['password']:
            raise ValueError('Passwords do not match')
        return v

class LoginRequest(BaseModel):
    username: str
    password: str

class AuthResponse(BaseModel):
    token: str
    user: Dict

class Coordinates(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)

class WeatherResponse(BaseModel):
    latitude: float
    longitude: float
    temperature: Optional[float] = None
    windspeed: Optional[float] = None
    precipitation: Optional[float] = None
    humidity: Optional[float] = None
    weather_code: Optional[int] = None
    weather_description: Optional[str] = None
    uv_index: Optional[float] = None
    feels_like: Optional[float] = None
    hourly_times: List[str] = []
    hourly_temp: List[Optional[float]] = []
    hourly_precip: List[Optional[float]] = []
    hourly_humidity: List[Optional[float]] = []
    hourly_weather_code: List[Optional[int]] = []
    pressure: Optional[float] = None
    visibility: Optional[float] = None
    daily: List[Dict] = []
    
class PredictionResponse(BaseModel):
    period: str
    avg_temperature: float
    total_precipitation: float
    avg_humidity: float
    forecast_data: List[Dict]

class AdviceRequest(BaseModel):
    crop: Optional[str] = Field(default=None)
    period: str = Field(default="day", pattern="^(day|week|month)$")

class AdviceResponse(BaseModel):
    summary: str
    recommendations: List[str]
    weather_alerts: List[str]
    best_activities: List[str]
    
class NotificationCreate(BaseModel):
    title: str = Field(..., min_length=3, max_length=100)
    message: str = Field(..., min_length=10, max_length=500)
    type: str = Field(default="info", pattern="^(info|warning|alert|success)$")

class NotificationResponse(BaseModel):
    id: int
    title: str
    message: str
    type: str
    created_at: str
    is_read: bool = False
class HistoricalEventCreate(BaseModel):
    event_name: str = Field(..., min_length=3, max_length=200)
    event_type: str = Field(..., pattern="^(typhoon|flood|drought|earthquake|landslide|storm_surge)$")
    event_date: str  # ISO format date
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    severity: str = Field(..., pattern="^(low|moderate|high|extreme)$")
    description: Optional[str] = None
    affected_areas: Optional[str] = None
    casualties: Optional[int] = Field(default=0, ge=0)
    damage_estimate: Optional[str] = None
    source: Optional[str] = None

class HistoricalEventResponse(BaseModel):
    id: int
    event_name: str
    event_type: str
    event_date: str
    latitude: float
    longitude: float
    severity: str
    description: Optional[str]
    affected_areas: Optional[str]
    casualties: int
    damage_estimate: Optional[str]
    distance_km: Optional[float] = None
    source: Optional[str]
    created_at: str
    
async def get_current_user(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith('Bearer '):
        raise HTTPException(status_code=401, detail="Not authenticated")

    token = authorization.split(' ')[1]
    conn = get_db()
    c = conn.cursor()

    session = c.execute(
        'SELECT s.*, f.* FROM sessions s JOIN farmers f ON s.user_id = f.id WHERE s.token = ?',
        (token,)
    ).fetchone()

    conn.close()

    if not session:
        raise HTTPException(status_code=401, detail="Invalid session")

    expires_at = datetime.fromisoformat(session['expires_at'])
    if datetime.now() > expires_at:
        raise HTTPException(status_code=401, detail="Session expired")

    return dict(session)

def calculate_heat_index(temp_c: float, humidity: float) -> float:
    """Calculate heat index (feels-like temperature) in Celsius"""
    # Convert to Fahrenheit for calculation
    temp_f = (temp_c * 9/5) + 32
    
    # Heat index formula
    hi = -42.379 + 2.04901523*temp_f + 10.14333127*humidity \
         - 0.22475541*temp_f*humidity - 0.00683783*temp_f**2 \
         - 0.05481717*humidity**2 + 0.00122874*temp_f**2*humidity \
         + 0.00085282*temp_f*humidity**2 - 0.00000199*temp_f**2*humidity**2
    
    # Convert back to Celsius
    hi_c = (hi - 32) * 5/9
    return round(hi_c, 1)

def calculate_wind_chill(temp_c: float, wind_speed_kmh: float) -> float:
    """Calculate wind chill temperature in Celsius"""
    if temp_c > 10 or wind_speed_kmh < 4.8:
        return temp_c
    
    # Wind chill formula
    wc = 13.12 + 0.6215*temp_c - 11.37*(wind_speed_kmh**0.16) + 0.3965*temp_c*(wind_speed_kmh**0.16)
    return round(wc, 1)

async def fetch_weather_enhanced(latitude: float, longitude: float, days: int = 1) -> Dict:
    """Fetch weather from Open-Meteo with enhanced parameters including UV index"""
    params = {
    "latitude": latitude,
    "longitude": longitude,
    "current": [
        "temperature_2m", 
        "relative_humidity_2m", 
        "precipitation", 
        "weather_code", 
        "wind_speed_10m",
        "uv_index",
        "uv_index_clear_sky",
        "apparent_temperature",
        "surface_pressure"
    ],
    "hourly": [
        "temperature_2m", 
        "relative_humidity_2m", 
        "precipitation",
        "weather_code"
    ],
    "daily": [
        "temperature_2m_max", 
        "temperature_2m_min", 
        "precipitation_sum", 
        "precipitation_probability_max",
        "uv_index_max",
        "weather_code"
    ],
    "forecast_days": days,
    "timezone": "auto",
} 
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            resp = await client.get(OPEN_METEO_BASE, params=params)
            resp.raise_for_status()
            data = resp.json()
            
            current = data.get("current", {})
            temp = current.get("temperature_2m")
            humidity = current.get("relative_humidity_2m")
            wind_speed = current.get("wind_speed_10m")
            
            if temp is not None and humidity is not None:
                if temp >= 27:
                    feels_like = calculate_heat_index(temp, humidity)
                elif temp <= 10 and wind_speed is not None:
                    feels_like = calculate_wind_chill(temp, wind_speed)
                else:
                    feels_like = temp
                data["current"]["feels_like"] = feels_like
            
            return data
        except httpx.HTTPError as e:
            logger.error(f"Weather API error: {e}")
            raise HTTPException(status_code=502, detail=f"Weather provider error: {e}")

async def fetch_openweather_supplement(latitude: float, longitude: float) -> Dict:
    """Fetch additional data from OpenWeatherMap (optional supplement)"""
    if not OPENWEATHER_API_KEY:
        return {}
    
    params = {
        "lat": latitude,
        "lon": longitude,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric"
    }
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(OPENWEATHER_BASE, params=params)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as e:
            logger.warning(f"OpenWeather supplement failed: {e}")
            return {}

def get_weather_description(code: Optional[int]) -> str:
    if code is None:
        return "Unknown"
    weather_codes = {
        0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
        45: "Foggy", 48: "Depositing rime fog",
        51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
        61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
        71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
        80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
        95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail"
    }
    return weather_codes.get(code, "Unknown")

def safe_float(value, default=0.0) -> float:
    """Safely convert value to float"""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def generate_agro_advice(weather_data: Dict, crop: Optional[str] = None, period: str = "day") -> AdviceResponse:
    current = weather_data.get("current", {}) or {}
    daily = weather_data.get("daily", {}) or {}

    # Safely extract values
    temp = safe_float(current.get("temperature_2m"), None)
    humidity = safe_float(current.get("relative_humidity_2m"), None)
    precip = safe_float(current.get("precipitation"), 0.0)
    wind = safe_float(current.get("wind_speed_10m"), 0.0)
    weather_code = current.get("weather_code")
    uv = safe_float(current.get("uv_index") or current.get("uv_index_clear_sky"), None)
    feels_like = safe_float(current.get("feels_like"), None)

    categories = {
        "Rainfall Alerts": [],
        "Temperature Alerts": [],
        "Wind Alerts": [],
        "Solar Radiation Alerts": [],
        "Humidity and Disease Alerts": [],
        "Severe Weather Alerts": [],
        "Agro-Meteorological Alerts": []
    }

    recs: List[str] = []
    activities: List[str] = []

    # Temperature analysis
    if temp is not None:
        if temp < 5:
            categories["Temperature Alerts"].append("🥶 FROST RISK - Protect sensitive crops immediately")
            recs.append("Cover seedlings with frost cloth or plastic sheets")
            recs.append("Avoid irrigation in the evening to prevent ice formation")
        elif temp > 35:
            categories["Temperature Alerts"].append("🔥 EXTREME HEAT - High crop stress risk")
            recs.append("Irrigate early morning (4-6 AM) or late evening (6-8 PM)")
            recs.append("Apply mulch to retain soil moisture and reduce temperature")
            if feels_like and feels_like > temp + 3:
                recs.append(f"Feels like {feels_like}°C - ensure worker safety with frequent breaks")
        elif temp > 30:
            recs.append("Monitor for heat stress in crops, especially during flowering")
        elif 15 <= temp <= 28:
            activities.append("✓ Ideal temperature for planting and field work")
    
    # Precipitation analysis
    if precip > 5:
        categories["Rainfall Alerts"].append("🌧️ HEAVY RAIN - Delay field operations")
        recs.append("Postpone pesticide and fertilizer application")
        recs.append("Avoid harvesting - wet crops are prone to disease")
        recs.append("Check drainage systems and prevent waterlogging")
    elif precip > 2:
        categories["Rainfall Alerts"].append("💧 Light to moderate rain expected - beneficial for crops")
        activities.append("✓ Good time for transplanting seedlings")
    elif precip < 0.5 and humidity and humidity < 40:
        categories["Agro-Meteorological Alerts"].append("☀️ DRY CONDITIONS - Monitor soil moisture")
        recs.append("Schedule irrigation based on crop water requirements")
        recs.append("Consider drip irrigation to conserve water")

    # Humidity analysis
    if humidity is not None:
        if humidity > 85:
            categories["Humidity and Disease Alerts"].append("💧 HIGH HUMIDITY - Disease risk increased")
            recs.append("Monitor for fungal diseases (leaf blight, rust, mildew)")
            recs.append("Ensure good air circulation in crops")
            recs.append("Avoid overhead irrigation")
        elif humidity < 30:
            recs.append("Low humidity - increase irrigation frequency")

    # Wind analysis
    if wind > 40:
        categories["Wind Alerts"].append("💨 STRONG WINDS - Protect tall crops")
        recs.append("Avoid spraying pesticides or fertilizers")
        recs.append("Provide support stakes for tall plants")
    elif wind > 25:
        categories["Wind Alerts"].append("🌬️ Moderate winds - delay fine spray applications")
    elif wind < 10:
        activities.append("✓ Calm conditions - ideal for spraying")

    # UV/Solar radiation analysis
    if uv is not None:
        if uv >= 8:
            categories["Solar Radiation Alerts"].append("☀️ Very high UV index - protective measures recommended")
            recs.append("Consider shading sensitive seedlings during peak sun hours (11 AM - 3 PM)")
            recs.append("Ensure workers use sun protection (hats, sunscreen)")
        elif uv >= 6:
            categories["Solar Radiation Alerts"].append("☀️ High UV index - monitor crop stress")

    # Severe weather
    if weather_code in (95, 96, 99):
        categories["Severe Weather Alerts"].append("⚡ THUNDERSTORM/HAIL RISK - Avoid outdoor operations")
        recs.append("Seek shelter and delay field work immediately")
        recs.append("Secure lightweight structures and equipment")

    # Crop-specific recommendations
    crop_recs: List[str] = []
    if crop:
        crop_lower = crop.lower()
        if crop_lower in {"wheat", "barley", "oats"}:
            crop_recs.append("🌾 Cereal crops: Monitor for rusts after wet periods")
            if temp is not None and 18 <= temp <= 24:
                activities.append("✓ Optimal temperature for cereal growth")
        elif crop_lower in {"rice", "palay"}:
            crop_recs.append("🌾 Rice: Maintain 5-10cm water depth in vegetative stage")
            if humidity and humidity > 80:
                crop_recs.append("Watch for bacterial leaf blight and blast disease")
        elif crop_lower in {"corn", "maize"}:
            crop_recs.append("🌽 Corn: Critical water need during tasseling and silking")
            if temp and temp > 32:
                crop_recs.append("High heat can affect pollination - ensure adequate moisture")
        elif crop_lower in {"tomato", "eggplant", "pepper"}:
            crop_recs.append("🍅 Solanaceae crops: Stake plants and apply mulch")
            if humidity and humidity > 80:
                crop_recs.append("High risk of early and late blight - apply fungicides preventively")
        elif crop_lower in {"cabbage", "lettuce", "cauliflower"}:
            if temp and temp > 28:
                crop_recs.append("🥬 Cool season crops stressed - provide shade if possible")
        elif crop_lower in {"mango", "banana", "coconut"}:
            crop_recs.append(f"🌴 {crop}: Ensure proper nutrition and pest monitoring")

    weather_desc = get_weather_description(weather_code)
    if "clear" in weather_desc.lower() or "mainly clear" in weather_desc.lower():
        activities.append("✓ Excellent conditions for harvesting")
        activities.append("✓ Good for land preparation and plowing")

    # Compile final alerts
    final_alerts: List[str] = []
    for cat in ["Severe Weather Alerts", "Temperature Alerts", "Rainfall Alerts", "Wind Alerts", 
                "Humidity and Disease Alerts", "Solar Radiation Alerts", "Agro-Meteorological Alerts"]:
        items = categories.get(cat, [])
        if items:
            for item in items:
                final_alerts.append(item)

    if not final_alerts:
        final_alerts = ["✅ No weather alerts - Favorable conditions"]

    final_recs = recs[:]
    final_recs.extend([r for r in crop_recs if r not in final_recs])

    if not final_recs:
        final_recs = ["Monitor weather conditions regularly and adjust farming activities accordingly"]

    # Create summary
    summary_temp = f"{temp}°C" if temp is not None else "N/A"
    summary_hum = f"{humidity}%" if humidity is not None else "N/A"
    feels_text = f" (feels like {feels_like}°C)" if feels_like and abs(feels_like - temp) > 2 else ""
    summary = f"Weather advisory for {period} - Current: {weather_desc}, {summary_temp}{feels_text}, {summary_hum} humidity"

    return AdviceResponse(
        summary=summary,
        recommendations=final_recs,
        weather_alerts=final_alerts,
        best_activities=activities if activities else ["Continue routine farm monitoring"]
    )

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two coordinates using Haversine formula (in km)"""
    R = 6371  # Earth's radius in kilometers
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    return R * c

# API Routes
@app.get("/")
async def read_root():
    index_path = BASE_DIR / "static" / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "AgroWeather API is running", "version": "1.0.0"}

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

@app.post("/api/signup", response_model=AuthResponse)
async def signup(req: SignUpRequest):
    conn = get_db()
    c = conn.cursor()

    existing = c.execute('SELECT id FROM farmers WHERE username = ?', (req.username,)).fetchone()
    if existing:
        conn.close()
        raise HTTPException(status_code=400, detail="Username already exists")

    password_hash = hash_password(req.password)
    created_at = datetime.now().isoformat()

    c.execute('''INSERT INTO farmers (name, age, gender, location, username, password_hash, created_at)
                 VALUES (?, ?, ?, ?, ?, ?, ?)''',
              (req.name, req.age, req.gender, req.location, req.username, password_hash, created_at))

    user_id = c.lastrowid

    token = create_session_token()
    expires_at = (datetime.now() + timedelta(days=7)).isoformat()

    c.execute('INSERT INTO sessions (token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)',
              (token, user_id, created_at, expires_at))

    conn.commit()

    user = c.execute('SELECT id, name, age, gender, location, username, is_admin FROM farmers WHERE id = ?',
                     (user_id,)).fetchone()

    conn.close()  

    user_dict = dict(user)
    user_dict['is_admin'] = bool(user['is_admin']) if 'is_admin' in user.keys() else False

    return AuthResponse(token=token, user=user_dict)

@app.post("/api/login", response_model=AuthResponse)
async def login(req: LoginRequest):
    conn = get_db()
    c = conn.cursor()

    password_hash = hash_password(req.password)
    user = c.execute('SELECT * FROM farmers WHERE username = ? AND password_hash = ?',
                     (req.username, password_hash)).fetchone()

    if not user:
        conn.close()
        raise HTTPException(status_code=401, detail="Invalid username or password")

    token = create_session_token()
    created_at = datetime.now().isoformat()
    expires_at = (datetime.now() + timedelta(days=7)).isoformat()

    c.execute('INSERT INTO sessions (token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)',
              (token, user['id'], created_at, expires_at))

    conn.commit()
    conn.close()

    user_dict = {
        'id': user['id'],
        'name': user['name'],
        'age': user['age'],
        'gender': user['gender'],
        'location': user['location'],
        'username': user['username'],
        'is_admin': bool(user['is_admin']) if 'is_admin' in user.keys() else False
    }

    return AuthResponse(token=token, user=user_dict)

@app.post("/api/logout")
async def logout(user: dict = Depends(get_current_user), authorization: str = Header(...)):
    token = authorization.split(' ')[1]
    conn = get_db()
    c = conn.cursor()
    c.execute('DELETE FROM sessions WHERE token = ?', (token,))
    conn.commit()
    conn.close()
    return {"message": "Logged out successfully"}

@app.get("/api/me")
async def get_me(user: dict = Depends(get_current_user)):
    return {
        'id': user['id'],
        'name': user['name'],
        'age': user['age'],
        'gender': user['gender'],
        'location': user['location'],
        'username': user['username'],
        'is_admin': bool(user['is_admin']) if 'is_admin' in user.keys() else False
    }

@app.get("/api/weather", response_model=WeatherResponse)
async def get_weather(latitude: float, longitude: float, user: dict = Depends(get_current_user)):
    data = await fetch_weather_enhanced(latitude, longitude, days=7)  # Changed to 7 days
    current = data.get("current", {})
    hourly = data.get("hourly", {})
    daily = data.get("daily", {})
    
    # Build daily forecast data
    daily_forecast = []
    if daily and daily.get("time"):
        times = daily.get("time", [])
        max_temps = daily.get("temperature_2m_max", [])
        min_temps = daily.get("temperature_2m_min", [])
        codes = daily.get("weather_code", [])
        
        for i in range(min(len(times), 7)):
            daily_forecast.append({
                "date": times[i],
                "temp_max": max_temps[i] if i < len(max_temps) else None,
                "temp_min": min_temps[i] if i < len(min_temps) else None,
                "weather_code": codes[i] if i < len(codes) else None
            })

    return WeatherResponse(
        latitude=data.get("latitude", latitude),
        longitude=data.get("longitude", longitude),
        temperature=current.get("temperature_2m"),
        windspeed=current.get("wind_speed_10m"),
        precipitation=current.get("precipitation"),
        humidity=current.get("relative_humidity_2m"),
        weather_code=current.get("weather_code"),
        weather_description=get_weather_description(current.get("weather_code")),
        uv_index=current.get("uv_index"),
        feels_like=current.get("feels_like"),
        pressure=current.get("surface_pressure"),        # ADD THIS
        visibility=10.0,                                  # ADD THIS (default)
        hourly_times=hourly.get("time", []) or [],
        hourly_temp=hourly.get("temperature_2m", []) or [],
        hourly_precip=hourly.get("precipitation", []) or [],
        hourly_humidity=hourly.get("relative_humidity_2m", []) or [],
        hourly_weather_code=hourly.get("weather_code", []) or [],  # ADD THIS
        daily=daily_forecast                              # ADD THIS
    )

@app.get("/api/prediction/{period}", response_model=PredictionResponse)
async def get_prediction(period: str, latitude: float, longitude: float, user: dict = Depends(get_current_user)):
    days_map = {"day": 1, "week": 7, "month": 16}
    days = days_map.get(period, 7)

    data = await fetch_weather_enhanced(latitude, longitude, days=days)
    daily = data.get("daily", {})

    temps = daily.get("temperature_2m_max", [])
    precips = daily.get("precipitation_sum", [])
    times = daily.get("time", [])

    forecast_data = []
    for i in range(len(times)):
        forecast_data.append({
            "date": times[i],
            "temp_max": temps[i] if i < len(temps) else None,
            "precipitation": precips[i] if i < len(precips) else None
        })

    avg_temp = sum(t for t in temps if t is not None) / len(temps) if temps else 0
    total_precip = sum(p for p in precips if p is not None) if precips else 0

    return PredictionResponse(
        period=period,
        avg_temperature=round(avg_temp, 1),
        total_precipitation=round(total_precip, 1),
        avg_humidity=70.0,
        forecast_data=forecast_data
    )

@app.post("/api/advice", response_model=AdviceResponse)
async def get_advice(
    req: AdviceRequest, 
    latitude: float, 
    longitude: float, 
    user: dict = Depends(get_current_user)
):
    days_map = {"day": 1, "week": 7, "month": 16}
    days = days_map.get(req.period, 1)

    weather_data = await fetch_weather_enhanced(latitude, longitude, days=days)
    return generate_agro_advice(weather_data, crop=req.crop, period=req.period)

@app.post("/api/notifications/create")
async def create_notification(
    req: NotificationCreate,
    user: dict = Depends(get_current_user)
):
    # Check if user is admin
    if not user.get('is_admin'):
        raise HTTPException(status_code=403, detail="Admin access required")
    
    conn = get_db()
    c = conn.cursor()
    
    created_at = datetime.now().isoformat()
    
    # Create notification
    c.execute('''INSERT INTO notifications (title, message, type, created_at, created_by)
                 VALUES (?, ?, ?, ?, ?)''',
              (req.title, req.message, req.type, created_at, user['id']))
    
    notif_id = c.lastrowid
    
    # Send to all users
    users = c.execute('SELECT id FROM farmers').fetchall()
    for u in users:
        c.execute('''INSERT INTO user_notifications (user_id, notification_id)
                     VALUES (?, ?)''', (u['id'], notif_id))
    
    conn.commit()
    conn.close()
    
    return {"message": "Notification sent to all users", "id": notif_id}

@app.get("/api/notifications")
async def get_notifications(user: dict = Depends(get_current_user)):
    conn = get_db()
    c = conn.cursor()
    
    notifications = c.execute('''
        SELECT 
            n.id as notification_id,
            n.title, 
            n.message, 
            n.type, 
            n.created_at,
            un.is_read, 
            un.read_at
        FROM notifications n
        JOIN user_notifications un ON n.id = un.notification_id
        WHERE un.user_id = ? AND n.is_active = 1
        ORDER BY n.created_at DESC
        LIMIT 50
    ''', (user['id'],)).fetchall()
    
    conn.close()
    
    return [dict(n) for n in notifications]

@app.post("/api/notifications/{notif_id}/read")
async def mark_notification_read(
    notif_id: int,
    user: dict = Depends(get_current_user)
):
    conn = get_db()
    c = conn.cursor()
    
    c.execute('''UPDATE user_notifications 
                 SET is_read = 1, read_at = ?
                 WHERE user_id = ? AND notification_id = ?''',
              (datetime.now().isoformat(), user['id'], notif_id))
    
    conn.commit()
    conn.close()
    
    return {"message": "Notification marked as read"}

@app.get("/api/notifications/unread/count")
async def get_unread_count(user: dict = Depends(get_current_user)):
    conn = get_db()
    c = conn.cursor()
    
    count = c.execute('''
        SELECT COUNT(*) as count
        FROM user_notifications un
        JOIN notifications n ON un.notification_id = n.id
        WHERE un.user_id = ? AND un.is_read = 0 AND n.is_active = 1
    ''', (user['id'],)).fetchone()
    
    conn.close()
    
    return {"count": count['count'] if count else 0}

@app.post("/api/make-admin/{username}")
async def make_admin(username: str):
    conn = get_db()
    c = conn.cursor()
    
    c.execute('UPDATE farmers SET is_admin = 1 WHERE username = ?', (username,))
    conn.commit()
    
    affected = c.total_changes
    conn.close()
    
    if affected == 0:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {"message": f"User {username} is now an admin"}
# ==================== HISTORICAL EVENTS API ====================

@app.post("/api/historical-events/create")
async def create_historical_event(
    req: HistoricalEventCreate,
    user: dict = Depends(get_current_user)
):
    """Admin only: Create a new historical event record"""
    if not user.get('is_admin'):
        raise HTTPException(status_code=403, detail="Admin access required")
    
    conn = get_db()
    c = conn.cursor()
    
    created_at = datetime.now().isoformat()
    
    c.execute('''INSERT INTO historical_events 
                 (event_name, event_type, event_date, latitude, longitude, 
                  severity, description, affected_areas, casualties, 
                  damage_estimate, source, created_at)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (req.event_name, req.event_type, req.event_date, req.latitude, 
               req.longitude, req.severity, req.description, req.affected_areas,
               req.casualties, req.damage_estimate, req.source, created_at))
    
    event_id = c.lastrowid
    conn.commit()
    conn.close()
    
    return {"message": "Historical event created", "id": event_id}


@app.get("/api/historical-events", response_model=List[HistoricalEventResponse])
async def get_historical_events(
    latitude: float,
    longitude: float,
    radius_km: float = 100.0,
    limit: int = 20,
    event_type: Optional[str] = None,
    ):
    """Get historical events within radius of given coordinates"""
    conn = get_db()
    c = conn.cursor()
    
    # Fetch all events (we'll filter by distance in Python)
    if event_type:
        events = c.execute('''SELECT * FROM historical_events 
                             WHERE event_type = ?
                             ORDER BY event_date DESC''', 
                          (event_type,)).fetchall()
    else:
        events = c.execute('''SELECT * FROM historical_events 
                             ORDER BY event_date DESC''').fetchall()
    
    conn.close()
    
    # Calculate distances and filter
    results = []
    for event in events:
        distance = calculate_distance(
            latitude, longitude,
            event['latitude'], event['longitude']
        )
        
        if distance <= radius_km:
            event_dict = dict(event)
            event_dict['distance_km'] = round(distance, 2)
            results.append(event_dict)
    
    # Sort by distance and limit
    results.sort(key=lambda x: x['distance_km'])
    return results[:limit]


@app.get("/api/historical-events/stats")
async def get_historical_stats(
    latitude: float,
    longitude: float,
    radius_km: float = 100.0,
):
    """Get statistics about historical events in the area"""
    conn = get_db()
    c = conn.cursor()
    
    events = c.execute('SELECT * FROM historical_events').fetchall()
    conn.close()
    
    nearby_events = []
    for event in events:
        distance = calculate_distance(
            latitude, longitude,
            event['latitude'], event['longitude']
        )
        if distance <= radius_km:
            nearby_events.append(dict(event))
    
    # Calculate statistics
    stats = {
        "total_events": len(nearby_events),
        "by_type": {},
        "by_severity": {},
        "total_casualties": 0,
        "most_recent": None
    }
    
    for event in nearby_events:
        # Count by type
        event_type = event['event_type']
        stats['by_type'][event_type] = stats['by_type'].get(event_type, 0) + 1
        
        # Count by severity
        severity = event['severity']
        stats['by_severity'][severity] = stats['by_severity'].get(severity, 0) + 1
        
        # Sum casualties
        stats['total_casualties'] += event.get('casualties', 0)
    
    # Most recent event
    if nearby_events:
        most_recent = max(nearby_events, key=lambda x: x['event_date'])
        stats['most_recent'] = {
            "name": most_recent['event_name'],
            "date": most_recent['event_date'],
            "type": most_recent['event_type']
        }
    
    return stats


@app.delete("/api/historical-events/{event_id}")
async def delete_historical_event(
    event_id: int,
    user: dict = Depends(get_current_user)
):
    """Admin only: Delete a historical event"""
    if not user.get('is_admin'):
        raise HTTPException(status_code=403, detail="Admin access required")
    
    conn = get_db()
    c = conn.cursor()
    
    c.execute('DELETE FROM historical_events WHERE id = ?', (event_id,))
    affected = c.total_changes
    
    conn.commit()
    conn.close()
    
    if affected == 0:
        raise HTTPException(status_code=404, detail="Event not found")
    
    return {"message": "Event deleted successfully"}

@app.post("/api/notifications/broadcast")
async def broadcast_notification(
    req: NotificationCreate,
    user: dict = Depends(get_current_user)
):
    """Admin only: Broadcast notification to ALL users"""
    if not user.get('is_admin'):
        raise HTTPException(status_code=403, detail="Admin access required")

    conn = get_db()
    c = conn.cursor()

    created_at = datetime.now().isoformat()

    # Create notification
    c.execute('''INSERT INTO notifications (title, message, type, created_at, created_by)
        VALUES (?, ?, ?, ?, ?)''',
        (req.title, req.message, req.type, created_at, user['id']))

    notif_id = c.lastrowid

    # Send to ALL users (including admin)
    users = c.execute('SELECT id FROM farmers').fetchall()
    for u in users:
        c.execute('''INSERT INTO user_notifications (user_id, notification_id, is_read)
            VALUES (?, ?, 0)''', (u['id'], notif_id))

    conn.commit()
    affected = len(users)
    conn.close()

    logger.info(f"📤 Broadcast notification sent to {affected} users by admin {user['username']}")

    return {
        "message": f"Notification sent to {affected} users",
        "id": notif_id,
        "recipients": affected
    }
    
@app.get("/api/admin/users")
async def get_all_users(user: dict = Depends(get_current_user)):
    """Get list of all registered users with online status"""
    if not user.get('is_admin'):
        raise HTTPException(status_code=403, detail="Admin access required")

    conn = get_db()
    c = conn.cursor()

    users = c.execute('''
        SELECT id, name, username, age, gender, location, is_admin, created_at, last_seen
        FROM farmers
        ORDER BY created_at DESC
    ''').fetchall()

    conn.close()

    # Convert to list of dicts and add online status
    result = []
    now = datetime.now()
    
    for user_row in users:
        user_dict = dict(user_row)
        
        # Determine if online (active within last 5 minutes)
        if user_dict.get('last_seen'):
            last_seen = datetime.fromisoformat(user_dict['last_seen'])
            diff_minutes = (now - last_seen).total_seconds() / 60
            user_dict['is_online'] = diff_minutes < 5
        else:
            user_dict['is_online'] = False
            
        result.append(user_dict)
    
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("agroweather_backend:app", host="0.0.0.0", port=8000, reload=True)