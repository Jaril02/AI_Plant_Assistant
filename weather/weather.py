import requests
import streamlit as st
import os
from dotenv import load_dotenv

def get_weather(city="Delhi"):
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(env_path)
    api_key = os.getenv("WEATHER_API")
    if not api_key:
        return {"error": "⚠️ WEATHER_API_KEY not found. Check your .env file."}
    
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"


    try:
        response = requests.get(url,timeout=10)
        data = response.json()
        
        if data.get("cod") != 200:
            return {"error": f"API Error: {data.get('message', 'Unknown error')}"}
        
        weather = {
            "temp": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "description": data["weather"][0]["description"],
            "rain": data.get("rain", {}).get("1h", 0)  
        }
        return weather
    
    except Exception as e:
        return {"error": f"Request failed: {str(e)}"}

def watering_advice(weather):
    if weather is None or "error" in weather:
        return "⚠️ Weather data unavailable. Please check your API key or internet."
    
    advice = []
    
    # Rule 1: Hot weather
    if weather["temp"] > 32:
        advice.append("🌡️ It's very hot. Water your plant more frequently.")
    
    # Rule 2: Rainfall
    if weather["rain"] > 0:
        advice.append("☔ It has rained recently. You may not need to water the plant today.")
    else:
        advice.append("💧 No rainfall detected. Check soil moisture and water if dry.")
    
    # Rule 3: Humidity
    if weather["humidity"] < 40:
        advice.append("🌬️ Air is dry. Consider light watering to keep soil moist.")
    
    if not advice:
        advice.append("✅ Your plant is fine. Regular watering schedule is enough.")
    
    return " ".join(advice)



