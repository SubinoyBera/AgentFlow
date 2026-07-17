import os
import requests
from fastmcp import FastMCP
from utils import covert_to_exact_time
from src.logger import logging
import json
from dotenv import load_dotenv

load_dotenv()

mcp = FastMCP("weather-tool")

@mcp.tool
async def weather_tool(location: str) -> str:
    """
    Weather tool to get the current weather report of any given location.
    Args:
        location (str): Name of the location to get the weather report for.
    """
    try:
        weather_api = os.getenv("WEATHER_API_KEY")
        loc = location.split(":")[-1].strip(' "{}')
        lat_long_url = f"http://api.openweathermap.org/geo/1.0/direct?q={loc}&limit=1&appid={weather_api}"
        
        response = requests.get(lat_long_url).json()
        
        lat = response[0]["lat"]
        lon = response[0]["lon"]
        weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={weather_api}&units=metric"
        
        weather = requests.get(weather_url).json()
        sunrise_utc, sunset_utc, tz_offset, dt_utc = (
            weather["sys"]["sunrise"], weather["sys"]["sunset"], weather["timezone"], weather["dt"]
        )
        sunrise, sunset, report_time = covert_to_exact_time(sunrise_utc, sunset_utc, tz_offset, dt_utc)
        
        report = {
            "main": weather["weather"][0]["main"],
            "description": weather["weather"][0]["description"],
            "conditions": weather["main"],
            "visibility": weather["visibility"],
            "wind": weather["wind"],
            "clouds": weather["clouds"],
            "extras": {"sunrise": sunrise, "sunset": sunset, "report_time": report_time, "country": weather["sys"]["country"]},
        }
        
        logging.info("weather_tool called, and tool results obtained")
        return json.dumps(report)
    
    except Exception as e:
        logging.error(f"weather_tool failed: {e}")
        return json.dumps({
            "weather_tool_error": "Weather tool failed to get weather report"
        })


if __name__ == "__main__":
    mcp.run(transport="stdio")