from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
LOGS = PROJECT_ROOT / "logs"
DB_PATH = PROJECT_ROOT / "data" / "energy.db"

for folder in (DATA_RAW, DATA_PROCESSED, LOGS):
    folder.mkdir(parents=True, exist_ok=True)

TIMEZONE_STORAGE = "UTC"
TIMEZONE_DISPLAY = "Australia/Brisbane"

WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")