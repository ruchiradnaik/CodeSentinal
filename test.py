import os
from dotenv import load_dotenv

load_dotenv()

OPEN_API_KEY = os.getenv("OPEN_API_KEY")
print(f"API Key loaded: {OPEN_API_KEY[:10]}..." if OPEN_API_KEY else "API Key is None!")