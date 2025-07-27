import requests
import os
from dotenv import load_dotenv
from pathlib import Path

# Load .env file
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# Get the API key from .env
api_key = os.getenv("GNEWS_API_KEY")

if not api_key:
    print("❌ GNEWS_API_KEY not found in .env file.")
    exit()

# GNews API endpoint
url = f"https://gnews.io/api/v4/top-headlines?lang=en&country=in&max=5&token={api_key}"

try:
    response = requests.get(url)
    data = response.json()

    if "articles" in data:
        print("📰 Top 5 News Headlines from GNews:\n")
        for i, article in enumerate(data["articles"], start=1):
            print(f"{i}. Title: {article['title']}")
            print(f"   Description: {article.get('description', 'No description')}\n")
    else:
        print("❌ No articles found. Response:\n", data)
except Exception as e:
    print("❌ Error fetching news:", e)
