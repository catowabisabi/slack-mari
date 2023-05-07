import os
import sys
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

load_dotenv(find_dotenv())

# Set Slack API credentials
SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_SIGNING_SECRET = os.environ["SLACK_SIGNING_SECRET"]
SLACK_BOT_USER_ID = os.environ["SLACK_BOT_USER_ID"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# Set Slack API credentials
DB_NAME = os.environ["DB_NAME"]
DB_USER_NAME = os.environ["DB_USER_NAME"]
DB_PASSWORD = os.environ["DB_PASSWORD"]
DB_HOST = os.environ["DB_HOST"]
DB_PORT = os.environ["DB_PORT"]

SERPAPI_API_KEY = os.environ["SERPAPI_API_KEY"]