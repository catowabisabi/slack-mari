import os
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from dotenv import find_dotenv, load_dotenv


# Load environment variables from .env file
load_dotenv(find_dotenv())

# Set Slack API credentials
SLACK_BOT_TOKEN2 = os.environ["SLACK_BOT_TOKEN2"]
SLACK_SIGNING_SECRET2 = os.environ["SLACK_SIGNING_SECRET2"]
SLACK_BOT_USER_ID2 = os.environ["SLACK_BOT_USER_ID2"]



def get_bot_user_id():
    """
    Get the bot user ID using the Slack API.
    Returns:
        str: The bot user ID.
    """
    try:
        # Initialize the Slack client with your bot token
        slack_client = WebClient(token=os.environ["SLACK_BOT_TOKEN"])
        response = slack_client.auth_test()
        return response["user_id"]
    except SlackApiError as e:
        print(f"Error: {e}")

user_id = get_bot_user_id()
print (user_id)