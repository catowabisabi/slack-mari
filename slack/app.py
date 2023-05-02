import os
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from slack_bolt.adapter.flask import SlackRequestHandler
from slack_bolt import App
from dotenv import find_dotenv, load_dotenv
from flask import Flask, request
from functions import draft_email, get_GPT_response
import random
from greetings import replies, questions
import time






# Load environment variables from .env file
load_dotenv(find_dotenv())

# Set Slack API credentials
SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_SIGNING_SECRET = os.environ["SLACK_SIGNING_SECRET"]
SLACK_BOT_USER_ID = os.environ["SLACK_BOT_USER_ID"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# Initialize the Slack app
app = App(token=SLACK_BOT_TOKEN)

# Initialize the Flask app
# Flask is a web application framework written in Python
flask_app = Flask(__name__)
handler = SlackRequestHandler(app)


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



def get_first_message():
    first_sentence = random.choice(replies)
    second_sentence = random.choice(questions)
    response = f"{first_sentence} {second_sentence}"
    return response

@app.event("app_mention")
def handle_mentions(body, say):
    text = body["event"]["text"]
    mention = f"<@{SLACK_BOT_USER_ID}>"
    text = text.replace(mention, "").strip()
    print(text)
    if text == "":
        time.sleep(6) 
        ans = get_first_message()
        say(ans)
    else: 
    
    # response = my_function(text)
    #response = draft_email(text)
        response = get_GPT_response(text, OPENAI_API_KEY)
        say(response)


@flask_app.route("/slack/events", methods=["POST"])
def slack_events():
    return handler.handle(request)

@flask_app.route("/", methods=["POST"])
def server_running():
    return {
        "message": "Server is running"
    }


# Run the Flask app
if __name__ == "__main__":
    flask_app.run()




