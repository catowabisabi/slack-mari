import os
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from slack_bolt.adapter.flask import SlackRequestHandler
from slack_bolt import App
from dotenv import find_dotenv, load_dotenv
from flask import Flask, request
from functions import get_GPT_response, get_youtube_summary
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

def is_youtube_url(text):
    # 簡單的正則表達式，用於檢查文本是否為 YouTube URL
    pattern = r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})'
    match = re.match(pattern, text)
    return bool(match)


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
    
    elif is_youtube_url(text):
        say("收到, 宜家幫你望望個Youtube, 之後同你講返...")
        try:
            title, dialogue, en_summary, zh_summary, cn_summary = get_youtube_summary(text)
            time.sleep(5)
            if not title and not dialogue and not en_summary and not zh_summary and not cn_summary:
                time.sleep(2)
                say("Sorry呀... 好似唔係咁得...你不如搵下其他既影片?")
                return
            if title:
                time.sleep(2)
                say(f"Youtube的題目為: {title}")
            if dialogue:
                time.sleep(2)
                say(f"Youtube的Caption為: \n{dialogue}")
            if zh_summary:
                time.sleep(2)
                say(f"Youtube的中文總結為: \n{zh_summary}")
            if cn_summary:
                time.sleep(2)
                say(f"Youtube的英文總結為: \n{en_summary}")
            
        except Exception as e:
            time.sleep(2)
            print("Error: ", e)
            say("Sorry呀... 我搵唔到呢個影片既字幕。你不如搵下其他既影片?")

    else: 

    
    # response = my_function(text)
    #response = draft_email(text)
        response = get_GPT_response(text, OPENAI_API_KEY)
        time.sleep(6)
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




