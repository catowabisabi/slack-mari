import os
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from slack_bolt.adapter.flask import SlackRequestHandler
from slack_bolt import App
from dotenv import find_dotenv, load_dotenv


import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


from flask import Flask, request, render_template
from slack_m.functions import get_GPT_response, get_youtube_summary
import random
from slack_m.greetings import replies, questions
import time
import re

import psycopg2







# Load environment variables from .env file
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

# Initialize the Slack app
app = App(token=SLACK_BOT_TOKEN)

# Initialize the Flask app
# Flask is a web application framework written in Python
flask_app = Flask(__name__)
handler = SlackRequestHandler(app)

def get_data():
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER_NAME,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )

    cur = conn.cursor()
    cur.execute("SELECT * FROM youtube;")
    data = cur.fetchall()
    cur.close()
    conn.close()
    return data

@flask_app.route("/youtube")
def youtube():
    data = get_data()
    return render_template("youtube.html", data=data)


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

import re

def yt_obj_to_str(obj):
    return obj.replace('<', '').replace('>', '')

def is_youtube_url(text):
    print("Type of text:", type(text))
    text  = yt_obj_to_str(text)
    print("Processed text:", text)
    
    pattern = r'(\<)?(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})(\>)?'
    pattern2 = r'(\<)?(https?://)?(www\.)?(\S+)?(youtube|youtu|youtube-nocookie)\.(com|be)/(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})(\>)?'
    
    match = re.match(pattern, text)
    match2 = re.match(pattern2, text)
    match_all = match or match2
    
    print("Match result:", match_all)
    return bool(match_all)




@app.event("app_mention")
def handle_mentions(body, say):
    
    text = body["event"]["text"]
    mention = f"<@{SLACK_BOT_USER_ID}>"
    text = text.replace(mention, "").strip()
    print(text)

    # 去掉 @U056CQVSS4B 这样的 Slack 用户 ID
    text = re.sub(r'<@[\w]+>', '', text).strip()


    is_yt = is_youtube_url(text)
    print("is_yt:", is_yt)

    if text == "":
        time.sleep(6) 
        ans = get_first_message()
        say(ans)
    
    elif is_youtube_url(text):
        text = yt_obj_to_str(text)
        say("收到, 宜家幫你望望個Youtube, 之後同你講返...")
        try:
            title, dialogue, en_summary, zh_summary, cn_summary, zh_paraphrase= get_youtube_summary(text)
            time.sleep(5)
            if not title and not dialogue and not en_summary and not zh_summary and not cn_summary and not zh_paraphrase: 
                time.sleep(2)
                say("Sorry呀... 好似唔係咁得...你不如搵下其他既影片?")
                
            if title:
                time.sleep(2)
                say(f"Youtube的題目為: {title}")
            if dialogue:
                time.sleep(2)
                say(f"Youtube的Caption為: \n{dialogue[:100]} ... ")
            if zh_summary:
                time.sleep(2)
                say(f"Youtube的中文總結為: \n{zh_summary}")
            

            if zh_paraphrase:
                time.sleep(2)
                say(f"Youtube的中文內容: \n{zh_paraphrase}")
            
        except Exception as e:
            time.sleep(2)
            print("Error: ", str(e))
            say("Sorry呀... 我搵唔到呢個影片既字幕。你不如搵下其他既影片?")
            return

    else: 
        #response = my_function(text)
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




