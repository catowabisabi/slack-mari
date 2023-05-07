import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT, register_adapter, AsIs
import uuid
import os
from dotenv import load_dotenv, find_dotenv






# Load environment variables from .env file
load_dotenv(find_dotenv())

# Set Slack API credentials
DB_NAME = os.environ["DB_NAME"]
DB_USER_NAME = os.environ["DB_USER_NAME"]
DB_PASSWORD = os.environ["DB_PASSWORD"]
DB_HOST = os.environ["DB_HOST"]
DB_PORT = os.environ["DB_PORT"]


# Register UUID type adapter for psycopg2
register_adapter(uuid.UUID, AsIs)


class YouTubeData:
    def __init__(self, data):
        self.data = data

    def insert_data(self):
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER_NAME,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )

        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()

        insert_query = sql.SQL("""
            INSERT INTO youtube (id, title, url, zh_summary, en_summary, cn_summary, zh_dialogue, dialogue)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (url)
            DO UPDATE SET
                title = EXCLUDED.title,
                zh_summary = EXCLUDED.zh_summary,
                en_summary = EXCLUDED.en_summary,
                cn_summary = EXCLUDED.cn_summary,
                zh_dialogue = EXCLUDED.zh_dialogue,
                dialogue = EXCLUDED.dialogue,
                updated_date = NOW();
        """)

        cur.execute(insert_query, (
            str(uuid.uuid4()),
            self.data['title'],
            self.data['url'],
            self.data['zh_summary'],
            self.data['en_summary'],
            self.data['cn_summary'],
            self.data['zh_dialogue'],
            self.data['dialogue']
        ))

        cur.close()
        conn.close()


""" data = {
    'title': "This is a title of a video",
    'url': "https://www.youtube.com/watch?v=123456789010",
    'zh_summary': '繁體中文摘要',
    'en_summary': '英文摘要',
    'cn_summary': '简体中文摘要',
    'zh_dialogue': '繁體中文對話',
    'dialogue': "英文對話2"
} """

# 使用示例
""" data_obj = YouTubeData(data=data)
data_obj.insert_data() """
