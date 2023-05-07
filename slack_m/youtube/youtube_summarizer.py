from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, VideoUnavailable, TranscriptsDisabled
from google.cloud import speech_v1p1beta1 as speech
import youtube_dl
from collections import OrderedDict
from pytube import YouTube

import json
from typing import Any
import requests
from bs4 import BeautifulSoup
import re
import csv
import pandas as pd
import openpyxl
# 在文件開頭添加以下引入
import io


""" import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))) """


import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from slack_m.my_llms.my_llms import LLM_OpenAI
from slack_m.split.markdown_split import MDSplit
from slack_m.summarizer.fx_llms import DocsSummarizer #語言模型
from slack_m.summarizer.fx_long_text_summarizer import LongTextParaphaser

from langchain.text_splitter import RecursiveCharacterTextSplitter

from database.save_json import YouTubeData


def pprint(text):
    print("\n========\n"+str(text)+"\n\n========\n\n\n")


class YoutubeSummarizer:

    def pprint(self, text):
        print("\n========\n"+str(text)+"\n\n========\n\n\n")

    def __init__(self, url, tech_sum = True, to_zh=False):
        self.dialogue = None
        self.audio = None
        self.transcript = None
        self.audio_file = None
        self.dialogue = None
        self.summary = None
        self.to_zh = to_zh
        self.tech_sum = tech_sum
        self.zh_text = None
        self.video_captions=None
        self.data = None
        self.filename = None
        self.title = None
        self.zh_paraphrase = None

        self.url = url
        self.video_id = self.get_youtube_id(self.url) 
                
    # 獲取 YouTube 視頻 ID
    def get_youtube_id(self, link):
        self.pprint("Getting YouTube video ID...")
        """
        從 YouTube 鏈接中獲取視頻 ID。
        """
        # 匹配 YouTube 鏈接的正則表達式
        regex = r'(https?://)?(www\.)?youtu(be\.com/watch\?v=|\.be/)([\w-]+)'
        match = re.match(regex, link)
        if match:
            # 從鏈接中提取視頻 ID
            #print(match.group(4))
            return match.group(4)
        else:
            # 如果鏈接無效，返回空字符串
            return ''

    # 下載音頻文件
    def download_youtube_audio(self, _video_id):#"XLG-qtZwxIw"  tFsUuvlYyqE
        if not _video_id:
            _video_id = self.video_id
        self.pprint("Downloading audio file...")
        id_cleaned = re.sub(r'\W+', '', self.video_id)
        filename = f"{id_cleaned}.mp4"  # pytube 下載的音頻文件將為 mp4 格式
        url = f"https://www.youtube.com/watch?v={_video_id}"
        
        # 建立 YouTube 物件
        yt = YouTube(url)

        try:
            # 選擇最高品質的音頻
            self.audio = yt.streams.filter(only_audio=True).first()
            self.audio = self.audio.download(output_path='audio_folder', filename=filename)
            self.pprint("Finished downloading audio file.")
            return self.audio
        
        except Exception as e:
            self.pprint("Error downloading audio file." + str(e))
            self.audio = None
            return self.audio
        
        

    # 將音頻轉換為文字
    def transcribe_audio_file(self, file_path):
        
        try:
            self.pprint("Transcribing audio file...")
            client = speech.SpeechClient()

            with io.open(file_path, "rb") as audio_file:
                content = audio_file.read()

            audio = speech.RecognitionAudio(content=content)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code="zh-TW",
            )

            response = client.recognize(config=config, audio=audio)

            self.transcript = ''
            for result in response.results:
                self.transcript += result.alternatives[0].transcript

            self.pprint("Finished transcribing audio file.")
            return self.transcript
        
        except Exception as e:
            self.pprint("Error transcribing audio file." + str(e))
            self.transcript = None
            return self.transcript


    def get_youtube_title(self, _video_id=None):
        try:
            if not _video_id:
                _video_id = self.video_id
            url = f'https://www.youtube.com/watch?v={_video_id}'

            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            title = soup.title.string.replace(" - YouTube", "") # 影片標題
            return title
        except Exception as e:
            self.pprint("Error getting youtube title." + str(e))
            return None
        
    # 從 YouTube 影片 ID 取得字幕請單
    def get_transcript_list(self):
        try:
            self.pprint("Retrieving captions...")
            transcript_list = YouTubeTranscriptApi.list_transcripts(self.video_id) #拿到字幕列表
            return transcript_list
        except (VideoUnavailable, TranscriptsDisabled) as e:
            self.pprint("Failed to retrieve captions: " + str(e))
            return None

    def get_transcript_from_list(self, transcript_list):

        try:
            self.pprint("Trying to get manually uploaded captions...")
            transcript = transcript_list.find_transcript(['en', 'zh-Hant', 'zh-Hans'])
        except NoTranscriptFound:
            # 如果找不到, 就找自動生成的字幕
            self.pprint("Trying to get automatically generated captions...")
            transcript = transcript_list.find_generated_transcript(['en', 'zh-Hant','zh-Hans']) #找到自動生成的字幕
        
        if transcript:
            self.dialogue = ''
            for caption in transcript.fetch():
                text = caption['text']
                self.dialogue += text.strip() + ', ' #把字幕加到對話裡面
            
        else:
            self.pprint("Failed to retrieve captions, using Speech-to-Text API instead.")
            self.audio_file = self.download_youtube_audio(self.video_id)
            self.dialogue = self.transcribe_audio_file(self.audio_file)

    def save_dialogue(self):
        self.pprint("Saving captions...")
        try:
            self.data = {
                'title': self.title,
                'url': self.url,
                'zh_summary': '',
                'en_summary': '',
                'cn_summary': '',
                'zh_dialogue': '',
                'dialogue': self.dialogue[:-2]
            }

            script_dir = os.path.dirname(os.path.abspath(__file__))
            directory = os.path.join(script_dir, "dl_captions")

            if not os.path.exists(directory):
                os.makedirs(directory)

            self.filename = re.sub(r'[^a-zA-Z0-9一-龥]+', '_', self.title)

            with open(os.path.join(directory, f'{self.filename}.json'), 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
                print('Captions saved as captions.json.')
        except Exception as e:
            print("Error saving captions." + str(e))



    def get_youtube_video_captions(self): #這個ID基本上不用給
        self.title = self.get_youtube_title()
        
        transcript_list = self.get_transcript_list()
        self.get_transcript_from_list(transcript_list) #把字幕加到對話裡面 self.dialogue


        if self.dialogue:
            try:
                self.save_dialogue()
                _ctx = self.dialogue[:-2]
                #self.pprint(_ctx[:100] + "...")
                markdown = '"""'+_ctx+'"""'
                self.pprint(markdown[:100])
                self.mds = MDSplit('"""'+markdown+'"""',split_length=10000, overlap=1000)    
                self.docs = self.mds.get_docs() # dialogues splitted into docs
                return self.data, self.filename, self.docs

            except Exception as e:
                print("get_youtube_video_captions (get docs)... : " + str(e))
                return None, None, None
        else:
            print("Error retrieving captions.")
            return None, None, None
    
    def get_summary(self, docs):
        try:
            self.pprint("Summarizing captions...")
            self.ds = DocsSummarizer(docs) #其實呢個doc係唔關事只係一開始要init...以後有時間再改
            #self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)

            en_summary, zh_summary, cn_summary = self.ds.get_zh_summary(docs)
            self.summary = zh_summary 
            return en_summary, zh_summary, cn_summary
        except Exception as e:
            print("Error summarizing captions." + str(e))
            return None, None, None
    
    def save_to_db(self):
        data_obj = YouTubeData(data=None)
        data_obj.insert_data()
        
    def run(self):

        #所有野行一次先!!!
        #data 係json, filename 係Title && 檔名, splitted_caption_docs 係分割好的對話
        video_captions, filename, splitted_caption_docs = self.get_youtube_video_captions()
        #self.pprint(splitted_caption_docs[0])

        if filename and video_captions:
            directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dl_captions")

            #openai = LLM_OpenAI()
            
            try:     
                self.pprint("Summarizing captions...")
                self.title = video_captions['title']
                print("run title: ", self.title)
                en_summary, zh_summary, cn_summary = self.get_summary(video_captions['dialogue'])
                #print("run zh_summary: ", zh_summary)
          
                #print("run self.summary: ", self.summary)
            except Exception as e:
                print("Error summarizing captions." + str(e))
                return None, None, None, None, None, None

            try:
                ltp = LongTextParaphaser()
                self.zh_paraphrase = ltp.get_zh_paraphrase(video_captions['dialogue'])
                #print(zh_paraphrase)
            except Exception as e:
                print("Error paraphrasing captions." + str(e))
                return None, None, None, None, None, None


            if self.summary:
                # 将摘要添加到 JSON 数据中
                self.pprint("Saving summary...")
                video_captions['summary'] = self.summary
                

                # 将摘要添加到标题后面
                ordered_video_captions = OrderedDict()
                ordered_video_captions['title'] = video_captions['title']
                ordered_video_captions['url'] = self.url
                ordered_video_captions['zh_summary'] = zh_summary
                ordered_video_captions['en_summary'] = en_summary
                ordered_video_captions['cn_summary'] = cn_summary
                ordered_video_captions["zh_paraphrase"] = self.zh_paraphrase
                ordered_video_captions['dialogue'] = video_captions['dialogue']

                print("preparing jsondata...")
                jsondata ={
                    "title": video_captions['title'],
                    "url": self.url,
                    "zh_summary": zh_summary,
                    "en_summary": en_summary,
                    "cn_summary": cn_summary,
                    "zh_dialogue": self.zh_paraphrase,
                    "dialogue": video_captions['dialogue']
                }
                print("jsondata: ", jsondata)

                print("inserting data to DB...")
                
                try:
                    ytd = YouTubeData(data=jsondata)
                    ytd.insert_data()
                except Exception as e:
                    print("Error inserting data to DB45862." + str(e))
                    return None, None, None, None, None, None
                
                try:
          
                    # 将修改后的数据保存到原有的 JSON 文件中
                    with open(os.path.join(directory, f'{filename}.json'), 'w', encoding='utf-8') as f:
                        json.dump(ordered_video_captions, f, ensure_ascii=False, indent=2)

                    #self.pprint(f"Title: {self.title}")
                    #self.pprint(f"GPT: {self.summary}")
                    #print(f"GPT Tech Sum: {summary_tech}")
                except Exception as e:
                    print("Error saving summary. 58963" + str(e))
                
                return self.title, ordered_video_captions['dialogue'], en_summary, zh_summary, cn_summary, self.zh_paraphrase

            else:
                print("Failed to summarize video. Error: 最後個度")
                return None, None, None, None, None, None





        

#yt = YoutubeSummarizer('https://www.youtube.com/watch?v=vGP4pQdCocw')
#yt.run2()
