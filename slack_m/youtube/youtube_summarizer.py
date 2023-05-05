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



def pprint(text):
    print("\n========\n"+str(text)+"\n\n========\n\n\n")

class YoutubeSummarizer:

    def pprint(self, text):
        print()
        print("========")
        print(text)
        print("========")
        print()

    def __init__(self, url, tech_sum = True, to_zh=False):
        self.url = url
        self.video_id = self.get_youtube_id(self.url) 
        self.audio = None
        self.transcript = None
        self.audio_file = None
        self.dialogue = None
        self.summary = None
        self.to_zh = to_zh
        self.tech_sum = tech_sum
        self.zh_text = None
        
                

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
            print(match.group(4))
            return match.group(4)
        else:
            # 如果鏈接無效，返回空字符串
            return ''

    # 添加下載音頻的函數
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
        
        

    # 添加將音頻轉換為文字的函數
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



    # 從 YouTube 影片 ID 取得字幕

    def get_youtube_video_captions(self, _video_id):
        self.pprint("Getting captions...")
        if not _video_id:
            _video_id = self.video_id
        url = f'https://www.youtube.com/watch?v={_video_id}'

        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.title.string.replace(" - YouTube", "")

        self.dialogue = ''
        try:
            self.pprint("Retrieving captions...")
            transcript_list = YouTubeTranscriptApi.list_transcripts(_video_id)
        except (VideoUnavailable, TranscriptsDisabled) as e:
            self.pprint("Failed to retrieve captions: " + str(e))
            return None
            
        
        try:
            # 首先尝试获取手动上传的字幕
            try:
                self.pprint("Trying to get manually uploaded captions...")
                transcript = transcript_list.find_transcript(['en', 'zh-Hans'])
            except NoTranscriptFound:
                # 如果找不到手动上传的字幕，则尝试获取自动生成的字幕
                self.pprint("Trying to get automatically generated captions...")
                transcript = transcript_list.find_generated_transcript(['en', 'zh-Hans'])

            for caption in transcript.fetch():
                text = caption['text']
                self.dialogue += text.strip() + ', '

        except Exception as e:
            self.pprint("Failed to retrieve captions, using Speech-to-Text API instead.")
            self.audio_file = self.download_youtube_audio(_video_id)
            self.dialogue = self.transcribe_audio_file(self.audio_file)

        if self.dialogue:
            self.pprint("Saving captions...")
            try:
                data = {
                    'title': title,
                    'summary': '',
                    'dialogue': self.dialogue[:-2]
                }

                script_dir = os.path.dirname(os.path.abspath(__file__))
                directory = os.path.join(script_dir, "dl_captions")

                if not os.path.exists(directory):
                    os.makedirs(directory)

                filename = re.sub(r'[^a-zA-Z0-9一-龥]+', '_', title)

                with open(os.path.join(directory, f'{filename}.json'), 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                    print('Captions saved as captions.json.')

                _ctx = data['dialogue']
                self.pprint("Splitting captions...")
                self.pprint(_ctx)
                markdown = '"""'+_ctx+'"""'
                self.pprint(markdown)
                self.mds = MDSplit('"""'+markdown+'"""')    
                self.docs = self.mds.get_docs()

                return data, filename, self.docs
                
            except Exception as e:
                print("Error saving captions." + str(e))
                return None, None, None
        else:
            print("Error retrieving captions.")
            return None, None, None


    def run(self):
        
        video_captions, filename, splitted_caption_docs = self.get_youtube_video_captions(self.video_id)

        #self.pprint(splitted_caption_docs[0])

        if filename and video_captions:
            directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dl_captions")

            openai = LLM_OpenAI()
            
            try:     
                self.pprint("Summarizing captions...")
                self.title = video_captions['title']
                
                """ if self.tech_sum == True:
                    self.summary = openai.summarize_youtube_video_tech(video_captions)
                else:
                    self.summary = openai.summarize_youtube_video(video_captions)
                
                if self.to_zh == True:
                    self.zh_text = openai.text_to_zh(video_captions) """
                ds = DocsSummarizer(video_captions['dialogue']) 
                en_summary, zh_summary, cn_summary = ds.get_zh_summary(video_captions['dialogue'])


                self.summary = zh_summary 
                

                if self.summary:
                    # 将摘要添加到 JSON 数据中
                    self.pprint("Saving summary...")
                    video_captions['summary'] = self.summary
                    

                    # 将摘要添加到标题后面
                    ordered_video_captions = OrderedDict()
                    ordered_video_captions['title'] = video_captions['title']
                    ordered_video_captions['zh_summary'] = zh_summary
                    ordered_video_captions['en_summary'] = en_summary
                    ordered_video_captions['cn_summary'] = cn_summary
                    ordered_video_captions['dialogue'] = video_captions['dialogue']
                    ordered_video_captions['zh text'] = self.zh_text

                    # 将修改后的数据保存到原有的 JSON 文件中
                    with open(os.path.join(directory, f'{filename}.json'), 'w', encoding='utf-8') as f:
                        json.dump(ordered_video_captions, f, ensure_ascii=False, indent=2)

                    #self.pprint(f"Title: {self.title}")
                    #self.pprint(f"GPT: {self.summary}")
                    #print(f"GPT Tech Sum: {summary_tech}")
                    
                    return self.title, ordered_video_captions['dialogue'], en_summary, zh_summary, cn_summary

            except Exception as e:
                print("Failed to summarize video. Error: ", e)
                return None, None, None, None, None


""" my_yt_summarizer = YoutubeSummarizer("https://www.youtube.com/watch?v=SxAn6f7gM44", tech_sum=True, to_zh=False)
title, dialogue, en_summary, zh_summary, cn_summary = my_yt_summarizer.run()

pprint(title)
pprint(dialogue)
pprint(en_summary)
pprint(zh_summary)
pprint(cn_summary) """