from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import os
from slack_m.my_llms.prompts.prompt_templates import prompt_zh_points_and_summary, prompt_youtube_summary, prompt_youtube_summarize_tech2, prompt_text_to_zh


from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

class LLM_OpenAI:
    def __init__(self, OPENAI_API_KEY=OPENAI_API_KEY, temperature=0.5, max_tokens=-1):
        self.llm = OpenAI(openai_api_key = OPENAI_API_KEY, temperature=temperature, max_tokens=max_tokens)
    
    def get_llm_anwser(self, question):
        text = question
        reply = self.llm(text)
        print(reply)

    def get_zh_answer(self, input_question):
        prompt = PromptTemplate(
            input_variables=["question"],
            template="請用繁體中文回答以下問題: {qestion}?"
            )
        print(prompt.format(question=input_question))
        print(self.llm(prompt.format(question=input_question)))

    def summarize_text(self, text, prompt_template=prompt_zh_points_and_summary):
        prompt = prompt_template
        #print(prompt.format(text=text))
        print(self.llm(prompt.format(text=text)))
    
    def summarize_text_with_prompt(self, text, prompt):
        print(self.llm(prompt.format(text=text)))
    
    def summarize_youtube_video(self, video_captions, prompt_template=prompt_youtube_summary):
        text = video_captions["dialogue"]
        prompt = prompt_template
        #print(prompt.format(text=text))
        response = self.llm(prompt.format(text=text))
        return response
    
    def summarize_youtube_video_tech(self, video_captions, prompt_template=prompt_youtube_summarize_tech2):
        text = video_captions["dialogue"]
        prompt = prompt_template
        #print(prompt.format(text=text))
        response = self.llm(prompt.format(text=text))
        return response
    
    def text_to_zh(self, video_captions, prompt_template=prompt_text_to_zh):
        text = video_captions["dialogue"]
        prompt = prompt_template
        #print(prompt.format(text=text))
        response = self.llm(prompt.format(text=text))
        return response

    def get_en_zh_summary(self, text):
        prompt = prompt_zh_points_and_summary
        response = self.llm(prompt.format(text=text))
        return response
