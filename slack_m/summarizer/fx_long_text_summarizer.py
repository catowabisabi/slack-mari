import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from slack_m.my_llms.my_llms import LLM_OpenAI #語言模型
from slack_m.summarizer.fx_llms import DocsSummarizer
from fake_data import text

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
llm = LLM_OpenAI(max_tokens=1200).llm

class LongTextParaphaser:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=500)
        self.chain = load_qa_chain(llm=llm, chain_type="stuff", verbose=False)
        
    def get_zh_paraphrase(self, text):
        docs = self.text_splitter.create_documents([text]) #docs
        print("get_zh_paraphrase recived docs: ")

        try:
            print("在get_zh_paraphrase, 我們有{}個文檔\n".format(len(docs)))
            i = 0
            zh_paraphrase = ""
            for doc in docs:
                i+=1
                print("第{}個文檔".format(i))

                list_doc = []
                list_doc.append(doc)
                print("list_doc: ", list_doc)
                query = "請把有意義的內容改寫成為繁體中文。"
                reply = self.chain.run(input_documents=list_doc, question=query)
                print(reply)
                zh_paraphrase += (reply + "\n")
            print(zh_paraphrase)
            print("get_zh_paraphrase finished 1256863")
            return zh_paraphrase

        except Exception as e:
            print("get_zh_paraphrase Error: " + str(e))
            return None



""" ltp = LongTextParaphaser()
text3 = text.text3
ltp.get_zh_paraphrase(text3) """
