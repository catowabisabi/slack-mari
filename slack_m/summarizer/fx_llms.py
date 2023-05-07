import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from slack_m.split.markdown_split import MDSplit, markdown_text #分割文檔
from slack_m.my_llms.my_llms import LLM_OpenAI #語言模型

from textblob import TextBlob
from googletrans import Translator

from langchain import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import AnalyzeDocumentChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

# Loaders
from langchain.schema import Document

# Model
from langchain.chat_models import ChatOpenAI

# Embedding Support
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings


# Data Science
import numpy as np
from sklearn.cluster import KMeans

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Taking out the warnings
import warnings
from warnings import simplefilter


from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain





from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

llm = LLM_OpenAI(max_tokens=1200).llm


def pprint(text):
    print("\n========\n"+str(text)+"\n\n========\n\n\n")


class DocsSummarizer:

    def __init__(self, docs=None, split_length=1000, overlap=0):
        self.docs = docs
        self.mds = MDSplit(docs, split_length, overlap)
        self.splitted_docs = self.mds.get_docs()
        self.summary = None
        self.zh_summary = None
        self.cn_summary = None
        self.en_summary = None

    # 打印分割後的文檔
    def print_docs(self):
        print("Length of docs: " + str(len(self.docs)))
        i = 1
        list_of_docs = []
        for doc in self.splitted_docs:
            print("Doc " + str(i))
            pprint(doc.page_content)
            i+=1
         
    # 返回分割後的文檔清單
    def get_splitted_docs(self):
        return self.splitted_docs
    
    def summarize_docs(self):
        summary_chain = load_summarize_chain(llm, chain_type="map_reduce")
        summarize_document_chain = AnalyzeDocumentChain(combine_docs_chain=summary_chain)
        self.summary = summarize_document_chain.run(self.docs)
        return self.summary
    
    def get_zh_summary(self, text):
        if self.summary == None:
            try:
                self.get_long_summary(text)
            except Exception as e:
                pprint("Error in summarizing docs:  " + str(e))
                return None
        translator = Translator()
        self.zh_summary = translator.translate(str(self.summary), dest='zh-tw').text
        self.cn_summary = translator.translate(str(self.summary), dest='zh-cn').text
        self.en_summary = translator.translate(str(self.summary), dest='en').text
        return self.en_summary, self.zh_summary, self.cn_summary
    
    


        
    
    def get_tokens_num(self, text):
        return llm.get_num_tokens(text)

    def get_long_summary(self, text): #成個dialogue
        pprint("get_long_summary: Summarizing docs...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=500)
        #text_splitter = RecursiveCharacterTextSplitter(separators=["\\n\\n", "\\n"], chunk_size=10000, chunk_overlap=500)
        docs = text_splitter.create_documents([text])
        num_docs = len(docs)
        num_tokens_first_doc = llm.get_num_tokens(docs[0].page_content)#文字
        print (f"現在我們有{num_docs}份文件，第一份文件有{num_tokens_first_doc}個令牌")

        #summary_chain = load_summarize_chain(llm=llm, chain_type='map_reduce', verbose=False)
        #summary = summary_chain.run(docs)
        #print("Summary1: " + str(summary))
        
#========================================================================================================
        map_prompt = """
            Write a detail summary include all the technical key points if the content is related to technology of the following:
            "{text}"
            DETAIL SUMMARY:
            """
        combine_prompt = """
            Write a detail summary of the following text delimited by triple backquotes, 
            include all the technical key points if the content is related to technology.
            Then, write a bullet point summary of the detail summary.
            ```{text}```
            DETAIL SUMMARY:
            BULLET POINT SUMMARY:
            """
#========================================================================================================
        map_prompt2 = """
            Paraphrase the following text into an explanatory format:
            "{text}"
            EXPLANATORY FORMAT PARAPHRASE:
            """
        combine_prompt2 = """
            Paraphrase the following text delimited by triple backquotes into an explanatory format. 
            include all the technical key points if the content is related to technology.
            Then, write a bullet point summary of the detail summary.
            ```{text}```
            EXPLANATORY FORMAT PARAPHRASE:
            BULLET POINT SUMMARY:
            """
#========================================================================================================      
#改為中文Summarize
        map_prompt_zh= """
            對以下內容進行繁體中文詳細摘要，如果內容與技術相關，請包含所有技術要點。格式如下：
            "繁體中文詳細摘要:
            [詳細摘要內容...]"

            用戶輸入內容:
            ```{text}```
            """
        combine_prompt_zh = """
            對以下在三重反引號中的內容進行繁體中文詳細摘要。如果內容與技術相關，請包含所有技術要點。
            然後，對內容進行BULLET POINT摘要。格式如下：

            "繁體中文詳細摘要:
            [詳細摘要內容...]
            
            BULLET POINT摘要:
            [BULLET POINT摘要內容...]"
            
            用戶輸入內容:
            ```{text}```
            """
#======================================================================================================== 
        map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])
        combine_prompt_template = PromptTemplate(template=combine_prompt_zh, input_variables=["text"])
        try:
            print("summary_chain in get_long_summary/Summarizing docs...")
            summary_chain = load_summarize_chain(llm=llm,
                                        chain_type='map_reduce',
                                        map_prompt=map_prompt_template,
                                        combine_prompt=combine_prompt_template,
                                        verbose=False
                                        )
        except Exception as e:
            print ( str(e))
            return None
        
        self.summary = summary_chain.run(docs)
        return self.summary #繁體中文摘要
    

    def get_zh_paraphase(self, docs):
        print(str(docs))
        #embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        # create the vectorestore to use as the index
        try:
            result = ""
            for doc in docs:
                list_doc = []
                list_doc.append(doc)
                chain = load_qa_chain(llm=llm, chain_type="stuff", verbose=False)
                query = "請把有義意的內容改寫成為繁體中文的講稿格式。"
       
                reply = chain.run(input_documents=list_doc, question=query)
                result += reply
            return result
        
        except Exception as e:
            print("Error: " + str(e))
    
    def print_docs(self, docs):
        for doc in docs:
            print (doc)
            print ("=========================================")
        


    def get_vector_summary(self):

        # Combine the pages, and replace the tabs with spaces
        text = ""
        for page in self.splitted_docs:
            text += page.page_content
        text = text.replace('\t', ' ')

        num_tokens = llm.get_num_tokens(text)
        print (f"This book has {num_tokens} tokens in it")

        text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "\t"], chunk_size=10000, chunk_overlap=3000)
        docs = text_splitter.create_documents([text])
        num_documents = len(docs)
        print (f"Now our book is split up into {num_documents} documents")

        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vectors = embeddings.embed_documents([x.page_content for x in docs])

        #Assuming 'embeddings' is a list or array of 1536-dimensional embeddings

        # Choose the number of clusters, this can be adjusted based on the book's content.
        # I played around and found ~10 was the best.
        # Usually if you have 10 passages from a book you can tell what it's about
        num_clusters = len(docs)

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(vectors)
        print(kmeans.labels_)

        """ import numpy as np

        vectors = np.array(vectors)

        # Filter out FutureWarnings
        simplefilter(action='ignore', category=FutureWarning)

        # Perform t-SNE and reduce to 2 dimensions
        tsne = TSNE(n_components=2, random_state=42)
        reduced_data_tsne = tsne.fit_transform(vectors)

        # Plot the reduced data
        plt.scatter(reduced_data_tsne[:, 0], reduced_data_tsne[:, 1], c=kmeans.labels_)
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.title('Book Embeddings Clustered')
        plt.show() """



markdown_text = """hey friends welcome to one little coder, today we're going to see how can you fit, or how can you use a very heavy model, like bloom, 1756 i think 176 bloom 176 billion, parameter or opt 175 billion parameter, how can you run these big models like, large language models on a single, mission or single gpu, and for that we are going to use a new, library called bnb bytes so bnb bytes is, a live sorry bits and bytes is the, library that we're going to use and the, entire content that you're going to see, on this particular video is not created, by me i just simply took it from this, post um this document that tim did mess, shade so thanks to tim ditmers for, sharing this document, so now what's happening here so if you, see this um you can actually see that um, so you have got models and you have got, computers like machines and you've got, gpu memory and you can see how uh the, largest model like the, the actual original model could be run, on um using using the 8 bit and 16 bit, precision so and you can see the details, so what is happening um in, the background is if you are familiar, with the concept of tiny ml you know, that one of the ways people, reduce the size of a model is something, called quantization so quantization is a, process of um, i'm over simplifying the definition but, quantization is a process where you take, your floating points which is which is, quite common in your neural networks, and then bring it to a lower order like, like for example you're going to make it, int 8 so that's what it says int 8 so, you're you're changing the bits and, bytes and then you're converting it into, ink 8 and, what what is happening today is that you, can do this for a hugging face models, with just one line of code change you, don't have to do anything big you don't, have to change anything a lot all you, have to do is just include one line of, code install a couple of libraries and, in one line of code you will be able to, use opt-175b, or bloom 176b this is super huge these, are like super heavy huge models like i, was i never knew that i could use this, on google cola but today big thanks to, thanks to the book tim ditmers and the, team has done you can use it if you want, more details about it there is a, document that talks about the details, about the mission the comparison and, what's happening here and you know like, what kind of computers and how do you, have to do it i'll not get into the, detail here, but what we are going to see now is two, cases one i'm going to create i have, i've got a google collab which is just, um tim's google collab i've got this, google collab and as you can see i have, enabled gpu but what i've done is um you, can see that i just simply installed the, transformers, and then you can see the mission i got, from google it's a tesla t4 machine, despite having tesla t4 machine when i, tried to load a three billion model, and at the end of when the model, downloaded you can see that my session, has crashed, because i have used all available ram, and google says if you are interested in, getting more ram runtime please buy, google collab pro which i don't have, money for so i don't want to buy, so what i'm trying to show here is, even for a 3 billion model 3 billion, parameter model, the problem is your google collab cannot, handle your three billion parameter, model because, you know it's a huge model it's so heavy, that your 16 gb i think 16 gig memory, cannot handle it from the google collab, what does it mean it means, until unless you do something like, quantization, you cannot use a three billion, parameter model on google collab, and now comes the best part thanks to, the google collab notebook that uh tim, tim and tim's team has put together so, what we are seeing here is hugging face, meets bits and bytes for lighter models, on gpu for inference, and you can see there is a logo for it, and all you have to do is um if you want, to run your own 8-bit model on any, hugging face model like you take any, angling face model and you want to like, basically do the quantization or a run, the 8-bit model, first install bits and bytes uh it's, available on pi pi second install the, latest hugging face transformers library, and third install the accelerate library, from hugging page so you need three, libraries, i mean usually you would install only, the transformers but now you need three, libraries, after you install three libraries make, sure that you have got gpu enabled and, when you get gpu just make sure that you, have got the right mission, in this case if you are going to use it, on google collab you need d4 gpu at, least so, if you don't get p4 cpu then um, then, restart it like re disconnected trace um, and reconnect it there is a chance that, you might get g4 but if you are not, using google collab like let's say, you're using paper space and let's say, you are using some other service, provider then you can check for these, missions, now, the pitch here is that you can use such, a large language model or like large, model from hugging phase just on a, single, gpu, so what do you have to do for it you, don't have to do anything special for, example you're going to use a very, simple code this almost in the same code, name of the model, the prompt text like these these are, like model parameters like name of the, model and this is your input takes, prompt x and maximum how much tokens you, want, and then you have got the tokenizer um, that would tokenize it, and then you have got a certain details, um like the function generate from model, but you know even like if you do not, want to use this if you just simply want, to use hugging base pipeline you can, still do it the way you can do it is, from transformers import pipeline and, pipeline what is the model name the, model name is you want to use big, science bloom 3 billion parameter model, which we just saw that it crashed our, google collab session, i mean we just literally saw that it, crashed our google collab session so, we're going to do the same thing here, and after we do that we're going to send, some model quirks which is um the device, map which is auto and we're going to say, load in 8 bit is equal to true so we're, telling the model, um pipeline to load the model in eight, bit that's it this is this is the only, thing that you're going to do add extra, other than installing the libraries the, required libraries and what happens next, is pure magic, because not just your model is, successfully downloaded without crashing, your ram you can see that the ram went, up and down you can see that it went up, and down like it, it it it is not that it is not using ram, but you can generate text like you can, take the text and generate it like for, example i can, i can say i love one little coder, youtube channel, and, i hope it doesn't show anything bad, and, i mean i just want to say that it's the, same computer it's the same machine this, tesla t4, and uh tesla d4 here like if you're, wondering how am i running two gpu, session um it's two different gmail like, this i mean there is no other black, magic to it so you can see it's the same, computer but because we used um bits and, bytes, we are able to successfully load the, three billion parameter model on our, google collab environment and it says i, love one little youtube channel i've, been watching it for a while i love the, way he explains things and the way, i i mean should i be happy about the, compliment that i got from bloom um, i think i should be happy like maybe the, artificial intelligence, future, future leader is going to appreciate me, maybe i can use this as a proof but you, know, jokes aside that we have successfully, managed to load the six billion sorry, three billion parameter model, and it just takes only one line of, transformers pipeline, and it's just one extra argument, which is model quarks with two, values, device map and load in 8-bit that's it, that's all you had to do to load a large, language model which was not previously, possible to load on a google collab in, notebook on a tesla d4 machine but now, you can do it because thanks to the, 8-bit quantized model using, bits and bytes on the go just directly, from working face i mean you didn't have, to do anything like for example let's, say you don't want to do that, you don't want to use pipeline you like, the flexibility that you have got then, all you have to do is, specify the, the model um that you want to download, for example auto model for causal alarm, and then auto tokenizer i want to run it, probably it might take a little bit of, time um because it has to download, but then you can see that here you are, passing these two arguments which is, device map is equal to auto load in, eight bit is equal to true, and then you have the same tokenizer the, next thing is you're going to refer back, to this section of the function that you, created got the tokenizer model.generate, and then you are going to decode very, simple stuff that you do all the time, with hugging phase, and then when you say generate from, model it's going to finally generate the, result right now you know you can see, that we have already loaded the model so, i'm not sure like you can see the system, run peeking i'm not sure if it will work, the reason why i'm saying may not work, is because i've already you know managed, to download a very heavy model so, maybe the ram is occupied even if it, fails the point is if you just do this, it's going to work so the point that, your you see is, it's not just, you are able to load the quantized model, i mean that's something that's quite, obvious right everybody knows that you, can load quantized model, um when you quantize model size reduces, so you you would be able to the point is, it loads without degrading uh the, performance oh it it finished, let me try generate from model, and it's going to, maybe generate yeah it says hello my, name is john maybe i should give the, same text, i should i should give this text here, okay where is the function so the, function takes, text text as input okay let me, give this as text run this once i mean, this is going to set the parameters, and then go back, let me close this and i want to say, generate from model, and it's going to generate i love only, decoder youtube channel and i've been, forging it for a while now and you can, see that, there is, like, so this is this is as same as this but, instead of using pipeline we are using, the model but when you are trying to do, it just um, without, uh without the quantized model still it, would do like for example instead of, saying load in 8-bit you can actually, say torch, and dtape is auto and then you can load, it i think that will most likely crash, your google collapsation but you know i, don't want to do that but the point here, is um, you, are, you are using a quantized model which is, lesser in size like smaller in memory, footprint but also it doesn't degrade, the performance i think this is, brilliant, um so you can see the difference in, intake model and you can see the, floating point 16 model so basically, what quantization is doing is at every, stage of the neural network it's taking, all the floating point of values that, does fp8 16 and then quantizing it to in, eight and again i'm not i'm get i'm over, simplifying the process but you can see, what's happening so you can see that, this i mean what they're saying is that, we have saved 1.65 x almost closer to, two double of the memory for a three, billion parameters model note that, internally we replace all the linear, layers by the ones implemented in bits, and bytes by scaling up the model, the number of linear layers will, increase therefore impact of saving, memory on those layers will be huge for, very large models for example quantizing, bloom 176 billion parameter model gives, a gain of, 1.9 x memory footprint which can save a, lot of computer power in practice, imagine, you have to self-host this model i mean, i cannot tell you how much this can save, um memory cost i mean carbon footprint, like come from any any angle i think, this is amazing um and i think this is, really good for the fact that people can, now load these models on a single gpu, if you've got a gpu at home now you'll, be able to host a model and then serve, it and i think this is quite amazing, this google collab notebook i'll link it, in the youtube description please give a, shout out to tim um i'll i'll do the, same thing when i publish this video but, please give a shout out to tim and tim's, team for making this amazingly simple i, mean what i like about this entire, process i don't have to learn a lot to, do quantization learn how to do, quantization go layer by layer i don't, have to do any of those things all i, have to do is install a couple of, libraries like with bits and bytes and, accelerate and just make sure that i've, got the right machine and that's it i, add just a parameter and then my job is, done and then the, library takes care of everything and, then i'm able to use a very large, language model on a free service that, google has given me i think this is, absolutely brilliant and i would really, like to thank tim and the team, for making this um amazing amazing, advancement i think they are still it to, publish the details on the research um, so we can we can wait for that but for, now, make sure that you enjoy this amazing, opportunity and also thank the team, thank you so much for watching this, video see in the next one, peace"""

#ds = DocsSummarizer(markdown_text)
#ds.get_long_summary(markdown_text)
