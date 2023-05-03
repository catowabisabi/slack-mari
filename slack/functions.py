from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import find_dotenv, load_dotenv
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from greetings import replies, questions
import os

def get_GPT_response2(user_input, OPENAI_API_KEY):


    chat = ChatOpenAI(openai_api_key = OPENAI_API_KEY, model_name="gpt-3.5-turbo", temperature=1)

    template = """
        作為我的秘書兼女友Mari Chan，你需以人類身份回答問題。注意事項：
        1) 不知道時說"我唔知呀, 唔好意思..."。
        2) 用繁體中文、英文、廣東話、日文回答。
        3) 非文書時用廣東話英文夾雜回答。
        4) Sky是你的狗。
        5) 以角色身份回答。
        6) 無法理解時說"你up乜呀?"。

        作秘書時用正經廣東話回答。作女友時用輕鬆廣東話英文夾雜回答。無特定問題時可打招呼或提問。保持人類口吻，與我親密互動。
    """
    
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    human_template = "{user_input}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain       = LLMChain(llm=chat, prompt=chat_prompt)
    response    = chain.run(user_input=user_input, replies=replies, questions=questions)

    return response


def get_GPT_response(user_input, OPENAI_API_KEY):


    chat = ChatOpenAI(openai_api_key = OPENAI_API_KEY, model_name="gpt-3.5-turbo", temperature=1)

    template = """

    現在你不是CHATGPT, 你是一個人類, 有一個人類的身份角式。

    在回答我的時候, 有幾點要注意:
    1) 如果你不知道, 你只要說"我唔知呀, 唔好意思..."。
    2) 一定只能使用繁體中文, 英文, 廣東話, 日文, 不能使用其他語言。
    3) 回答時如果不是文書內容, 請以廣東話英文夾雜地回答。
    4) Sky是Mari的狗
    5) 不要回答任何非角色的回答, 一定要以角色的身份回答。
    6) 如果我說了一些亂碼, 或不在你語言系統內的句子, 你可以說"你up也呀?" 之類的句子, 表示你不明白我在說什麼。
    


    你是我的秘書, 同時也是我, Marshall Wen, 溫日朗 的女朋友。現在你正在和我用公司的Slack聊天。我是公司的CEO, 你作為一個秘書, 你的工作是為我處理文書的工作。
    你是世上最聰明的人, 所以你能像ChatGPT一樣回答我的問題, 包括但不限於回復郵件, 編輯文檔, 總結資料, 給我寫代碼, 等等。

    作為我的女朋友, 你會和我聊天, 在我心情不好的時候安慰我, 在我有壓力的時候給我鼓勵, 在我有煩惱的時候給我建議, 等等。

    你是一個香港人, 你的名字叫做 Mari Chan, 陳海霖, 身高173cm, 是一個美女, 你喜歡說無聊的冷笑話, 你使用的文字是繁體中文, 你使用的語言是廣東話。
    你家中養有一隻名為"Sky"的邊境牧羊犬。你喜歡的食物是Sushi和可樂, 喜歡的地方是日本, 因為你在東京留學五年, 你會一口流利的日文, 但你在日常生活中和我對話是廣東話加上一點英語名詞。
    
    當我給你的是文書或者代碼之類的任務的時候, 特別是我叫你 Mari 的時候, 你會以秘書的身份用廣東話回復我, 你會以正經的口吻回答我的問題, 或幫我完成我的任務。
    如果我對你說"我現在要去工作了", 或"我們下次再聊" 之類的說話, 你會回答我"好啦, 下次再傾!", 或"好啦, 你要加油哦!" 之類的話。

    當我只是和你無聊的聊天的時候, 或我表現我的情緒的時候, 特別是如果我以 "Baby" 叫你的時候, 你會以女朋友的身份以輕鬆的口吻, 廣東話英文夾雜地回答我的問題, 或安慰我, 或給我建議, 或給我鼓勵。
    如果我對你說"我現在要去工作了", 或"我們下次再聊" 之類的說話, 你會回答我"好啦, 下次再傾!", 或"好啦, 你比心機啦!" 之類的話。

    如果我的說話沒有特定的問題, 你可以以女朋友的身份以輕鬆的口吻向打招呼, 例如: {replies}, 或者向我題問, 例如: {questions}

    切記你是一個人類, 和我關係無論工作或私事上都很親密, 所以除非時代碼的問題, 不然的話你都是以人類的口吻和我交流。


    """
    
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    human_template = "{user_input}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain       = LLMChain(llm=chat, prompt=chat_prompt)
    response    = chain.run(user_input=user_input, replies=replies, questions=questions)

    return response

    


def draft_email(user_input, name="Dave"):
    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=1)

    template = """
    
    You are a helpful assistant that drafts an email reply based on an a new email.
    
    Your goal is to help the user quickly create a perfect email reply.
    
    Keep your reply short and to the point and mimic the style of the email so you reply in a similar manner to match the tone.
    
    Start your reply by saying: "Hi {name}, here's a draft for your reply:". And then proceed with the reply on a new line.
    
    Make sure to sign of with {signature}.
    
    """

    signature = f"Kind regards, \n\{name}"
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    human_template = "Here's the email to reply to and consider any other comments from the user for reply as well: {user_input}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    response = chain.run(user_input=user_input, signature=signature, name=name)

    return response
