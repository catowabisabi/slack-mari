
from langchain.prompts import PromptTemplate

prompt_zh_points_and_summary = PromptTemplate(    
       
    input_variables=["text"],
    template="請用繁體中文例出用戶提供的文字的重點, 以點的形式。 \
        例如:\n \
        重點1:\
        重點2:\
        重點3:\
        如此類推。\n \
        然後再用繁體中文總結用戶提供的文字的重點。並說明文章所述的各項事件對經濟, 巿場, 或加密幣貨幣的影響, 請詳細說明有正面還是負面的影響, 並列出有可能受影響的股票或加密産品\
        例如: \
        \n總結:\n\
        用戶提供的文字: {text}"
    )

prompt_youtube_summary = PromptTemplate(    
       
    input_variables=["text"],
    template="""
    這是一段Youtube影片的字幕, 請用繁體中文例出影片的重點, 以點的形式。
    例如:
    1) 重點1
    2) 重點2
    3) 重點3
    如此類推。
    然後再用繁體中文詳細說明這重點的內容。
    用戶提供的文字: {text}"
    最後給出總結。
    "總結: "總結內容...""
    """
    )

prompt_youtube_summarize_tech = PromptTemplate(    
       
    input_variables=["text"],
    template="""
    這是一段Youtube影片的字幕, 請用繁體中文例出影片的技術重點, 以點的形式。
    例如:
    1) 重點1
    2) 重點2
    3) 重點3
    如此類推。
    然後再用繁體中文詳細說明這重點的內容。
    用戶提供的文字: {text}"
    
    最後, 如果有技術內容, 就詳細說明。然後把相關的內容的文字引用出來(不要給我整段文字, 只要相關的重點的句子就可以了)
    "技術1: "技術內容1..."
    "引用1: "引用內容1..."
    "技術2: "技術內容2..."
    "引用2: "引用內容2..."
    "技術3: "技術內容3..."
    "引用3: "引用內容3..."
    如此類推。
    """
    )

prompt_youtube_summarize_tech2 = PromptTemplate(    
       
    input_variables=["text"],
    template="""
    這是一段Youtube影片的字幕, 請用繁體中文例出影片的技術重點, 以點的形式。
    例如:
    "重點1":"重點1..."
    "重點2":"重點2..."
    "重點3":"重點3..."
    如此類推。

    如果有技術內容, 詳細說明。
    例如:
    "技術1: "技術說明1..."
    "技術2: "技術說明2..."
    "技術3: "技術說明3..."
    如此類推。

    然後再用繁體中文詳細說明這重點的內容。
    例如:
    "總結: "總結內容...""

    用戶提供的文字: {text}"
    """
    )

prompt_text_to_zh = PromptTemplate(    
       
    input_variables=["text"],
    template="""
    這是一段Youtube影片的字幕, 幫我改為繁體中文。要求: 我需要所有的內容, 但我希望是以說明文形式, 而不是口語形式。幫我理解對話內容然後以繁體中文說明所有細節內容
    用戶提供的文字: {text}"
    """
    )