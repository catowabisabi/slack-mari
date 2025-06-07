# Slack Mari Bot 🤖 / Slack Mari 機器人 🤖

## 📖 Introduction / 簡介

Slack Mari Bot is an intelligent assistant that combines the power of GPT, YouTube summarization, and natural conversation capabilities. It serves as both a professional secretary and a friendly companion in your Slack workspace, capable of communicating in Traditional Chinese, English, Cantonese, and Japanese.

Slack Mari Bot 是一個智能助手，結合了 GPT、YouTube 影片摘要和自然對話功能。它在您的 Slack 工作空間中既可以作為專業秘書，也可以作為友好的伴侶，能夠使用繁體中文、英文、粵語和日語進行溝通。

## 🎯 Purpose / 用途

The bot is designed to enhance team productivity and communication by:
- Providing intelligent responses to queries
- Summarizing YouTube videos in multiple languages
- Offering professional assistance as a secretary
- Creating a more engaging and interactive workspace environment

機器人旨在通過以下方式提升團隊生產力和溝通效率：
- 為查詢提供智能回應
- 以多種語言總結 YouTube 影片內容
- 提供專業的秘書協助
- 創造更具互動性的工作空間環境

## ✨ Features / 主要功能

1. **Multilingual Communication / 多語言溝通**
   - Traditional Chinese / 繁體中文
   - English / 英文
   - Cantonese / 粵語
   - Japanese / 日語

2. **YouTube Video Processing / YouTube 影片處理**
   - Video summarization / 影片摘要
   - Caption extraction / 字幕提取
   - Multi-language summaries / 多語言總結

3. **Intelligent Conversation / 智能對話**
   - Context-aware responses / 上下文感知回應
   - Professional secretary mode / 專業秘書模式
   - Casual conversation mode / 輕鬆對話模式

## ⚙️ Installation / 安裝

1. **Prerequisites / 前置要求**
   ```bash
   # Install required packages / 安裝所需套件
   pip install slack-sdk slack-bolt Flask langchain openai youtube-dl pytube google-cloud-speech-v1p1beta1 python-dotenv
   ```

2. **Environment Setup / 環境設置**
   ```bash
   # Create .env file / 創建 .env 文件
   touch .env

   # Add the following environment variables / 添加以下環境變量
   SLACK_BOT_TOKEN2=your-slack-bot-token
   SLACK_SIGNING_SECRET2=your-slack-signing-secret
   SLACK_BOT_USER_ID2=your-slack-bot-user-id
   OPENAI_API_KEY=your-openai-api-key
   ```

## 🚀 Usage / 使用方法

1. **Start the Bot / 啟動機器人**
   ```bash
   python slack_m/app.py
   ```

2. **Interact with the Bot / 與機器人互動**
   - Mention the bot with `@Mari` in any channel
   - Send YouTube URLs for summarization
   - Ask questions or request assistance
   - Empty mention for casual conversation

   ```
   @Mari How are you today?
   @Mari https://youtube.com/watch?v=example
   @Mari Can you help me draft an email?
   ```

## 📂 Module Breakdown / 模組說明

1. **Core Components / 核心組件**
   - `app.py`: Main application logic / 主要應用邏輯
   - `functions.py`: Core functionality implementations / 核心功能實現
   - `get_token.py`: Slack authentication handling / Slack 認證處理

2. **YouTube Processing / YouTube 處理**
   - `youtube_summarizer.py`: Video processing and summarization / 影片處理和摘要
   - `youtube/`: YouTube-related utilities / YouTube 相關工具

3. **Language Processing / 語言處理**
   - `my_llms/`: Language model integration / 語言模型整合
   - `summarizer/`: Text summarization utilities / 文本摘要工具
   - `split/`: Text splitting and processing / 文本分割和處理

4. **Utility Modules / 工具模組**
   - `greetings.py`: Response templates / 回應模板
   - `database/`: Data storage utilities / 數據存儲工具

## 🔒 Security Note / 安全注意事項

Always keep your API keys and tokens secure. Never commit them directly to your repository.
請務必確保您的 API 密鑰和令牌安全。切勿直接將它們提交到代碼庫中。

---

For more detailed information about each module and feature, please refer to the individual module documentation.
有關每個模組和功能的更詳細信息，請參閱各個模組的文檔。
