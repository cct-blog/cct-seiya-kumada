import os

from dotenv import load_dotenv

# .envファイルから環境変数を読み込み
load_dotenv(override=True)

# Azure OpenAI設定
azure_openai_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
azure_openai_api_key = os.environ["AZURE_OPENAI_API_KEY"]
azure_openai_api_version = os.environ["AZURE_OPENAI_API_VERSION"]
azure_openai_model_name = os.environ["AZURE_OPENAI_MODEL_NAME"]
azure_openai_detail = os.environ["AZURE_OPENAI_DETAIL"]
