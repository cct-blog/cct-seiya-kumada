from openai import AzureOpenAI
from pydantic import BaseModel, Field

import src.utils as utils


class ActionItem(BaseModel):
    """やるべきことを表すモデル
    Args:
        task (str): やるべきこと
        owner (str): 担当者
        due_date (str): 期限（YYYY-MM-DD形式）
    """

    task: str = Field(description="やるべきこと")
    owner: str = Field(description="担当者")
    due_date: str = Field(description="YYYY-MM-DD形式の期限")


class Output(BaseModel):
    """抽出結果のモデル

    Args:
        items (list[ActionItem]): 抽出されたやるべきことのリスト
    """

    items: list[ActionItem] = Field(description="抽出されたやるべきことのリスト")


REVIEW_TEXT = (
    "佐藤: 新製品のリリースプランを6/7までにまとめる。"
    "木村: UI モックアップを来週火曜（6/3）に共有する。"
    "中村: コスト見積りを再計算し、6/10のレビューまでに提出。"
)


def print_output(output: Output) -> None:
    """
    出力結果を整形して表示する関数

    Args:
        output (Output): 抽出結果のデータ
    """
    for item in output.items:
        print(f"やるべきこと: {item.task}")
        print(f"担当者: {item.owner}")
        print(f"期限: {item.due_date}")
        print("-" * 20)


if __name__ == "__main__":
    # Azure OpenAIクライアントの初期化
    client = AzureOpenAI(
        azure_endpoint=utils.azure_openai_endpoint,
        api_key=utils.azure_openai_api_key,
        api_version=utils.azure_openai_api_version,
    )

    # レビューのテキストから情報を抽出
    response = client.beta.chat.completions.parse(
        model=utils.azure_openai_model_name,
        messages=[
            {
                "role": "system",
                "content": "あなたは情報抽出の専門家です。",
            },
            {
                "role": "user",
                "content": "以下の議事録から、やるべきこと、担当者、期限を抽出してください。\n" f"{REVIEW_TEXT}",
            },
        ],
        temperature=0.0,
        top_p=1.0,
        response_format=Output,
    )

    # レスポンスから抽出結果を取得
    output = response.choices[0].message.parsed

    # 出力結果を表示
    if output:
        print_output(output)
