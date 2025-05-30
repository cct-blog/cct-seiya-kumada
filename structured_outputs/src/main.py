from openai import AzureOpenAI
from pydantic import BaseModel

import src.utils as utils


class Output(BaseModel):
    """
    出力形式の定義

    Attributes:
        persons_name (str): 抽出された人物名
        datetime (str): 抽出された日時
        organization_names (list[str]): 抽出された組織名のリスト
    """

    persons_name: str
    datetime: str
    organization_names: list[str]


REVIEW_TEXT = """
小泉氏は2015年10月から自民党の農林部会長を務め、JAグループの改革に取り組んだ。
政府の規制改革会議は、競争力を強化するためにJA全農の株式会社化などの組織改革案を打ち出し、
小泉氏も支持していた。
"""


def print_output(output: Output) -> None:
    """
    出力結果を整形して表示する関数

    Args:
        output (Output): 抽出結果のデータ
    """
    print(f"人物名: {output.persons_name}")
    print(f"日時: {output.datetime}")
    print(f"組織名: {', '.join(output.organization_names)}")


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
                "content": "以下のテキストから、人物名、日時、組織名を抽出してください。\n" f"{REVIEW_TEXT}",
            },
        ],
        response_format=Output,
    )

    # レスポンスから抽出結果を取得
    output = response.choices[0].message.parsed

    # 出力結果を表示
    if output:
        print_output(output)
