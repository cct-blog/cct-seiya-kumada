from unittest import skip

from openai import AzureOpenAI
from pydantic import BaseModel, Field

import src.utils as utils


class WorkExperience(BaseModel):
    """職歴を表すモデル
    Args:
        company (str): 会社名
        position (str): 役職
        year (str): 在籍年数（例: "2021-2025"）
    """

    company: str = Field(description="会社名")
    position: str = Field(description="役職")
    year: str = Field(description="在籍年数")


class Output(BaseModel):
    """抽出結果のモデル
    Args:
        name (str): 名前
        email (str): メールアドレス
        phone (str): 電話番号
        skills (list[str]): スキルのリスト
        experiences (list[WorkExperience]): 職務経歴のリスト
    """

    name: str = Field(description="名前")
    email: str = Field(description="メールアドレス")
    phone: str = Field(description="電話番号")
    skills: list[str] = Field(description="スキルのリスト")
    experiences: list[WorkExperience] = Field(description="職務経歴のリスト")


REVIEW_TEXT = (
    "氏名: 田中 太郎 \n"
    "連絡先: tanaka@example.com, 090-1234-5678 \n"
    "スキル: Python, TensorFlow, Azure OpenAI, Docker \n"
    "職歴: \n"
    "- 2021-2025 ABCテクノロジーズ株式会社 / AIエンジニア \n"
    "- 2018-2021 XYZソフトウェア / ソフトウェア開発者 \n"
)


def print_output(output: Output) -> None:
    """
    出力結果を整形して表示する関数

    Args:
        output (Output): 抽出結果のデータ
    """
    print(f"名前: {output.name}")
    print(f"メールアドレス: {output.email}")
    print(f"電話番号: {output.phone}")
    print("スキル:")
    for skill in output.skills:
        print(f"  - {skill}")
    print("職務経歴:")
    for exp in output.experiences:
        print(f"  - 会社名: {exp.company}, 役職: {exp.position}, 在籍年数: {exp.year}")


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
                "content": "以下の履歴書からプロフィール情報を抽出してください。\n" f"{REVIEW_TEXT}",
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
