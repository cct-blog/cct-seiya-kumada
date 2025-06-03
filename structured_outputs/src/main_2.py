from openai import AzureOpenAI
from pydantic import BaseModel, Field

import src.utils as utils


class ProductSpecs(BaseModel):
    """製品仕様を表すモデル

    Args:
        weight (str): 重量
        dimentsions (str): サイズ
        color (list[str]): 色のリスト
    """

    weight: str = Field(description="重さ")
    dimentsions: str = Field(description="サイズ")
    color: list[str] = Field(description="色")


class Output(BaseModel):
    """抽出結果のモデル
    Args:
        product_name (str): 製品名
        price_jpy (int): 価格（日本円）
        brand (str): ブランド名
        specs (ProductSpecs): 製品仕様
    """

    product_name: str = Field(description="製品名")
    price_jpy: int = Field(description="価格（日本円）")
    brand: str = Field(description="ブランド名")
    specs: ProductSpecs = Field(description="製品仕様")


REVIEW_TEXT = (
    "最新モデルの「GalaxyFit Pro」は、サムスン製のフィットネストラッカー。"
    "重さは25g、サイズは40mm×20mm×10mm。カラーバリエーションはブラックのみ。"
    "公式ストアでの販売価格は税込み 14,980円 です。"
)


def print_output(output: Output) -> None:
    """
    出力結果を整形して表示する関数

    Args:
        output (Output): 抽出結果のデータ
    """
    print(f"製品名: {output.product_name}")
    print(f"価格（円）: {output.price_jpy}")
    print(f"ブランド: {output.brand}")
    print(f"重量: {output.specs.weight}")
    print(f"サイズ: {output.specs.dimentsions}")
    print(f"色: {', '.join(output.specs.color)}")


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
                "content": "以下の説明文から、製品名、価格（円）、ブランド、主な仕様（重さ・サイズ・色）を抽出してください。\n"
                f"{REVIEW_TEXT}",
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
