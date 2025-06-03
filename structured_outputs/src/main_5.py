from openai import AzureOpenAI

import src.utils as utils

REVIEW_TEXT = (
    "小泉氏は2015年10月から自民党の農林部会長を務め、JAグループの改革に取り組んだ。"
    "政府の規制改革会議は、競争力を強化するためにJA全農の株式会社化などの組織改革案を打ち出し、"
    "小泉氏も支持していた。"
)


if __name__ == "__main__":
    # Azure OpenAIクライアントの初期化
    client = AzureOpenAI(
        azure_endpoint=utils.azure_openai_endpoint,
        api_key=utils.azure_openai_api_key,
        api_version=utils.azure_openai_api_version,
    )

    # レビューのテキストから情報を抽出
    response = client.chat.completions.create(
        model=utils.azure_openai_model_name,
        messages=[
            {
                "role": "system",
                "content": "あなたは情報抽出の専門家です。",
            },
            {
                "role": "user",
                "content": (
                    "以下のテキストから、[人物名]、[日時]、[組織名]を抽出してください。\n"
                    "回答はそのままJSON形式で保存できるように出力してください。\n"
                    "その際、キー名は[]内の文字列を使ってください。\n"
                    f"{REVIEW_TEXT}"
                ),
            },
        ],
        temperature=0.0,
        top_p=1.0,
    )

    # レスポンスから抽出結果を取得
    output = response.choices[0].message.content

    # 出力結果を表示
    if output:
        print(output)
