from deep_translator import MicrosoftTranslator

# Azure TranslatorのAPIキーとリージョンを設定
AZURE_TRANSLATOR_KEY = "8EcxdPAJmDpozinTPiTn5m99zJ3CbmPP4AGQxOWs6G8h2ZdlogeeJQQJ99BBACi0881XJ3w3AAAbACOGurLK"
AZURE_TRANSLATOR_REGION = "japaneast"  # 例: "eastus"

# 翻訳するテキスト
TEXT_1 = (
    "A meeting with the Alibaba co-founder and other prominent entrepreneurs "
    "signals Beijing’s endorsement for a long-marginalized private sector."
)
TEXT_2 = (
    "The departure of the acting commissioner is the latest backlash "
    "to the Department of Government Efficiency’s efforts to access sensitive data."
)


def translate_text(target: str) -> None:
    """Azure Translatorを使って翻訳する"""
    translator = MicrosoftTranslator(
        api_key=AZURE_TRANSLATOR_KEY, region=AZURE_TRANSLATOR_REGION, source="en", target=target
    )

    result = translator.translate(text=TEXT_1)
    print(f"{target}: {result}")

    result = translator.translate(text=TEXT_2)
    print(f"{target}: {result}")


if __name__ == "__main__":
    # 日本語への翻訳
    translate_text("ja")

    # フランス語への翻訳
    translate_text("fr")

    # ドイツ語への翻訳
    translate_text("de")
