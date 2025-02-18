from deep_translator import GoogleTranslator

TEXT_1 = (
    "A meeting with the Alibaba co-founder and other prominent entrepreneurs"
    " signals Beijing’s endorsement for a long-marginalized private sector."
)

TEXT_2 = (
    "The departure of the acting commissioner is the latest backlash "
    " to the Department of Government Efficiency’s efforts to access sensitive data."
)


def translate_text(target: str) -> None:
    tranlator = GoogleTranslator(source="auto", target=target)

    result = tranlator.translate(text=TEXT_1)
    print(f"{target}:{result}")

    result = tranlator.translate(text=TEXT_2)
    print(f"{target}:{result}")


if __name__ == "__main__":
    # 日本語への翻訳
    translate_text("ja")

    # 仏語への翻訳
    translate_text("fr")

    # 独語への翻訳
    translate_text("de")
    # 言語一覧
    # r = GoogleTranslator().get_supported_languages(as_dict=True)
    # print(len(r))  # 133
