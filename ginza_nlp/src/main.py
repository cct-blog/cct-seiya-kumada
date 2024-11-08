import spacy

# 参照したサイト
# https://qiita.com/kinakomochi_/items/95990d139f5e5e57fd67
# https://github.com/megagonlabs/ginza

output_path_1 = "./outputs/output_1.txt"
text_1 = (
    "7日昼すぎにインドネシアの火山で起きた大規模な噴火について、"
    "気象庁は噴火による津波の有無を調べていましたが、"
    "午後9時半に「日本への津波の影響はない」と発表しました。"
)

output_path_2 = "./outputs/output_2.txt"
text_2 = (
    "世界の火山について調査をしているアメリカのスミソニアン自然史博物館のホームページによりますと、"
    "「レウォトビ火山」はインドネシアのフローレス島、"
    "東部に位置する標高およそ1700メートルの火山です。"
)

output_path_3 = "./outputs/output_3.txt"
text_3 = (
    "アインシュタインが「奇跡の年」を過ごしたベルンのクラム通り49番地は、"
    "現在アインシュタイン・ハウスという名の記念館となっており、"
    "アインシュタイン一家が使っていた家具が当時のスタイルのまま再現されている。"
    "また、ベルン市内にあるベルン歴史博物館には、"
    "アインシュタインの業績や生涯を紹介するアインシュタイン・ミュージアムが入っている。"
)

text_4 = (
    "国境の長いトンネルを抜けると雪国であった。夜の底が白くなった。信号所に汽車が止まった。"
    "向側の座席から娘が立って来て、島村の前のガラス窓を落した。雪の冷気が流れこんだ。"
)

# 森鴎外「最後の一句」
output_path_5 = "./outputs/output_5.txt"
text_5 = (
    "元文三年十一月二十三日の事である。大阪で、船乘業桂屋太郎兵衞と云ふものを、"
    "木津川口で三日間曝した上、斬罪に處すると、高札に書いて立てられた。"
    "市中到る處太郎兵衞の噂ばかりしてゐる中に、それを最も痛切に感ぜなくてはならぬ太郎兵衞の家族は、"
    "南組堀江橋際の家で、もう丸二年程、殆ど全く世間との交通を絶つて暮してゐるのである。"
)


def extract_entities(
    text: str,
    output_path: str,
    nlp: spacy.language.Language,
) -> None:
    doc = nlp(text)

    with open(output_path, "w") as f:
        f.write(f"Text: {text}\n\n")
        # 固有表現の表示
        for ent in doc.ents:
            f.write(
                ent.text
                + ", "  # 固有表現のテキスト自体
                + ent.label_
                + ","  # 固有表現のラベル（人名、地名、組織名など）
                + str(ent.start_char)
                + ","  # 固有表現の開始位置（文字数）
                + str(ent.end_char)
                + "\n"  # 固有表現の終了位置（文字数）
            )


if __name__ == "__main__":
    nlp = spacy.load("ja_ginza_electra")

    # 固有表現の抽出
    extract_entities(text_5, output_path_5, nlp)
