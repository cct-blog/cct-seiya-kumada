import json
import os
import textwrap

import dotenv
import langextract as lx
from langextract.inference import OpenAILanguageModel

dotenv.load_dotenv()


def define_extraction_task_01() -> tuple[str, list[lx.data.ExampleData]]:

    # 1. Define the prompt and extraction rules
    prompt = textwrap.dedent(
        """\
        Extract characters, emotions, and relationships in order of appearance.
        Use exact text for extractions. Do not paraphrase or overlap entities.
        Provide meaningful attributes for each entity to add context."""
    )

    # 2. Provide a high-quality example to guide the model
    examples = [
        lx.data.ExampleData(
            text="ROMEO. But soft! What light through yonder window breaks? It is the east, and Juliet is the sun.",
            extractions=[
                lx.data.Extraction(
                    extraction_class="character", extraction_text="ROMEO", attributes={"emotional_state": "wonder"}
                ),
                lx.data.Extraction(
                    extraction_class="emotion", extraction_text="But soft!", attributes={"feeling": "gentle awe"}
                ),
                lx.data.Extraction(
                    extraction_class="relationship",
                    extraction_text="Juliet is the sun",
                    attributes={"type": "metaphor"},
                ),
            ],
        )
    ]
    return prompt, examples


def define_extraction_task_02() -> tuple[str, list[lx.data.ExampleData]]:
    # Define extraction prompt
    prompt_description = "Extract medication information including medication name, dosage, route, frequency, and\
         duration in the order they appear in the text."

    # Define example data with entities in order of appearance
    examples = [
        lx.data.ExampleData(
            text="Patient was given 250 mg IV Cefazolin TID for one week.",
            extractions=[
                lx.data.Extraction(extraction_class="dosage", extraction_text="250 mg"),  # 投薬量
                lx.data.Extraction(extraction_class="route", extraction_text="IV"),  # 投与経路 (IV = 静脈内)
                lx.data.Extraction(extraction_class="medication", extraction_text="Cefazolin"),  # 薬剤名
                lx.data.Extraction(extraction_class="frequency", extraction_text="TID"),  # TID = three times a day
                lx.data.Extraction(extraction_class="duration", extraction_text="for one week"),  # 投与期間
            ],
        )
    ]
    return prompt_description, examples


def define_extraction_task_02_japanese() -> tuple[str, list[lx.data.ExampleData]]:
    # Define extraction prompt
    prompt_description = "テキストから、薬剤名、投薬量、投与経路、投与頻度、投与期間を抽出せよ。"

    # Define example data with entities in order of appearance
    examples = [
        lx.data.ExampleData(
            text="患者は250 mgのIVセファゾリンを1日3回、1週間投与された。",
            extractions=[
                lx.data.Extraction(extraction_class="投薬量", extraction_text="250 mg"),
                lx.data.Extraction(extraction_class="投与経路", extraction_text="IV"),  # IV = 静脈内
                lx.data.Extraction(extraction_class="薬剤名", extraction_text="セファゾリン"),
                lx.data.Extraction(extraction_class="投与頻度", extraction_text="1日3回"),
                lx.data.Extraction(extraction_class="投与期間", extraction_text="1週間"),
            ],
        )
    ]
    return prompt_description, examples


def define_extraction_task_03_japanese() -> tuple[str, list[lx.data.ExampleData]]:
    # Define extraction prompt
    prompt_description = "テキストから、対象者、日時、場所、イベント、行動を抽出せよ。"

    # Define example data with entities in order of appearance
    examples = [
        lx.data.ExampleData(
            text="佐藤さんは、来週火曜日の10時から本社会議室で行われるプロジェクト進捗会議に出席するよう依頼された。",
            extractions=[
                lx.data.Extraction(extraction_class="対象者", extraction_text="佐藤さん"),
                lx.data.Extraction(extraction_class="日時", extraction_text="来週火曜日の10時"),
                lx.data.Extraction(extraction_class="場所", extraction_text="本社会議室"),
                lx.data.Extraction(extraction_class="イベント", extraction_text="プロジェクト進捗会議"),
                lx.data.Extraction(extraction_class="行動", extraction_text="出席するよう依頼された"),
            ],
        )
    ]
    return prompt_description, examples


def visualize_extraction(result) -> None:
    # Save the results to a JSONL file
    lx.io.save_annotated_documents([result], output_name="extraction_results.jsonl", output_dir=".")

    # Generate the visualization from the file
    html_content = lx.visualize("extraction_results.jsonl")
    with open("visualization.html", "w") as f:
        f.write(html_content)


def save_json(result, path):
    # Save result as JSON file (only extraction_class, extraction_text, attributes)
    simplified_extractions = []

    # Handle both single document and iterable of documents
    if hasattr(result, "__iter__") and not hasattr(result, "extractions"):
        # If result is iterable, get first document
        document = next(iter(result))
    else:
        # If result is a single document
        document = result

    for extraction in document.extractions:
        simplified_extractions.append(
            {
                "extraction_class": extraction.extraction_class,
                "extraction_text": extraction.extraction_text,
                "attributes": extraction.attributes,
            }
        )

    with open(path, "w", encoding="utf-8") as f:
        json.dump(simplified_extractions, f, ensure_ascii=False, indent=2)


def make_sample_01():
    prompt, examples = define_extraction_task_01()
    # The input text to be processed
    # input_text = "Lady Juliet gazed longingly at the stars, her heart aching for Romeo"
    input_text = "ジュリエットは星を見つめながら、ロミオへの想いに胸を焦がしていた。"
    return prompt, examples, input_text


def make_sample_02():
    prompt, examples = define_extraction_task_02()
    # The input text to be processed
    input_text = "患者は250 mgのIVセファゾリンを1日3回、1週間投与された。"
    return prompt, examples, input_text


def make_sample_02_japanese():
    prompt, examples = define_extraction_task_02_japanese()
    # The input text to be processed
    # input_text = "患者はセルニルトン錠を、痛みを感じるときに一回2錠、飲むよう言われた。"
    input_text = "医師からは、カロナール錠を発熱したときに1回1錠、服用するよう指示された。"
    return prompt, examples, input_text


def make_sample_03_japanese():
    prompt, examples = define_extraction_task_03_japanese()
    # The input text to be processed
    # input_text = "田中さんは、今週金曜日に本社で行われる製品企画会議に出席するよう依頼された。"
    input_text = "山本さんは、会議室Aで行われる打ち合せに呼ばれた。"
    return prompt, examples, input_text


def main() -> None:
    api_key = os.getenv("LANGEXTRACT_API_KEY")
    if api_key:
        print("API key loaded successfully.")
    else:
        print("Failed to load API key.")

    prompt, examples, input_text = make_sample_03_japanese()

    # Run the extraction
    result = lx.extract(
        text_or_documents=input_text,
        prompt_description=prompt,
        examples=examples,
        language_model_type=OpenAILanguageModel,
        model_id="gpt-4o",
        # api_key=os.environ.get("OPENAI_API_KEY"),
        fence_output=True,
        use_schema_constraints=False,
    )

    save_json(result, "extraction_result_03_japanese.json")
    # visualize_extraction(result)


if __name__ == "__main__":
    main()
