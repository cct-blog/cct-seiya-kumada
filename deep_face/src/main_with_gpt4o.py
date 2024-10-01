import argparse
import base64
import json
import os
from dataclasses import dataclass
from typing import Any, Final

from openai import AzureOpenAI

import src.prompt as prompt

DETAIL: Final = "high"
API_VERSION: Final = "2024-02-01"


def extract_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepFace CLI")
    parser.add_argument("--end_point")
    parser.add_argument("--secret_info_path")
    parser.add_argument("--model_name")
    parser.add_argument("--input_dir_path", type=str, help="Path to the directory")
    parser.add_argument("--output_dir_path", type=str, help="Path to the output directory")
    parser.add_argument("--max_tokens", type=int)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    return parser.parse_args()


# 画像変換してrequest用のdictを生成
def create_image_dict(filename: str) -> dict[str, Any]:
    with open(filename, "rb") as image_file:
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64.b64encode(image_file.read()).decode('utf-8')}",
                "detail": DETAIL,
            },
        }


def extract_name(image_path: str) -> str:
    return os.path.basename(image_path).split(".")[0]


@dataclass
class VerifiedInfo:
    label: str
    reason: str
    name_1: str
    name_2: str


def judge_image(client, input_path_1, input_path_2, args) -> list[str]:
    image_input_1 = create_image_dict(input_path_1)
    image_input_2 = create_image_dict(input_path_2)

    # request用のメッセージを作成します
    request_messages = {
        "role": "user",
        "content": [{"type": "text", "text": prompt.INPUT_PROMPT}] + [image_input_1, image_input_2],  # 2枚の画像を送信
    }

    response = client.chat.completions.create(
        model=args.model_name,  # 生成したモデルのデプロイ名を指定
        messages=[request_messages],  # type: ignore
        max_tokens=args.max_tokens,  # outputのトークン数の最大値
        temperature=args.temperature,
        top_p=args.top_p,
    )

    items = response.choices[0].message.content.split("\n")  # type: ignore
    return items


if __name__ == "__main__":
    # Extract arguments
    args = extract_args()
    assert os.path.isdir(args.input_dir_path)

    # Extract image paths
    image_paths = [os.path.join(args.input_dir_path, image) for image in os.listdir(args.input_dir_path)]
    image_paths.sort()

    secret_info_path = args.secret_info_path
    secret_info_data = json.load(open(secret_info_path, "r"))

    # clientを生成
    client = AzureOpenAI(
        # 生成したリソースのエンドポイントです
        azure_endpoint=args.end_point,
        # https://learn.microsoft.com/en-us/azure/ai-services/openai/reference を参照してバージョンを指定
        api_version=API_VERSION,
        # 生成したリソースのキーです
        api_key=secret_info_data["KEY_1"],
    )

    # Open the result file
    result_path = os.path.join(args.output_dir_path, "results.txt")
    infos = []

    # Iterate over all pairs of images
    for i in range(len(image_paths)):
        for j in range(i + 1, len(image_paths)):

            print("# Save the result to a JSON file and a text file")
            name_i = extract_name(image_paths[i])
            name_j = extract_name(image_paths[j])

            print(f"# Verify the similarity between the two images, {name_i} and {name_j}")
            try:
                items = judge_image(client, image_paths[i], image_paths[j], args)
                print(items)
                label = items[0].split(":")[1].strip()
                reason = items[1].split(":")[1].strip()
                info = VerifiedInfo(label, reason, name_i, name_j)
                infos.append(info)

            except Exception as e:
                print(f"Error: {e}")
                continue

    output_path = os.path.join(args.output_dir_path, "results.csv")
    with open(output_path, "w", encoding="utf-8-sig") as f:
        f.write("name_1,name_2,label,reason\n")
        for info in infos:
            f.write(f"{info.name_1},{info.name_2},{info.label},{info.reason}\n")
