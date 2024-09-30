from deepface import DeepFace
import os
import json
import argparse


def extract_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepFace CLI")
    parser.add_argument("--input_dir_path", type=str, help="Path to the directory")
    parser.add_argument("--output_dir_path", type=str, help="Path to the output directory")
    return parser.parse_args()


def extract_name(image_path: str) -> str:
    return os.path.basename(image_path).split(".")[0]


if __name__ == "__main__":
    # Extract arguments
    args = extract_args()
    assert os.path.isdir(args.input_dir_path)

    # Extract image paths
    image_paths = [os.path.join(args.input_dir_path, image) for image in os.listdir(args.input_dir_path)]
    image_paths.sort()

    # Open the result file
    result_path = os.path.join(args.output_dir_path, "results.txt")
    with open(result_path, "w") as f:

        # Iterate over all pairs of images
        for i in range(len(image_paths)):
            for j in range(i + 1, len(image_paths)):

                print("# Save the result to a JSON file and a text file")
                name_i = extract_name(image_paths[i])
                name_j = extract_name(image_paths[j])

                print(f"# Verify the similarity between the two images, {name_i} and {name_j}")
                try:
                    result = DeepFace.verify(image_paths[i], image_paths[j])
                except Exception as e:
                    print(f"Error: {e}")
                    continue

                output_path = os.path.join(args.output_dir_path, f"{name_i}_vs_{name_j}.json")
                f.write(f"{name_i}-{name_j}: {result['verified']}\n")
                json.dump(result, open(output_path, "w"))
