import glob
import os
from email.mime import base

from PIL import Image
from rembg import remove

if __name__ == "__main__":
    IMAGE_PATH = "/home/kumada/data/gemini_trial/images/cat.jpg"
    OUTPUT_PATH = "./output.png"

    input_paths = glob.glob("/home/kumada/data/rembg/inputs/*")
    output_dir_path = "/home/kumada/data/rembg/outputs"
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    for input_path in input_paths:
        print(f"> input_path: {input_path}")
        src_image = Image.open(input_path)
        dst_image = remove(src_image)
        basename = os.path.basename(input_path)
        head, _ = os.path.splitext(basename)
        output_path = os.path.join(output_dir_path, f"{head}.png")
        print(f"> output_path: {output_path}")
        dst_image.save(output_path)
        print("Done.")
