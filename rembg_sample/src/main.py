import glob
import os

from PIL import Image
from rembg import remove

if __name__ == "__main__":

    # Set input paths and output directory path
    input_paths = glob.glob("/home/kumada/data/rembg/inputs/*")
    output_dir_path = "/home/kumada/data/rembg/outputs"

    # Create output directory if not exists
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    for input_path in input_paths:
        print(f"> input_path: {input_path}")
        src_image = Image.open(input_path)

        # Remove background
        dst_image = remove(src_image)

        basename = os.path.basename(input_path)
        head, _ = os.path.splitext(basename)
        output_path = os.path.join(output_dir_path, f"{head}.png")
        print(f"> output_path: {output_path}")

        # Save image
        dst_image.save(output_path)  # type: ignore

        print("Done.")
