from PIL import Image
from rembg import remove

if __name__ == "__main__":
    IMAGE_PATH = "/home/kumada/data/gemini_trial/images/cat.jpg"
    OUTPUT_PATH = "./output.png"
    src_image = Image.open(IMAGE_PATH)
    dst_image = remove(src_image)
    dst_image.save(OUTPUT_PATH)
