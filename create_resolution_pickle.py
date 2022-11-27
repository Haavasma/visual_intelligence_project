from pathlib import Path
from PIL import Image
import pickle
import os
import sys
import argparse
from tqdm import tqdm


def main():
    img_folder_default = (
        "/cluster/projects/vc/courses/TDT17/2022/open/RDD2022/Norway/train/images"
    )
    output_default = "./custom_data/resolution.pkl"
    parser = argparse.ArgumentParser()

    # get folders from arguments
    parser.add_argument("--images", type=str, default=img_folder_default)
    parser.add_argument("--output", type=str, default=output_default)

    args = parser.parse_args()
    image_folder = Path(args.images).resolve()
    output_file = Path(args.output).resolve()

    resolutions = {}

    for images in tqdm(image_folder.iterdir()):
        img = Image.open(images)
        width, height = img.size
        print("HEIGHT: ", height)
        print("WIDTH: ", width)
        resolutions[images.stem] = (height, width)

    print("WRITING TO ", output_file)
    if not output_file.parent.exists():
        os.mkdir(str(output_file.parent))

    pickle.dump(resolutions, open(output_file, "wb"))

    return


if __name__ == "__main__":
    main()
