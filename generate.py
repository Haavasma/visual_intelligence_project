from typing import Dict, List, Tuple
import sys
from pathlib import Path
import os
import argparse
import pickle
from pathlib import Path
from format_data import CROP_NORWAY

from show_data import extract_lbrt


def main():
    folder = "./runs/train/exp/labels/"
    resolution_file = "./runs/train/exp/resolutions.pkl"
    output_default = "result.csv"
    image_folder_default = "fuck off"
    parser = argparse.ArgumentParser()

    parser.add_argument("--yolo-labels", type=str, default=folder)
    parser.add_argument("--resolution-file", type=str, default=resolution_file)
    parser.add_argument("--output", type=str, default=output_default)
    parser.add_argument("--image-folder", type=str, default=image_folder_default)

    args = parser.parse_args()
    folder = Path(args.yolo_labels).resolve() / "exp" / "labels"
    resolution_file = args.resolution_file
    output_file = Path(args.output)
    image_folder = Path(args.image_folder)

    print("FOLDER: ", folder)

    image_to_resolution = {}
    with open(resolution_file, "rb") as f:
        image_to_resolution: Dict[str, Tuple[int, int]] = pickle.load(
            open(resolution_file, "rb")
        )

    label_folder = Path(folder).resolve()

    file_names = []
    for file_name in image_folder.iterdir():
        file_names.append(str(label_folder / (file_name.stem + ".txt")))

    file_names.sort()

    text = ""
    for name in file_names:
        file_name = Path(name).resolve()

        if file_name.exists():
            file = open(file_name, "r")
            boxes, classes = extract_lbrt(file, image_to_resolution[file_name.stem])
            text += build_row(file_name.stem + ".jpg", boxes, classes)
        else:
            text += build_row(file_name.stem + ".jpg", [], [])

    if not output_file.parent.exists():
        os.mkdir(output_file.parent)

    with open(output_file, "w") as f:
        print("writing result to: ", output_file)
        f.write(text)

    return


def build_row(
    file_name: str, boxes: List[Tuple[int, int, int, int]], classes: List[int]
) -> str:
    row = file_name + ","

    entries = []
    for box, class_id in zip(boxes, classes):
        entries.append(f"{class_id+1} {box[0]} {box[1]} {box[2]} {box[3]}")

    row += " ".join(entries)
    row += "\n"

    return row


if __name__ == "__main__":
    main()
