from glob import glob
from functools import partial
from multiprocessing.pool import Pool
from typing import Dict, List, Tuple
import os
import shutil
import cv2
from tqdm import tqdm
from xml.etree import ElementTree as ET
from sklearn.model_selection import train_test_split

from pathlib import Path


CROP_NORWAY = (1824, 1824)


def get_class_name_to_id() -> Dict[str, int]:
    return {"D00": 0, "D10": 1, "D20": 2, "D40": 3}


def main():
    folder_name = "test_full_validation"
    output_folder = Path(f"./custom_data/{folder_name}").resolve()

    if not output_folder.exists():
        output_folder.mkdir(parents=True)

    images, annotations = create_yolo_dataset(
        [
            Path("../data/Norway/train/images/").resolve(),
            Path("../data/India/train/images/").resolve(),
            Path("../data/Japan/train/images/").resolve(),
            Path("../data/Czech/train/images/").resolve(),
            Path("../data/United_States/train/images/").resolve(),
            Path("../data/China_MotorBike/train/images/").resolve(),
        ],
        [
            Path("../data/Norway/train/annotations/xmls/").resolve(),
            Path("../data/India/train/annotations/xmls/").resolve(),
            Path("../data/Japan/train/annotations/xmls/").resolve(),
            Path("../data/Czech/train/annotations/xmls/").resolve(),
            Path("../data/United_States/train/annotations/xmls/").resolve(),
            Path("../data/China_MotorBike/train/annotations/xmls/").resolve(),
        ],
        output_folder,
    )

    split_and_save_data(images, annotations, output_folder)

    return


def split_and_save_data(
    images: List[Path],
    annotations: List[Path],
    output_folder: Path,
    val_split_filter="Norway",
):
    image_strs = [str(image) for image in images]
    annotation_strs = [str(annotation) for annotation in annotations]
    # Set training data to to be all images except 20% of Norwegian images

    train_images, train_annotations, validation_images, validation_annotations = (
        [],
        [],
        [],
        [],
    )

    for i in range(len(images)):
        if val_split_filter in images[i].name:
            validation_images.append(image_strs[i])
            validation_annotations.append(annotation_strs[i])
        else:
            train_images.append(image_strs[i])
            train_annotations.append(annotation_strs[i])

    (
        train_images_rest,
        val_images,
        train_annotations_rest,
        val_annotations,
    ) = train_test_split(
        validation_images, validation_annotations, test_size=0.1, random_state=1
    )

    train_images.extend(train_images_rest)
    train_annotations.extend(train_annotations_rest)

    train_images, train_annotations = remove_empty_labels(
        train_images, train_annotations
    )

    # Utility function to move images
    def move_files_to_folder(list_of_files, destination_folder):
        for f in list_of_files:
            print(f"Moving {f} to {destination_folder}")
            try:
                shutil.move(f, destination_folder)
            except:
                print(f)
                assert False

    types = ["images", "labels"]
    sets = ["train", "val"]
    for t in types:
        for s in sets:
            os.makedirs(str(output_folder / t / s), exist_ok=False)

    # Move the splits into their folders
    move_files_to_folder(train_images, str(output_folder / "images" / "train") + "/")
    move_files_to_folder(val_images, str(output_folder / "images" / "val") + "/")
    move_files_to_folder(
        train_annotations, str(output_folder / "labels" / "train") + "/"
    )
    move_files_to_folder(val_annotations, str(output_folder / "labels" / "val") + "/")


def convert_to_yolov5(
    info_dict, output_path: Path, preset_size: Tuple[int, int] | None = None
):
    print_buffer = []
    class_name_to_id = get_class_name_to_id()

    # For each bounding box
    for b in info_dict["bboxes"]:
        class_id = 0
        try:
            class_id = class_name_to_id[b["class"]]
        except KeyError:
            print("Invalid Class. Must be one from ", class_name_to_id.keys())

        # Transform the bbox co-ordinates as per the format required by YOLO v5
        b_center_x = (b["xmin"] + b["xmax"]) / 2
        b_center_y = (b["ymin"] + b["ymax"]) / 2
        b_width = b["xmax"] - b["xmin"]
        b_height = b["ymax"] - b["ymin"]

        # Normalise the co-ordinates by the dimensions of the image
        if preset_size is not None:
            image_w, image_h = preset_size
        else:
            image_w, image_h = info_dict["image_size"]
        b_center_x /= image_w
        b_center_y /= image_h
        b_width /= image_w
        b_height /= image_h

        if b_center_x > 1 or b_center_y > 1 or b_width > 1 or b_height > 1:
            print("Invalid bbox co-ordinates")
            print(info_dict["image_size"])
            continue

        # Write the bbox details to the file
        print_buffer.append(
            "{} {:.3f} {:.3f} {:.3f} {:.3f}".format(
                class_id, b_center_x, b_center_y, b_width, b_height
            )
        )

    # Name of the file which we have to save
    save_file_name = os.path.join(
        output_path, info_dict["filename"].replace("jpg", "txt")
    )

    # Save the annotation to disk
    print("\n".join(print_buffer), file=open(save_file_name, "w"))


# Function to get the data from XML Annotation
def extract_info_from_xml(xml_file: Path):
    """
    extract annotation from xml file
    """
    root = ET.parse(str(xml_file)).getroot()

    # Initialise the info dict
    info_dict = {}
    info_dict["bboxes"] = []

    # Parse the XML Tree
    for elem in root:
        # Get the file name
        if elem.tag == "filename":
            info_dict["filename"] = elem.text

        # Get the image size
        elif elem.tag == "size":
            image_size = [0, 0]
            for subelem in elem:
                if subelem.tag == "width":
                    image_size[0] = int(subelem.text)
                if subelem.tag == "height":
                    image_size[1] = int(subelem.text)

            info_dict["image_size"] = tuple(image_size)

        # Get details of the bounding box
        elif elem.tag == "object":
            bbox = {}
            for subelem in elem:
                if subelem.tag == "name":
                    bbox["class"] = subelem.text

                elif subelem.tag == "bndbox":
                    for subsubelem in subelem:
                        bbox[subsubelem.tag] = int(float(subsubelem.text))
            info_dict["bboxes"].append(bbox)

    return info_dict


def create_yolo_dataset(
    image_input_paths: List[Path],
    annotation_input_paths: List[Path],
    output_path: Path,
) -> Tuple[List[Path], List[Path]]:
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    def create_if_not_exists(path: Path):
        if not path.exists():
            os.mkdir(str(path))

    image_output_path = output_path / "images"
    ann_output_path = output_path / "labels"

    [create_if_not_exists(path) for path in [image_output_path, ann_output_path]]

    images: List[Path] = []
    for image_path in image_input_paths:
        for image in image_path.iterdir():
            images.append(image)

    images.sort()
    already_processed = os.listdir(str(image_output_path))
    images = [file for file in images if file.name not in already_processed]
    images.sort()

    processes = 12
    pool = Pool(processes=processes)
    image_splits = [[] for _ in range(processes)]

    for i in range(processes):
        # split images into 5 parts
        image_splits[i] = images[
            int(i * (len(images) / processes)) : int(
                (i + 1) * (len(images) / processes) + 0.5
            )
        ]

    pool.map(partial(process_images, image_output_path), image_splits)
    pool.close()

    # annotations
    ann_output_path = output_path / "labels"
    annotations: List[Path] = []

    for annotation_path in annotation_input_paths:
        for annotation in annotation_path.iterdir():
            annotations.append(annotation)

    annotations.sort()
    already_processed = os.listdir(str(ann_output_path))
    annotations = [ann for ann in annotations if ann not in already_processed]
    annotations.sort()
    for ann in tqdm(annotations):
        info_dict = extract_info_from_xml(ann)

        if "Norway" in ann.name:
            convert_to_yolov5(info_dict, Path(ann_output_path), CROP_NORWAY)
        else:
            convert_to_yolov5(info_dict, Path(ann_output_path))

    images = [image_output_path / image for image in os.listdir(str(image_output_path))]
    images.sort()
    annotations = [ann_output_path / ann for ann in os.listdir(str(ann_output_path))]
    annotations.sort()

    return images, annotations


def remove_empty_labels(
    images: List[Path], annotations: List[Path]
) -> Tuple[List[Path], List[Path]]:
    new_images = []
    new_annotations = []
    for image, annotation in zip(images, annotations):
        with open(str(annotation), "r") as f:
            # Check if the file has content
            if f.read(1).strip() != "":
                new_images.append(image)
                new_annotations.append(annotation)

    return new_images, new_annotations


def process_images(image_output_path: Path, images: List[Path], crop_filter="Norway"):
    for file in tqdm(images):
        file_name = str(file)
        output_name = str(image_output_path / file.name)
        im = cv2.imread(file_name)
        if crop_filter in file.name:
            im = im[: CROP_NORWAY[0], : CROP_NORWAY[1]]
        im = cv2.resize(im, (640, 640))
        cv2.imwrite(output_name, im)


if __name__ == "__main__":
    main()
