from io import TextIOWrapper
import os
from typing import List, Tuple
import cv2
from pathlib import Path
import time

from format_data import extract_info_from_xml, get_class_name_to_id


def main():
    # original_data()
    parsed_data()


def parsed_data():
    image_dir = Path("./custom_data/test_full_validation/images/train/").resolve()
    label_dir = Path("./custom_data/test_full_validation/labels/train/").resolve()
    show_data(image_dir, label_dir, xml=False, value_filter="China_MotorBike")


def original_data():
    image_dir = Path("../data/Norway/train/images/").resolve()
    label_dir = Path("../data/Norway/train/annotations/xmls/").resolve()
    show_data(image_dir, label_dir, xml=True)


def get_bounding_boxes_xml(
    annotation_file: Path,
) -> Tuple[List[Tuple[int, int, int, int]], List[Tuple[int, int, int]]]:
    labels = extract_info_from_xml(annotation_file)
    class_name_to_id = get_class_name_to_id()
    bboxes = []
    colors = []

    for box in labels["bboxes"]:
        class_id = 0

        try:
            class_id = class_name_to_id[box["class"]]
        except KeyError:
            print("Invalid Class. Must be one from ", class_name_to_id.keys())
        bboxes.append((box["xmin"], box["ymin"], box["xmax"], box["ymax"]))
        colors.append(get_class_color(class_id))

    return bboxes, colors


def get_class_color(class_id: int) -> Tuple[int, int, int]:
    if class_id == 0:
        return (0, 0, 255)
    elif class_id == 1:
        return (0, 255, 0)
    elif class_id == 2:
        return (255, 0, 0)
    else:
        return (0, 0, 0)


def extract_lbrt(
    file: TextIOWrapper, im_shape: Tuple[int, int]
) -> Tuple[List[Tuple[int, int, int, int]], List[int]]:
    """
    Extract the left, bottom, right, top coordinates from the label file
    from the YOLO format.
    """
    line = file.readline()
    # read lines of file until empty
    classes = []
    bboxes = []

    while line is not None and line not in ["", " ", "\n"]:
        class_id, x, y, w, h = [float(s.replace("\n", "")) for s in line.split(" ")]
        class_id = int(float(class_id))

        x *= im_shape[1]
        y *= im_shape[0]
        w *= im_shape[1]
        h *= im_shape[0]

        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        xmin = int(x - (w / 2))
        ymin = int(y - (h / 2))
        xmax = int(x + (w / 2))
        ymax = int(y + (h / 2))
        bboxes.append((xmin, ymin, xmax, ymax))
        classes.append(class_id)
        line = file.readline()

    return bboxes, classes


def get_bounding_boxes_txt(
    label_file: Path, im_shape: Tuple[int, int]
) -> Tuple[List[Tuple[int, int, int, int]], List[Tuple[int, int, int]]]:
    if not label_file.exists():
        return [], []
    file = open(label_file, "r")
    bboxes, classes = extract_lbrt(file, im_shape)
    colors = [get_class_color(class_id) for class_id in classes]

    return (bboxes, colors)


def show_data(image_dir: Path, label_dir: Path, xml=False, value_filter="Norway"):
    images = [image_dir / file_name for file_name in os.listdir(image_dir)]
    images = list(filter(lambda x: value_filter in x.name, images))
    images.sort()

    labels = []
    for image in images:
        labels.append(label_dir / (image.stem + ".txt"))
    labels.sort()

    for i, image in enumerate(images):
        print(image.name)
        print(labels[i].name)
        im = cv2.imread(str(image))

        if xml:
            bboxes, colors = get_bounding_boxes_xml(labels[i])
        else:
            bboxes, colors = get_bounding_boxes_txt(
                labels[i],
                im.shape,
            )

        if len(bboxes) == 0:
            continue

        for i, box in enumerate(bboxes):
            print(box)
            cv2.rectangle(im, (box[0], box[1]), (box[2], box[3]), colors[i], 2)

        # im = cv2.resize(im, None, fx=0.4, fy=0.4)

        cv2.imshow("test", im)

        while True:
            key = cv2.waitKey(0)
            if key & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                return

            if key & 0xFF == ord("n"):
                break

    cv2.destroyAllWindows()
    return


if __name__ == "__main__":
    main()
