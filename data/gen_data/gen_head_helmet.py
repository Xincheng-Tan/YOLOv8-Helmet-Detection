import os
import shutil
from pathlib import Path
from shutil import copyfile

from PIL import Image, ImageDraw
from xml.dom.minidom import parse
import numpy as np
from tqdm import tqdm

FILE_ROOT = Path("/root/autodl-fs/projects/helmet/dataset/VOC2028")

IMAGE_SET_ROOT = FILE_ROOT / "ImageSets" / "Main"
IMAGE_PATH = FILE_ROOT / "JPEGImages"
ANNOTATIONS_PATH = FILE_ROOT / "Annotations"
LABELS_ROOT = FILE_ROOT / "Labels"

DEST_IMAGES_PATH = Path("/root/autodl-fs/projects/helmet/Safety_Helmet_Train_dataset/images")
DEST_LABELS_PATH = Path("/root/autodl-fs/projects/helmet/Safety_Helmet_Train_dataset/labels")


def convert_voc_to_darknet(size, box):
    """
    Converts VOC (x_min, y_min, x_max, y_max) bounding box coordinates
    to Darknet (x_center, y_center, width, height) normalized coordinates.

    :param size: Image dimensions [width, height]
    :param box: Anchor box coordinates [x_min, y_min, x_max, y_max]
    :return: Normalized [x, y, w, h]
    """
    img_w, img_h = int(size[0]), int(size[1])
    x1, y1, x2, y2 = [int(val) for val in box]

    width, height = x2 - x1, y2 - y1
    x_center, y_center = x1 + (width / 2), y1 + (height / 2)

    # Normalize by image dimensions
    x_norm = x_center / img_w
    w_norm = width / img_w
    y_norm = y_center / img_h
    h_norm = height / img_h

    return [x_norm, y_norm, w_norm, h_norm]


def save_label_file(img_xml_name, size, img_boxes):
    """
    Processes image boxes, converts coordinates, and saves them to a Darknet-style .txt label file.

    :param img_xml_name: Name of the XML file (used for .txt label filename)
    :param size: Image dimensions [w, h]
    :param img_boxes: List of bounding box data [[cls_name, x1, y1, x2, y2], ...]
    """
    save_file_name = LABELS_ROOT / Path(img_xml_name).with_suffix('.txt')
    label_lines = []

    for box in img_boxes:
        cls_name = box[0]
        if cls_name == 'person':  # 'person' -> 'head' (class 1)
            cls_num = 1
        elif cls_name == 'hat':  # 'hat' -> (class 2)
            cls_num = 2
        else:
            continue

        new_box = convert_voc_to_darknet(size, box[1:])
        label_lines.append(f"{cls_num} {new_box[0]:.6f} {new_box[1]:.6f} {new_box[2]:.6f} {new_box[3]:.6f}")

    if label_lines:
        save_file_name.write_text('\n'.join(label_lines) + '\n', encoding="UTF-8")


def test_dataset_box_feature(file_name, point_array):
    """
    Visualizes bounding boxes on an image for verification.
    :param file_name: Image filename (stem, without extension)
    :param point_array: List of boxes [[cls_name, x1, y1, x2, y2], ...]
    """
    im = Image.open(IMAGE_PATH / Path(file_name).with_suffix(".jpg"))
    im_draw = ImageDraw.Draw(im)
    for box in point_array:
        # box format is [cls, x1, y1, x2, y2]
        x1, y1, x2, y2 = box[1], box[2], box[3], box[4]
        im_draw.rectangle((x1, y1, x2, y2), outline='red')
    im.show()


def get_xml_data(img_xml_file: Path):
    """
    Parses a single VOC XML annotation file to extract image size and object bounding boxes.
    Calls save_label_file to create the Darknet label.

    :param img_xml_file: Path to the XML file.
    """
    try:
        dom = parse(str(img_xml_file))
    except Exception as e:
        print(f"Error parsing XML file {img_xml_file}: {e}")
        return

    xml_root = dom.documentElement
    img_size = xml_root.getElementsByTagName("size")[0]
    objects = xml_root.getElementsByTagName("object")

    img_w = img_size.getElementsByTagName("width")[0].childNodes[0].data
    img_h = img_size.getElementsByTagName("height")[0].childNodes[0].data

    img_boxes = []
    for box_element in objects:
        cls_name = box_element.getElementsByTagName("name")[0].childNodes[0].data
        x1 = int(box_element.getElementsByTagName("xmin")[0].childNodes[0].data)
        y1 = int(box_element.getElementsByTagName("ymin")[0].childNodes[0].data)
        x2 = int(box_element.getElementsByTagName("xmax")[0].childNodes[0].data)
        y2 = int(box_element.getElementsByTagName("ymax")[0].childNodes[0].data)
        img_boxes.append([cls_name, x1, y1, x2, y2])

    # test a box:
    # test_dataset_box_feature(img_xml_file.stem, img_boxes)

    save_label_file(img_xml_file.stem, [img_w, img_h], img_boxes)


def copy_data(img_set_source, img_labels_root, imgs_source, dataset_type):
    """
    Copies image and label files for a specific dataset split (train/val/test)
    to the final destination directories.

    :param img_set_source: Path to the ImageSets/Main directory.
    :param img_labels_root: Path to the generated .txt labels directory.
    :param imgs_source: Path to the source JPEGImages directory.
    :param dataset_type: Dataset split name ('train', 'val', or 'test').
    """
    dest_img_dir = DEST_IMAGES_PATH / dataset_type
    dest_label_dir = DEST_LABELS_PATH / dataset_type

    os.makedirs(dest_img_dir, exist_ok=True)
    os.makedirs(dest_label_dir, exist_ok=True)

    # Read image names from the dataset split file (e.g., train.txt)
    set_file = img_set_source / f"{dataset_type}.txt"
    try:
        with open(set_file, encoding="UTF-8") as f:
            img_names = f.read().splitlines()
    except FileNotFoundError:
        print(f"Warning: Set file not found: {set_file}. Skipping {dataset_type} copy.")
        return

    for img_name in tqdm(img_names, desc=f"Copying {dataset_type}"):
        # Source paths (image name is stem, e.g., '000001')
        img_sor_file = imgs_source / f"{img_name}.jpg"
        label_sor_file = img_labels_root / f"{img_name}.txt"
        dest_img_file = dest_img_dir / f"{img_name}.jpg"
        dest_label_file = dest_label_dir / f"{img_name}.txt"

        copyfile(img_sor_file, dest_img_file)
        copyfile(label_sor_file, dest_label_file)


if __name__ == '__main__':
    # 1. Prepare/Clean Labels Directory
    if LABELS_ROOT.exists():
        print("Cleaning Label dir for safety generating label, pls wait...")
        shutil.rmtree(LABELS_ROOT)
        print("Cleaning Label dir done!")
    LABELS_ROOT.mkdir(exist_ok=True)

    # 2. Generate Darknet Labels (.txt) from VOC Annotations (.xml)
    print("Generating Label files...")
    xml_files = list(ANNOTATIONS_PATH.iterdir())
    with tqdm(total=len(xml_files), desc="Processing XMLs") as p_bar:
        for file in xml_files:
            p_bar.update(1)
            get_xml_data(file)

    # 3. Copy Images and Labels to Final Dataset Structure
    for dataset_input_type in ["train", "val", "test"]:
        copy_data(IMAGE_SET_ROOT, LABELS_ROOT, IMAGE_PATH, dataset_input_type)
        print(f"Copying data {dataset_input_type} complete.")
