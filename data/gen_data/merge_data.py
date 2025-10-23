import os
from typing import List

PROJECT_ROOT = "/root/autodl-fs/projects/helmet"

# PROJECT_ROOT/data/person_detection/train/labels/
YOLO_BASE_ROOT = os.path.join(PROJECT_ROOT, "data", "person_detection")

# PROJECT_ROOT/Safety_Helmet_Train_dataset/labels/train/
DATASET_BASE_ROOT = os.path.join(PROJECT_ROOT, "Safety_Helmet_Train_dataset", "labels")

SPLITS: List[str] = ['train', 'test', 'val']

def merge_person_labels():
    for split in SPLITS:
        yolo_source_dir = os.path.join(YOLO_BASE_ROOT, split, 'labels')
        dataset_dest_dir = os.path.join(DATASET_BASE_ROOT, split)

        os.makedirs(dataset_dest_dir, exist_ok=True)
        
        processed_count = 0
        for file_name in os.listdir(yolo_source_dir):
            if not file_name.endswith(".txt"):
                continue

            yolo_file_path = os.path.join(yolo_source_dir, file_name)
            dataset_file_path = os.path.join(dataset_dest_dir, file_name)

            lines_to_append = []

            with open(yolo_file_path, "r") as f_read:
                for line in f_read:
                    if line.split()[0] == '0':
                        lines_to_append.append(line)
                    else:
                        continue

            if lines_to_append:
                with open(dataset_file_path, "a") as f_append:
                    f_append.writelines(lines_to_append)
                processed_count += 1
                    

        print(f"{split}, {processed_count}")

if __name__ == '__main__':
    merge_person_labels()
    print("\n--- finish ---")
