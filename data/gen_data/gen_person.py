from ultralytics import YOLO

model = YOLO('/root/autodl-fs/projects/helmet/weights/yolov8x.pt')

datasets = {
    'train': '/root/autodl-fs/projects/helmet/Safety_Helmet_Train_dataset/images/train',
    'val': '/root/autodl-fs/projects/helmet/Safety_Helmet_Train_dataset/images/val',
    'test': '/root/autodl-fs/projects/helmet/Safety_Helmet_Train_dataset/images/test'
}

person_class_id = 0
class_to_detect = [person_class_id]

base_project_dir = './data/'
base_name = 'person_detection'

for name, path in datasets.items():

    print(f"\n--- {name} ({path}) ---")

    output_name = f"{base_name}/{name}"

    results = model.predict(
        source=path,
        classes=class_to_detect,
        save=False,
        save_txt=True,              # 保存 .txt 标注文件
        project=base_project_dir,
        name=output_name,
        exist_ok=True,
        conf=0.4,
        iou=0.7,
        imgsz=640
    )

