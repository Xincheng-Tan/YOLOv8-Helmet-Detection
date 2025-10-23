from ultralytics import YOLO

WEIGHTS_PATH = "./weights/best.pt"
DATA_YAML_PATH = "my_data.yaml"

model = YOLO(WEIGHTS_PATH)

metrics = model.val(
    data=DATA_YAML_PATH,
    imgsz=640,
    split='test',
    project='./runs/',
    name='test',
    save_json=True,
    save_hybrid=False,
    batch=16,
)

print("\n--- 所有类别 (all) 指标 ---")
print(f"Precision (all): {metrics.box.mp:.4f}")         # Mean Precision
print(f"Recall (all):    {metrics.box.mr:.4f}")         # Mean Recall
print(f"mAP@.5 (all):   {metrics.box.map50:.4f}")       # Mean Average Precision at IoU=0.5
print(f"mAP@.5:.95 (all):{metrics.box.map:.4f}")        # Mean Average Precision across IoU 0.5 to 0.95

class_names = metrics.names
map50s_per_class = metrics.box.ap50
maps_per_class = metrics.box.ap
precision_per_class = metrics.box.p
recall_per_class = metrics.box.r

print("\n--- 单类别指标 ---")
for i, name in class_names.items():
    if i < len(maps_per_class) and name in ['person', 'head', 'helmet']: 
        print(f"Class: {name}")
        print(f"  Precision (p): {precision_per_class[i]:.4f}")
        print(f"  Recall (r):    {recall_per_class[i]:.4f}")
        print(f"  mAP@.5:   {map50s_per_class[i]:.4f}")
        print(f"  mAP@.5:.95:{maps_per_class[i]:.4f}")

print(f"\nto ./runs/test")