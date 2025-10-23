from ultralytics import YOLO

model = YOLO('./weights/yolov8l.pt')

results = model.train(
    data='my_data.yaml',
    epochs=100,
    imgsz=640,
    batch=-1,
    project='./runs/',
    name='yolov8l_custom'
)

print("./runs/yolov8l_custom/weights/best.pt")