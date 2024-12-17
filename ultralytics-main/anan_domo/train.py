from ultralytics import YOLO

model = YOLO("D:/yolo/yolo8/ultralytics-main/ultralytics-main/anan_domo/image/yolov8n.pt")
results = model.train(data = "D:\yolo\yolo8/ultralytics-main/ultralytics-main/anan_domo\image/anan_data.yaml",epochs=100,imgsz=640,device="cpu",workers=0,batch=4,cache=True)
