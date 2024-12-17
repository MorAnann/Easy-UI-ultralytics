from ultralytics import YOLO

model = YOLO("D:\yolo\yolo8/ultralytics-main/ultralytics-main/anan_domo/runs\detect/train6\weights/best.pt")
# data指定yaml文件为“anan_data.yaml”,"imgsz"设置图像大小为640，batch为单次训练的图片数量四图片一批次，conf设置置信度，iou设置交并比为最大0.6，device设备为0，workers关闭多线程（Windows下多线程加载数据容易出问题）
results = model.val(data="D:\yolo\yolo8/ultralytics-main/ultralytics-main/anan_domo/image/anan_data.yaml", imgsz=640, batch=4, conf=0.25, iou=0.6, device="cpu", workers=0)

