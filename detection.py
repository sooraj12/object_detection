import torch
import cv2
import math

from ultralytics import YOLOv10
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from track import Tracking


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

if device.type == "cuda":
    # use bfloat16 if supported, else float16
    torch.autocast("cuda", dtype=torch.float16).__enter__()
    # turn on tfloat32 for Ampere GPUs
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


show_border = False
show_box = True
show_mask = True
output_video_path = "./runs/car1.mp4"

# source files
source_path = "./assets/car.mp4"
image_size = 640
conf_threshold = 0.4

# segmentation config
sam2_checkpoint = "./weights/sam2_hiera_tiny.pt"
sam2_cfg = "sam2_hiera_t.yaml"

# initialize yolo model
model_path = "./weights/yolov10n.pt"
model: YOLOv10 = YOLOv10(model=model_path)

# initialize sam2 model
sam2_model = build_sam2(sam2_cfg, sam2_checkpoint, device=device)
segment_predictor = SAM2ImagePredictor(sam2_model)

# initialize deepsort tracking model
tracking = Tracking()
deep_sort = tracking.initialize_deepsort()

# setup video capture
cap = cv2.VideoCapture(source_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(
    output_video_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (frame_width, frame_height),
)


while cap.isOpened():
    xywh_bboxs = []
    confs = []
    oids = []
    outputs = []
    ret, frame = cap.read()

    if ret:
        # get bounding box using yolo
        results = model.predict(
            source=frame,
            imgsz=image_size,
            conf=conf_threshold,
            save=False,
        )

        # draw box on the frame
        for result in results:
            boxes = result.boxes
            segment_predictor.set_image(frame)
            cls_names = result.names

            for box in boxes:
                # draw bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                bbox_width = abs(x1 - x2)
                bbox_height = abs(y1 - y2)
                xcycwh = [cx, cy, bbox_width, bbox_height]
                xywh_bboxs.append(xcycwh)

                conf = math.ceil(box.conf[0] * 100) / 100
                confs.append(conf)
                cls = int(box.cls[0])
                oids.append(cls)

        xywhs = torch.tensor(xywh_bboxs)
        confidence = torch.tensor(confs)
        outputs = []
        if len(xywhs) > 1:
            outputs = deep_sort.update(xywhs, confidence, oids, frame)
        if len(outputs) > 0:
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -2]
            classID = outputs[:, -1]
            tracking.draw_boxes(frame, bbox_xyxy, identities, classID)
            # add segmentation mask to the frame

        # write to output file
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("1"):
            break

    else:
        break
