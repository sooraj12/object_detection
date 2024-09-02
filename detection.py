import torch
import cv2
import math
import time
import numpy as np

from ultralytics import YOLOv10
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


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
output_video_path = "./runs/car.mp4"

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


ctime = 0
ptime = 0
count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        count += 1
        print(f"Frame Count: {count}")
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
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                conf = math.ceil(box.conf[0] * 100) / 100
                cls = int(box.cls[0])
                class_name = cls_names[cls]
                label = f"{class_name}: {conf}"
                textSize = cv2.getTextSize(label, 0, fontScale=0.5, thickness=2)[0]
                c2 = x1 + textSize[0], y1 - textSize[1] - 3
                cv2.rectangle(frame, (x1, y1), c2, (255, 0, 0), -1)
                # add segmentation mask to the frame
                masks, scores, _ = segment_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=box.xyxy,
                    multimask_output=False,
                )
                if show_mask:
                    for i, (mask, score) in enumerate(zip(masks, scores)):
                        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])

                    h, w = mask.shape[-2:]
                    mask = mask.astype(np.uint8)
                    mask_image = np.zeros((h, w, 3), dtype=np.uint8)
                    for i in range(3):
                        mask_image[..., i] = mask * int(color[i] * 255)

                    if show_border:
                        contours, _ = cv2.findContours(
                            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                        )
                        contours = [
                            cv2.approxPolyDP(
                                contour,
                                epsilon=0.01 * cv2.arcLength(contour, True),
                                closed=True,
                            )
                            for contour in contours
                        ]
                        mask_image = cv2.drawContours(
                            mask_image, contours, -1, (255, 255, 255), thickness=2
                        )

                    frame = cv2.addWeighted(frame, 1.0, mask_image, color[3], 0)

                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 2),
                    0,
                    0.5,
                    [255, 255, 255],
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )

        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        cv2.putText(
            frame,
            f"FPS: {str(int(fps))}",
            (10, 50),
            cv2.FONT_HERSHEY_PLAIN,
            3,
            (255, 0, 0),
            3,
        )
        cv2.putText(
            frame,
            f"Frame Count: {str(count)}",
            (10, 100),
            cv2.FONT_HERSHEY_PLAIN,
            3,
            (255, 0, 255),
            3,
        )

        # write to output file
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("1"):
            break

    else:
        break
