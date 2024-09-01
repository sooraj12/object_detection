import torch
import cv2
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
# source files
source_path = "./assets/"
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
predictor = SAM2ImagePredictor(sam2_model)

# get bounding box using yolo
results = model.predict(
    source=source_path,
    imgsz=image_size,
    conf=conf_threshold,
    save=False,
)

for result in results:
    bbox = result.boxes
    xyxy = bbox[0].xyxy
    box = xyxy[0]
    image = result.orig_img
    predictor.set_image(image)

    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=xyxy,
        multimask_output=False,
    )

    # save image with bounding box and mask
    x0, y0 = int(box[0]), int(box[1])
    x1, y1 = int(box[2]), int(box[3])
    for i, (mask, score) in enumerate(zip(masks, scores)):
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])

        h, w = mask.shape[-2:]
        print(h)
        print(w)
        mask = mask.astype(np.uint8)
        mask_image = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(3):  # Apply the color to each channel
            mask_image[..., i] = mask * int(color[i] * 255)

        if show_border:
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            # Try to smooth contours
            contours = [
                cv2.approxPolyDP(
                    contour, epsilon=0.01 * cv2.arcLength(contour, True), closed=True
                )
                for contour in contours
            ]
            mask_image = cv2.drawContours(
                mask_image, contours, -1, (255, 255, 255), thickness=2
            )

        image = cv2.addWeighted(image, 1.0, mask_image, color[3], 0)

    cv2.rectangle(image, (x0, y0), (x1, y1), color=(0, 255, 0), thickness=2)
    cv2.imwrite("./assets/image.jpg", image)
