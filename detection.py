import torch
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
    xyxy = bbox.xyxy
    image = result.orig_img
    predictor.set_image(image)

    input_box = np.array(xyxy[0].tolist())

    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=False,
    )

    print(masks)
