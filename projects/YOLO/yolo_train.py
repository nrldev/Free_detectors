from ultralytics import YOLO
import albumentations as A
import ultralytics.data.augment as uda
import cv2
import os
from ultralytics.utils import SETTINGS

try:
    SETTINGS["wandb"] = True  # включить коллбеки wandb в Ultralytics
    os.environ.setdefault("WANDB_SILENT", "true")
except Exception:
    pass

    # Correction of a size collision in collate_fn:
    # in the absence of boxes in some of the images, the class label tensor has the shape (0,1),
    # and in the presence of boxes — (N,), which breaks torch.cat in collate_fn.
    # Let's convert 'cls' to the form (N,1) always after Format.
_orig_format_call = uda.Format.__call__

def _format_call_patched(self, labels):
    """Patch Format.__call__:
    Aligns the cls shape to (N,1)
    Filters objects with sizes < 16x16 pixels in each image
    """
    out = _orig_format_call(self, labels)

    # img: Tensor [C, H, W]; bboxes: Tensor [N, 4] (xywh in relative, cause normalize=True)
    img_t = out.get("img")
    bboxes = out.get("bboxes")
    cls_t = out.get("cls")

    # Приводим cls к (N,1)
    if cls_t is not None and cls_t.ndim == 1:
        cls_t = cls_t.unsqueeze(1)

    if img_t is not None and bboxes is not None and bboxes.numel() > 0:
        H = int(img_t.shape[-2])
        W = int(img_t.shape[-1])
        bw = bboxes[:, 2] * W
        bh = bboxes[:, 3] * H
        keep = (bw >= 16) & (bh >= 16)
        if keep.sum() != keep.numel():
            bboxes = bboxes[keep]
            out["bboxes"] = bboxes
            if cls_t is not None:
                cls_t = cls_t[keep]
            if "batch_idx" in out:
                out["batch_idx"] = out["batch_idx"][keep]
            if "keypoints" in out and out["keypoints"].shape[:1] == keep.shape:
                out["keypoints"] = out["keypoints"][keep]
            if "masks" in out and hasattr(out["masks"], "shape") and out["masks"].shape[:1] == keep.shape:
                out["masks"] = out["masks"][keep]

    if cls_t is not None:
        out["cls"] = cls_t
    return out

uda.Format.__call__ = _format_call_patched

model = YOLO("yolo11l.pt")

# IMPORTANT: Ultralytics expects list A.* transforms, not A.Compose.
# He wraps them himself in A.Compose with the correct bbox_params and manages the labels.
# Therefore, we simply pass the list, without bbox_params.
alb_transforms = [
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=10, p=0.3, border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0)),
]

# training: passing the list of transformations to the augmentations parameter
# Also disable the built-in fliplr so that there is no double horizontal flip.
model.train(
    data="/home/jaa/Work/Prog/BSU/Detectors/mmdetection/data/Padded_GroundingDINO.v1i.yolov5pytorch/data.yaml",
    epochs=120,
    imgsz=1280,
    batch=8,
    project="Сomparison of detectors",
    name="YOLOv11l",
    save_period=10,
    # disabling all built-in Ultralytics augmentations, except those transmitted via alb_transforms
    mosaic=0.0,
    mixup=0.0,
    cutmix=0.0,
    copy_paste=0.0,
    degrees=0.0,
    translate=0.0,
    scale=0.0,
    shear=0.0,
    perspective=0.0,
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.0,
    flipud=0.0,
    fliplr=0.0,
    augmentations=alb_transforms,
)
