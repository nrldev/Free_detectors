import json
import os

# Loading ground truth COCO JSON
with open('/home/jaa/Work/Prog/BSU/Detectors/mmdetection/data/Padded_GroundingDINO.v1i.yolov5pytorch/test_coco.json', 'r') as f:
    coco_gt = json.load(f)

# Creating a mapping filename -> image_id, taking only the base name
filename_to_id = {os.path.basename(img['file_name']): img['id'] for img in coco_gt['images']}

# Uploading YOLO predictions
with open('/home/jaa/Work/Prog/BSU/Detectors/mmdetection/data/Padded_GroundingDINO.v1i.yolov5pytorch/Сomparison of detectors/YOLOv11l/predictions.json', 'r') as f:
    yolo_preds = json.load(f)

# Transform predictions
coco_preds = []
for pred in yolo_preds:
    base_name = pred['image_id']
    image_id = filename_to_id.get(base_name+".jpg")
    if image_id is not None:
        coco_pred = {
            'image_id': image_id,
            'category_id': pred['category_id'],
            'bbox': pred['bbox'],
            'score': pred['score']
        }
        coco_preds.append(coco_pred)

# Saving the result
with open('/home/jaa/Work/Prog/BSU/Detectors/work_dirs/results/YOLOv11/predictions.json', 'w') as f:
    json.dump(coco_preds, f)

print(f"Saved by {len(coco_preds)} predictions in predictions.json")
