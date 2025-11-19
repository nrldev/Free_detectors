import json
import torch
from pathlib import Path
from ultralytics import YOLO
from projects.dfine.src.core import YAMLConfig
import cv2
import numpy as np
from pycocotools.coco import COCO

IMAGE_DIR = '/home/jaa/Work/Prog/BSU/Detectors/mmdetection/data/Padded_GroundingDINO.v1i.yolov5pytorch/test/images/'
COCO_GT_PATH = '/home/jaa/Work/Prog/BSU/Detectors/mmdetection/data/Padded_GroundingDINO.v1i.yolov5pytorch/test_coco.json'


def load_model(model_name, device='cuda'):
    if model_name == "YOLOv11":
        model_path = "/home/jaa/Work/Prog/BSU/Detectors/mmdetection/data/Padded_GroundingDINO.v1i.yolov5pytorch/Сomparison of detectors/YOLOv11l_RESULT/weights/best.pt"
        model = YOLO(model_path)
        model.to(device)
        inference_params = {
            'imgsz': 1280,
            'device': device,
            'verbose': False,
            'conf': 0.0
        }
        print(f"YOLOv11 inference parameters: {inference_params}")
        return model, lambda x, model: model.predict(x, **inference_params)

    elif model_name == "D-FINE":
        config = f"/home/jaa/Work/Prog/BSU/Detectors/projects/dfine/configs/dfine/objects365/my_dfine_hgnetv2_l_obj2coco_base4.yml"
        weights = f"/home/jaa/Work/Prog/BSU/Detectors/projects/dfine/output/dfine_hgnetv2_l_obj2coco_base4/best_stg1.pth"
        cfg = YAMLConfig(config, resume=weights)
        if "HGNetv2" in cfg.yaml_cfg:
            cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

        checkpoint = torch.load(weights, map_location=device)
        state = checkpoint["ema"]["module"]
        cfg.model.load_state_dict(state)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = cfg.model.deploy()
                self.postprocessor = cfg.postprocessor.deploy()

            def forward(self, images, orig_target_sizes=None):
                outputs = self.model(images)
                if orig_target_sizes is None:
                    orig_target_sizes = torch.tensor([[1280, 1280]]).to(device)
                outputs = self.postprocessor(outputs, orig_target_sizes)
                return outputs

        model = Model().to(device)
        return model, lambda x, model: model(x, orig_target_sizes=torch.tensor([[1280, 1280]]).to(device))

    else:
        raise ValueError(f"Unknown model: {model_name}")


def load_images(image_dir):
    image_paths = list(Path(image_dir).glob('*.jpg')) + list(Path(image_dir).glob('*.png'))
    images = []
    image_filenames = []
    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is not None:
            images.append(img)
            image_filenames.append(img_path.name)  # используем полное имя
    return images, image_filenames


def process_yolo_results(results, image_id, category_id):
    annotations = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes else None
        scores = result.boxes.conf.cpu().numpy() if result.boxes else None
        labels = result.boxes.cls.cpu().numpy() if result.boxes else None
        for box, score, label in zip(boxes, scores, labels):
            print(f"Box: {box}, Score: {score}, Label: {label}")
            width = float(max(0, box[2] - box[0]))  # Making sure that the width is positiveя
            height = float(max(0, box[3] - box[1]))  # Making sure that the height is positive
            annotations.append({
                'image_id': image_id,
                'bbox': [float(box[0]), float(box[1]), width, height],
                'score': float(score),
                'category_id': int(label) + 1  # Category shift (YOLO uses 0-based, COCO - 1-based)
            })
        print(f"Annotations for image {image_id}: {len(annotations)}")
    return annotations


def process_dfine_results(outputs, image_id, category_id):
    annotations = []
    labels, boxes, scores = outputs
    print(f"Outputs structure: labels={labels.shape if labels is not None else None}, boxes={boxes.shape if boxes is not None else None}, scores={scores.shape if scores is not None else None}")
    scores = scores[0].cpu().numpy()
    labels = labels[0].cpu().numpy()
    boxes = boxes[0].cpu().numpy()
    print(f"Total predictions: {len(scores)}")
    print(f"Scores range: {min(scores)} to {max(scores)}")
    print(f"Labels range: {min(labels)} to {max(labels)}")
    annotation_id = 0
    for idx in range(len(scores)):
        box = boxes[idx]
        x_min, y_min, x_max, y_max = box
        # y_min, x_min, y_max, x_max = box
        width = float(max(0, x_max - x_min))
        height = float(max(0, y_max - y_min))
        if width > 0 and height > 0 and scores[idx] > 0.1:  # Фильтр по score > 0.001
            annotation = {
                'id': annotation_id,
                'image_id': image_id,
                'bbox': [float(x_min), float(y_min), width, height],
                'score': float(scores[idx]),
                'category_id': int(labels[idx]) + 1
            }
            annotations.append(annotation)
            print(f"Annotation {idx}: {annotation}")
            annotation_id += 1
    print(f"Annotations for image {image_id}: {len(annotations)}")
    return annotations


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    models = [
        {"name": "D-FINE", "normalize": True, 'path': '/home/jaa/Work/Prog/BSU/Detectors/work_dirs/results/DFINE'},
        {"name": "YOLOv11", "normalize": True, 'path': '/home/jaa/Work/Prog/BSU/Detectors/work_dirs/results/YOLOv11'},
    ]

    coco_gt = COCO(COCO_GT_PATH)
    cat_id = coco_gt.getCatIds()[0]

    img_ids = coco_gt.getImgIds()
    if img_ids:
        ann_ids = coco_gt.getAnnIds(imgIds=img_ids[0])
        gt_anns = coco_gt.loadAnns(ann_ids)
        print(f"Ground truth for image {img_ids[0]}: {gt_anns}")

    for model_info in models[:1]:
        model_name = model_info["name"]
        normalize = model_info["normalize"]
        model, inference_fn = load_model(model_name, device)
        model.eval()

        all_annotations = []

        for item in coco_gt.dataset["images"]:
            image_id = item["id"]
            # if image_id != 537:
            #     continue
            #
            # print(f"Processing image {image_id}")
            # print(f"Filename: {item['file_name']}")

            img = cv2.imread(f"{item['folder']}/{item['file_name']}")
            img_input = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            orig_target_sizes = torch.tensor([[w, h]]).to(device)

            # Добавляем ресайз, как в torch_inf_my.py
            img_input = cv2.resize(img_input, (1280, 1280), interpolation=cv2.INTER_LINEAR)
            if normalize:
                img_input = img_input.astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_input).permute(2, 0, 1).unsqueeze(0).float().to(device)

            with torch.inference_mode():
                outputs = inference_fn(img_tensor, model)

            if model_name == "YOLOv11":
                new_annotations = process_yolo_results(outputs, image_id, cat_id)
            else:
                new_annotations = process_dfine_results(outputs, image_id, cat_id)

            all_annotations.extend(new_annotations)

        output_path = f"{model_info['path']}/predictions.json"
        with open(output_path, 'w') as f:
            json.dump(all_annotations, f)
        print(f"[INFO] Saved {output_path} with {len(all_annotations)} predictions.")


if __name__ == "__main__":
    main()
