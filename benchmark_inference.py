import time
import torch
import csv
from pathlib import Path
import os
import sys

from mmdet.apis import init_detector
from ultralytics import YOLO
from projects.dfine.src.core import YAMLConfig
from mmdet.apis import init_detector
from mmdet.structures import DetDataSample

WORK_DIR = '/home/jaa/Work/Prog/BSU/Detectors/work_dirs/'

def set_cudnn(mode: str):
    if mode == "R":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    elif mode == "P":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    else:
        raise ValueError("Mode must be 'R' or 'P'")

def create_input_tensor(input_shape, device, normalize=False):
    if normalize:
       # For models requiring normalization (for example, YOLOv11): values in [0, 1]
        tensor = torch.rand(input_shape, device=device)
    else:
        # For models without normalization: values in [0, 255] as pixels
        tensor = torch.rand(input_shape, device=device) * 255
    return tensor

def create_dummy_data_samples(batch_size, input_shape, device):
    # Creating fictitious DetDataSample for mmdet models
    data_samples = []
    for _ in range(batch_size):
        sample = DetDataSample()
        sample.set_metainfo({
            'img_shape': input_shape[2:],  # (H, W), (1280, 1280)
            'ori_shape': input_shape[2:],  # (H, W), (1280, 1280)
            'pad_shape': input_shape[2:],  # (H, W), (1280, 1280)
            'batch_input_shape': input_shape[2:],  # Adding batch_input_shape for DETR and the like
            'scale_factor': (1.0, 1.0)
        })
        data_samples.append(sample)
    return data_samples

def benchmark_model(model, input_shape=(1, 3, 1280, 1280), device='cuda',
                    n_warmup=20, n_runs=100, include_postprocess=True, postprocess_fn=None, normalize=False,
                    requires_data_samples=False):

    model.to(device).eval()
    input_tensor = create_input_tensor(input_shape, device, normalize)
    batch_size = input_shape[0]
    data_samples = create_dummy_data_samples(batch_size, input_shape, device) if requires_data_samples else None

    with torch.inference_mode():
        for i in range(n_warmup):
            if requires_data_samples:
                output = model(input_tensor, data_samples, mode='predict')
                # print(f"Output during warmup with mode='predict': {output}")
            else:
                _ = model(input_tensor)
            if include_postprocess and postprocess_fn:
                _ = postprocess_fn(_)
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats(device)
        mem_mb = torch.cuda.memory_allocated(device) / 1024 / 1024

        start = time.time()
        for _ in range(n_runs):
            if requires_data_samples:
                _ = model(input_tensor, data_samples, mode='predict')
            else:
                _ = model(input_tensor)
            if include_postprocess and postprocess_fn:
                _ = postprocess_fn(_)
        torch.cuda.synchronize()
        end = time.time()

        mem_peak_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024
        mem_reserved_mb = torch.cuda.memory_reserved(device) / 1024 / 1024

    latency_ms = (end - start) / n_runs * 1000
    fps = 1000 / latency_ms
    return latency_ms, fps, mem_mb, mem_peak_mb, mem_reserved_mb

def write_result(csv_path, model_name, input_shape, latency, fps,
                 mem_mb, mem_peak_mb, mem_reserved_mb, mode):
    file_exists = Path(csv_path).exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "model", "mode", "input_h", "input_w",
                "latency_ms", "fps",
                "mem_MB", "mem_peak_MB", "mem_reserved_MB"
            ])
        writer.writerow([
            model_name, mode, input_shape[2], input_shape[3],
            f"{latency:.2f}", f"{fps:.2f}",
            f"{mem_mb:.2f}", f"{mem_peak_mb:.2f}", f"{mem_reserved_mb:.2f}"
        ])

def load_model(model_name, device):
    if model_name == "YOLOv11":
        model_path = "/home/jaa/Work/Prog/BSU/Detectors/mmdetection/data/Padded_GroundingDINO.v1i.yolov5pytorch/Сomparison of detectors/YOLOv11l_RESULT/weights/best.pt"
        model = YOLO(model_path)
        model.to(device)
        model.forward = lambda x: model(x)  # unify interface
        return model

    elif model_name == "D-FINE":
        config = f"/home/jaa/Work/Prog/BSU/Detectors/projects/dfine/configs/dfine/objects365/my_dfine_hgnetv2_l_obj2coco_base4.yml"
        weights = f"/home/jaa/Work/Prog/BSU/Detectors/projects/dfine/output/dfine_hgnetv2_l_obj2coco_base4/best_stg1.pth"
        cfg = YAMLConfig(config, resume=weights)
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
        return model

    elif model_name == "cascade_rcnn":
        subdir = "cascade_rcnn_r50_20e_8b_warm-up_custom"
        config = WORK_DIR + f"{model_name}/{subdir}/" + "cascade-rcnn_r50_fpn_20e_wandb_custom.py"
        checkpoint = WORK_DIR + f"{model_name}/{subdir}/" + "best_epoch_13.pth"
        model = init_detector(config, checkpoint, device=device)
        return model

    elif model_name == "CODETR":
        subdir = "CODETR_r50_20e_New_custom"
        config = WORK_DIR + f"{model_name}/{subdir}/" + "co-detr_r50_1x_custom.py"
        checkpoint = WORK_DIR + f"{model_name}/{subdir}/" + "best_epoch_17.pth"
        model = init_detector(config, checkpoint, device=device)
        return model

    elif model_name == "DETR":
        subdir = "DETR_r50_75e_AdamW_2_custom"
        config = WORK_DIR + f"{model_name}/{subdir}/" + "detr_m_1x_custom_resume.py"
        checkpoint = WORK_DIR + f"{model_name}/{subdir}/" + "best_epoch_72.pth"
        model = init_detector(config, checkpoint, device=device)
        return model

    elif model_name == "DINO":
        subdir = "DINO_r50_25e_AdamW_15_18_custom"
        config = WORK_DIR + f"{model_name}/{subdir}/" + "DINO_r50_1x_custom_new.py"
        checkpoint = WORK_DIR + f"{model_name}/{subdir}/" + "best_epoch_24.pth"
        model = init_detector(config, checkpoint, device=device)
        return model

    elif model_name == "RetinaNet":
        subdir = "RetinaNet_r50_24e_SGD_custom"
        config = WORK_DIR + f"{model_name}/{subdir}/" + "retinanet_r50_fpn_1x_new_custom.py"
        checkpoint = WORK_DIR + f"{model_name}/{subdir}/" + "best_epoch_10.pth"
        model = init_detector(config, checkpoint, device=device)
        return model

    elif model_name == "RTDETR":
        subdir = "RTDETR_r50_36e_5b_New_custom"
        config = WORK_DIR + f"{model_name}/{subdir}/" + "rtdetr_r50vd_1x_custom.py"
        checkpoint = WORK_DIR + f"{model_name}/{subdir}/" + "best_epoch_32.pth"
        model = init_detector(config, checkpoint, device=device)
        return model

    elif model_name == "RTMDet":
        subdir = "RTMDet_m_30e_new1_custom"
        config = WORK_DIR + f"{model_name}/{subdir}/" + "rtmdet_m_1x_custom_resume.py"
        checkpoint = WORK_DIR + f"{model_name}/{subdir}/" + "best_epoch_17.pth"
        model = init_detector(config, checkpoint, device=device)
        return model

    else:
        raise ValueError(f"Unknown model: {model_name}")

def main():
    models_to_test = [
        {"name": "YOLOv11", "input": (1, 3, 1280, 1280), "normalize": True, "requires_data_samples": False},
        {"name": "D-FINE",  "input": (1, 3, 1280, 1280), "normalize": False, "requires_data_samples": False},
        {"name": "cascade_rcnn", "input": (1, 3, 1280, 1280), "normalize": True, "requires_data_samples": True},
        {"name": "CODETR", "input": (1, 3, 1280, 1280), "normalize": False, "requires_data_samples": True},
        {"name": "DETR", "input": (1, 3, 1280, 1280), "normalize": False, "requires_data_samples": True},
        {"name": "DINO", "input": (1, 3, 1280, 1280), "normalize": False, "requires_data_samples": True},
        {"name": "RetinaNet", "input": (1, 3, 1280, 1280), "normalize": False, "requires_data_samples": True},
        {"name": "RTDETR", "input": (1, 3, 1280, 1280), "normalize": False, "requires_data_samples": True},
        {"name": "RTMDet", "input": (1, 3, 1280, 1280), "normalize": False, "requires_data_samples": True},
    ]

    device = "cuda"

    for model_info in models_to_test[:]:
        name = model_info["name"]
        shape = model_info["input"]
        normalize = model_info["normalize"]
        requires_data_samples = model_info["requires_data_samples"]
        print(f"\n--- Testing {name} ---")
        model = load_model(name, device)

        for mode in ["R", "P"]:
            set_cudnn(mode)
            csv_path = "results_repro.csv" if mode == "R" else "results_prod.csv"
            latency, fps, mem_mb, mem_peak_mb, mem_reserved_mb = benchmark_model(model, input_shape=shape,
                                                                                 normalize=normalize, requires_data_samples=requires_data_samples)
            print(f"[{mode}] {name}: {latency:.2f} ms | {fps:.2f} FPS | "
                  f"{mem_mb:.2f} MB used | {mem_peak_mb:.2f} MB peak | {mem_reserved_mb:.2f} MB reserved")
            write_result(csv_path, name, shape, latency, fps, mem_mb, mem_peak_mb, mem_reserved_mb, mode)

if __name__ == "__main__":
    main()
