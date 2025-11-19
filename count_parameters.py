from mmdet.apis import init_detector
import torch

models = {'cascade_rcnn': {'cfg': '/home/jaa/Work/Prog/BSU/Detectors/work_dirs/cascade_rcnn/cascade_rcnn_r50_20e_8b_warm-up_custom/cascade-rcnn_r50_fpn_20e_wandb_custom.py',
                           'weights': '/home/jaa/Work/Prog/BSU/Detectors/work_dirs/cascade_rcnn/cascade_rcnn_r50_20e_8b_warm-up_custom/best_epoch_13.pth'},
          'CODETR': {'cfg': '/home/jaa/Work/Prog/BSU/Detectors/work_dirs/CODETR/CODETR_r50_20e_New_custom/co-detr_r50_1x_custom.py',
                     'weights': '/home/jaa/Work/Prog/BSU/Detectors/work_dirs/CODETR/CODETR_r50_20e_New_custom/best_epoch_17.pth'},
          'DETR': {'cfg': '/home/jaa/Work/Prog/BSU/Detectors/work_dirs/DETR/DETR_r50_75e_AdamW_2_custom/detr_m_1x_custom_resume.py',
                     'weights': '/home/jaa/Work/Prog/BSU/Detectors/work_dirs/DETR/DETR_r50_75e_AdamW_2_custom/best_epoch_72.pth'},
          'DINO': {'cfg': '/home/jaa/Work/Prog/BSU/Detectors/work_dirs/DINO/DINO_r50_25e_AdamW_15_18_custom/DINO_r50_1x_custom_new.py',
                     'weights': '/home/jaa/Work/Prog/BSU/Detectors/work_dirs/DINO/DINO_r50_25e_AdamW_15_18_custom/best_epoch_24.pth'},
          'RetinaNet': {'cfg': '/home/jaa/Work/Prog/BSU/Detectors/work_dirs/RetinaNet/RetinaNet_r50_24e_SGD_custom/retinanet_r50_fpn_1x_new_custom.py',
                     'weights': '/home/jaa/Work/Prog/BSU/Detectors/work_dirs/RetinaNet/RetinaNet_r50_24e_SGD_custom/best_epoch_10.pth'},
          'RTDETR': {'cfg': '/home/jaa/Work/Prog/BSU/Detectors/work_dirs/RTDETR/RTDETR_r50_36e_5b_New_custom/rtdetr_r50vd_1x_custom.py',
                     'weights': '/home/jaa/Work/Prog/BSU/Detectors/work_dirs/RTDETR/RTDETR_r50_36e_5b_New_custom/best_epoch_32.pth'},
          'RTMDet': {'cfg': '/home/jaa/Work/Prog/BSU/Detectors/work_dirs/RTMDet/RTMDet_m_30e_new1_custom/rtmdet_m_1x_custom_resume.py',
                     'weights': '/home/jaa/Work/Prog/BSU/Detectors/work_dirs/RTMDet/RTMDet_m_30e_new1_custom/best_epoch_17.pth'},
          }

# Loading the model configuration and checkpoint
total = []
for model_name in models:
    config_file = models[model_name]['cfg']
    checkpoint_file = models[model_name]['weights']
    model = init_detector(config_file, checkpoint_file, device='cpu')

    # Counting the total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    total.append({model_name: total_params})
for item in total:
    print(f"Total number of parameters: {item}")
