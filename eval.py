import shutil
import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import subprocess

DIR = '/home/jaa/Work/Prog/BSU/Detectors/'
coco_gt = COCO("/home/jaa/Work/Prog/BSU/Detectors/mmdetection/data/Padded_GroundingDINO.v1i.yolov5pytorch/test_coco.json")

for root, dirs, files in os.walk(DIR + 'work_dirs/results/'):
    for file in files:
        if root != '/home/jaa/Work/Prog/BSU/Detectors/work_dirs/results/DFINE':
            continue
        if file != 'predictions.json':
            continue
        coco_dt = coco_gt.loadRes(root + '/predictions.json')
        evaluator = COCOeval(coco_gt, coco_dt, iouType='bbox')
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        with open(root + '/results.txt', 'w') as f:
            # Перенаправляем stdout в файл
            import sys
            original_stdout = sys.stdout
            sys.stdout = f
            evaluator.summarize()
            sys.stdout = original_stdout
