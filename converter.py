from pylabel import importer

# The path to YOLO annotations
path_to_annotations = "/home/jaa/Work/Prog/BSU/Detectors/mmdetection/data/Padded_GroundingDINO.v1i.yolov5pytorch/test/labels/"
path_to_images = "/home/jaa/Work/Prog/BSU/Detectors/mmdetection/data/Padded_GroundingDINO.v1i.yolov5pytorch/test/images/"
classes = ["cow"]

# Importing a YOLO dataset
dataset = importer.ImportYoloV5(
    path=path_to_annotations,
    path_to_images=path_to_images,
    cat_names=classes
)

# Export to COCO
dataset.export.ExportToCoco(
    output_path="/home/jaa/Work/Prog/BSU/Detectors/mmdetection/data/Padded_GroundingDINO.v1i.yolov5pytorch/test_coco.json"
)
