from mmdet.registry import MODELS
from mmdet.models.detectors.dfine import DFINE
print("DFINE registered:", DFINE.__name__ in MODELS)
