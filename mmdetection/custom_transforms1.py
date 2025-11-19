from mmpretrain.datasets.transforms import ColorJitter
from mmdet.registry import TRANSFORMS

TRANSFORMS.register_module(name='ColorJitter', module=ColorJitter)