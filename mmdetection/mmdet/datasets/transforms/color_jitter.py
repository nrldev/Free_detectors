from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS
from torchvision.transforms import ColorJitter as TorchColorJitter
from PIL import Image
import numpy as np

@TRANSFORMS.register_module()
class ColorJitter(BaseTransform):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.jitter = TorchColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )

    def transform(self, results):
        img = results['img']
        img_pil = Image.fromarray(img.astype(np.uint8))
        img_jittered = self.jitter(img_pil)
        results['img'] = np.array(img_jittered)
        return results