import numpy as np
import mmcv
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS

@TRANSFORMS.register_module()
class RandomCrop(BaseTransform):
    def __init__(self, crop_size, allow_negative_crop=False):
        assert isinstance(crop_size, (list, tuple))
        self.crop_size = crop_size
        self.allow_negative_crop = allow_negative_crop

    def transform(self, results):
        img = results['img']
        h, w, _ = img.shape
        ch, cw = self.crop_size
        ch = min(ch, h)
        cw = min(cw, w)

        x1 = np.random.randint(0, w - cw + 1) if w > cw else 0
        y1 = np.random.randint(0, h - ch + 1) if h > ch else 0
        x2 = x1 + cw
        y2 = y1 + ch

        results['img'] = img[y1:y2, x1:x2]
        results['img_shape'] = results['img'].shape

        if 'gt_bboxes' in results:
            bboxes = results['gt_bboxes']
            bboxes[:, 0::2] = bboxes[:, 0::2] - x1
            bboxes[:, 1::2] = bboxes[:, 1::2] - y1
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, cw)
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, ch)
            ws = bboxes[:, 2] - bboxes[:, 0]
            hs = bboxes[:, 3] - bboxes[:, 1]
            valid = (ws > 1) & (hs > 1)
            if not self.allow_negative_crop:
                bboxes = bboxes[valid]
                results['gt_bboxes'] = bboxes
                if 'gt_labels' in results:
                    results['gt_labels'] = results['gt_labels'][valid]
                if 'gt_masks' in results:
                    results['gt_masks'] = results['gt_masks'][valid]

        return results
