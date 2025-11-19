import torch.nn as nn
from mmdet.registry import MODELS
from mmdet.models.detectors import RTDETR

@MODELS.register_module()
class DFINE(RTDETR):
    """D-FINE model based on RT-DETR with FDR and GO-LSD."""
    def __init__(
        self,
        backbone,
        encoder,  # Используем encoder вместо neck
        decoder,  # Используем decoder вместо bbox_head
        criterion=None,
        dn_cfg=None,
        train_cfg=None,
        test_cfg=None,
        data_preprocessor=None,
        init_cfg=None
    ):
        super().__init__(
            backbone=backbone,
            neck=encoder,  # Передаём encoder как neck для RTDETR
            bbox_head=decoder,  # Передаём decoder как bbox_head для RTDETR
            decoder=decoder,  # Передаём decoder как decoder для DINO
            dn_cfg=dn_cfg,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg
        )
        self.criterion = MODELS.build(criterion) if criterion is not None else None

    def loss(self, batch_inputs, batch_data_samples):
        # Используем логику RTDETR, но с DFINECriterion
        x = self.extract_feat(batch_inputs)
        outputs = self.bbox_head(x, batch_data_samples)
        if self.criterion is not None:
            return self.criterion(outputs, batch_data_samples)
        return super().loss(batch_inputs, batch_data_samples)