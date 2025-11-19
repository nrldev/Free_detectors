_base_ = [
    '../_base_/datasets/coco_detection.py', '../_base_/default_runtime.py'
]
pretrained = '/home/jaa/Work/Prog/BSU/Detectors/mmdetection/projects/dfine/weight/hgnetv2/PPHGNetV2_B4_stage1.pth'  # noqa

model = dict(
    type='DFINE',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        batch_augments=[
            dict(
                type='BatchSyncRandomResize',
                interval=1,
                interpolations=['nearest', 'bilinear', 'bicubic', 'area'],
                random_sizes=[
                    480, 512, 544, 576, 608, 640, 640, 640, 672, 704, 736, 768,
                    800
                ])
        ],
        mean=[0, 0, 0],
        std=[255, 255, 255],
        bgr_to_rgb=True,
        pad_size_divisor=1),
    backbone=dict(
        type='HGNetv2',
        name='B4',
        return_idx=[1, 2, 3],
        freeze_stem_only=True,
        freeze_at=0,
        freeze_norm=True,
        pretrained=pretrained,
        local_model_dir='weight/hgnetv2/'),
    encoder=dict(  # Переименовано из neck
        type='HybridEncoder',
        in_channels=[128, 256, 512],
        feat_strides=[8, 16, 32],
        hidden_dim=256,
        nhead=8,
        dim_feedforward=1024,
        dropout=0.0,
        enc_act='gelu',
        use_encoder_idx=[2],
        num_encoder_layers=1,
        pe_temperature=10000,
        expansion=1.0,
        depth_mult=1.0,
        act='silu',
        eval_spatial_size=[640, 640]),
    decoder=dict(  # Переименовано из bbox_head
        type='DFINETransformer',
        num_layers=6,
        hidden_dim=256,
        num_queries=300,
        feat_channels=[256, 256, 256],
        feat_strides=[8, 16, 32],
        num_levels=3,
        num_points=[3, 6, 3],
        nhead=8,
        dim_feedforward=1024,
        dropout=0.0,
        activation='relu',
        num_denoising=100,
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        eval_idx=-1,
        cross_attn_method='default',
        query_select_method='default',
        reg_max=32,
        reg_scale=4.0,
        layer_scale=1,
        eval_spatial_size=[640, 640]),
    criterion=dict(
        type='DFINECriterion',
        weight_dict=dict(loss_vfl=1, loss_bbox=5, loss_giou=2, loss_fgl=0.15, loss_ddf=1.5),
        losses=['vfl', 'boxes', 'local'],
        alpha=0.75,
        gamma=2.0,
        num_classes=80,
        reg_max=32,
        matcher=dict(
            type='HungarianMatcher',
            weight_dict=dict(cost_class=2, cost_bbox=5, cost_giou=2),
            alpha=0.25,
            gamma=2.0)),
    dn_cfg=dict(
        label_noise_scale=0.5,
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=100)),
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
        ])),
    test_cfg=dict(max_per_img=300))

# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
interpolations = ['nearest', 'bilinear', 'bicubic', 'area', 'lanczos']
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomApply',
        transforms=dict(type='PhotoMetricDistortion'),
        prob=0.8),
    dict(type='Expand', mean=[0, 0, 0]),
    dict(
        type='RandomApply',
        transforms=dict(type='MinIoURandomCrop', cover_all_box=False),
        prob=0.8),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[[
            dict(
                type='Resize',
                scale=(640, 640),
                keep_ratio=False,
                interpolation=interpolation)
        ] for interpolation in interpolations]),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(
        type='Resize',
        scale=(640, 640),
        keep_ratio=False,
        interpolation='bicubic'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    dataset=dict(
        filter_cfg=dict(filter_empty_gt=False), pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))

# optimizer
optim_wrapper=dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.00025,
        betas=[0.9, 0.999],
        weight_decay=0.000125),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.05),  # lr=0.0000125
            'encoder|decoder': dict(decay_mult=0.0)  # Для norm/bn
        },
        norm_decay_mult=0,
        bypass_duplicate=True))

# learning policy
max_epochs = 36
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=2000)
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=16)

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        priority=49)
]
