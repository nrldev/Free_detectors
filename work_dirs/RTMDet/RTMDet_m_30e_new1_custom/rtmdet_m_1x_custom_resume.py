HOME = '/home/jaa/Work/Prog/BSU/Detectors'
MODEL_GROUP = 'yolo-like'
MODEL_TYPE = 'RTMDet'
WEIGHT_SIZE = 'm'
additions = 'new1'
auto_scale_lr = dict(base_batch_size=6, enable=False)
backend_args = None
base_lr = 0.004
batch_size = 6
checkpoint_config = dict(interval=1)
# custom_hooks = [
#     dict(type='CustomFreezeHook', unfreeze_epoch=5),
# ]
# custom_imports = dict(
#     allow_failed_imports=False, imports=[
#         'custom_hooks',
#     ])
data_root = '/home/jaa/Work/Prog/BSU/Detectors/mmdetection/data/Padded_GroundingDINO.v1i.yolov5pytorch/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=7, type='CheckpointHook'),
    early_stopping=dict(
        min_delta=0.001,
        monitor='coco/bbox_mAP',
        patience=3,
        type='EarlyStoppingHook'),
    logger=dict(interval=50, log_metric_by_epoch=True, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
deterministic = True
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
evaluation = dict(interval=1, metric='bbox')
experiment = 'RTMDet_m_30e_new1_custom'
img_scales = [
    (
        640,
        640,
    ),
    (
        320,
        320,
    ),
    (
        960,
        960,
    ),
]
interval = 10
launcher = 'none'
load_from = '/home/jaa/Work/Prog/BSU/Detectors/work_dirs/RTMDet/RTMDet_m_30e_new3_custom/epoch_5.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
max_epochs = 300
metainfo = dict(
    classes=('cow', ), palette=[
        (
            220,
            20,
            60,
        ),
    ])
model = dict(
    backbone=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        arch='P5',
        channel_attention=True,
        deepen_factor=0.67,
        expand_ratio=0.5,
        norm_cfg=dict(type='SyncBN'),
        type='CSPNeXt',
        widen_factor=0.75),
    bbox_head=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        anchor_generator=dict(
            offset=0, strides=[
                8,
                16,
                32,
            ], type='MlvlPointGenerator'),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        exp_on_reg=True,
        feat_channels=192,
        in_channels=192,
        loss_bbox=dict(loss_weight=2.0, type='GIoULoss'),
        loss_cls=dict(
            beta=2.0,
            loss_weight=1.0,
            type='QualityFocalLoss',
            use_sigmoid=True),
        norm_cfg=dict(type='SyncBN'),
        num_classes=1,
        pred_kernel_size=1,
        share_conv=True,
        stacked_convs=2,
        type='RTMDetSepBNHead',
        with_objectness=False),
    data_preprocessor=dict(
        batch_augments=None,
        bgr_to_rgb=False,
        mean=[
            103.53,
            116.28,
            123.675,
        ],
        std=[
            57.375,
            57.12,
            58.395,
        ],
        type='DetDataPreprocessor'),
    neck=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        expand_ratio=0.5,
        in_channels=[
            192,
            384,
            768,
        ],
        norm_cfg=dict(type='SyncBN'),
        num_csp_blocks=2,
        out_channels=192,
        type='CSPNeXtPAFPN'),
    test_cfg=dict(
        max_per_img=300,
        min_bbox_size=0,
        nms=dict(iou_threshold=0.65, type='nms'),
        nms_pre=30000,
        score_thr=0.001),
    train_cfg=dict(
        allowed_border=-1,
        assigner=dict(topk=13, type='DynamicSoftLabelAssigner'),
        debug=False,
        pos_weight=-1),
    type='RTMDet')
num_classes = 1
num_epochs = 20
num_workers = 6
optim_wrapper = dict(
    optimizer=dict(lr=0.0025, momentum=0.9, type='SGD', weight_decay=0.0005),
    paramwise_cfg=dict(
        bias_decay_mult=0, bypass_duplicate=True, norm_decay_mult=0),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=500, start_factor=0.001, type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=20,
        gamma=0.1,
        milestones=[
            7,
            14,
        ],
        type='MultiStepLR'),
    dict(T_max=20, begin=0, by_epoch=True, end=20, type='CosineAnnealingLR'),
]
resume = True
seed = 852
stage2_num_epochs = 20
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=6,
    dataset=dict(
        ann_file=
        '/home/jaa/Work/Prog/BSU/Detectors/mmdetection/data/Padded_GroundingDINO.v1i.yolov5pytorch/test_coco.json',
        backend_args=None,
        data_prefix=dict(
            img=
            '/home/jaa/Work/Prog/BSU/Detectors/mmdetection/data/Padded_GroundingDINO.v1i.yolov5pytorch/test/images'
        ),
        data_root=
        '/home/jaa/Work/Prog/BSU/Detectors/mmdetection/data/Padded_GroundingDINO.v1i.yolov5pytorch/',
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                keep_empty=False,
                min_gt_bbox_wh=(
                    16,
                    16,
                ),
                type='FilterAnnotations'),
            dict(keep_ratio=True, scale=(
                1280,
                1280,
            ), type='Resize'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=6,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file=
    '/home/jaa/Work/Prog/BSU/Detectors/mmdetection/data/Padded_GroundingDINO.v1i.yolov5pytorch/test_coco.json',
    backend_args=None,
    format_only=True,
    metric='bbox',
    outfile_prefix=
    '/home/jaa/Work/Prog/BSU/Detectors/mmdetection/data/Padded_GroundingDINO.v1i.yolov5pytorch/test/images',
    proposal_nums=(
        100,
        1,
        10,
    ),
    type='CocoMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        keep_empty=False, min_gt_bbox_wh=(
            16,
            16,
        ), type='FilterAnnotations'),
    dict(keep_ratio=True, scale=(
        1280,
        1280,
    ), type='Resize'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
train_cfg = dict(
    dynamic_intervals=[
        (
            280,
            1,
        ),
    ],
    max_epochs=30,
    type='EpochBasedTrainLoop',
    val_interval=1)
train_dataloader = dict(
    batch_sampler=None,
    batch_size=6,
    dataset=dict(
        ann_file=
        '/home/jaa/Work/Prog/BSU/Detectors/mmdetection/data/Padded_GroundingDINO.v1i.yolov5pytorch/train_coco.json',
        backend_args=None,
        data_prefix=dict(
            img=
            '/home/jaa/Work/Prog/BSU/Detectors/mmdetection/data/Padded_GroundingDINO.v1i.yolov5pytorch/train/images'
        ),
        data_root=
        '/home/jaa/Work/Prog/BSU/Detectors/mmdetection/data/Padded_GroundingDINO.v1i.yolov5pytorch/',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        metainfo=dict(classes=('cow', ), palette=[
            (
                220,
                20,
                60,
            ),
        ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(prob=0.5, type='RandomFlip'),
            dict(level=2, prob=0.3, type='Rotate'),
            dict(
                keep_empty=False,
                min_gt_bbox_wh=(
                    16,
                    16,
                ),
                type='FilterAnnotations'),
            dict(keep_ratio=True, scale=(
                1280,
                1280,
            ), type='Resize'),
            dict(type='PackDetInputs'),
        ],
        type='CocoDataset'),
    num_workers=6,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(prob=0.5, type='RandomFlip'),
    dict(level=2, prob=0.3, type='Rotate'),
    dict(
        keep_empty=False, min_gt_bbox_wh=(
            16,
            16,
        ), type='FilterAnnotations'),
    dict(keep_ratio=True, scale=(
        1280,
        1280,
    ), type='Resize'),
    dict(type='PackDetInputs'),
]
train_pipeline_stage2 = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.1,
            2.0,
        ),
        scale=(
            640,
            640,
        ),
        type='RandomResize'),
    dict(crop_size=(
        640,
        640,
    ), type='RandomCrop'),
    dict(type='YOLOXHSVRandomAug'),
    dict(prob=0.5, type='RandomFlip'),
    dict(pad_val=dict(img=(
        114,
        114,
        114,
    )), size=(
        640,
        640,
    ), type='Pad'),
    dict(type='PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=5,
    dataset=dict(
        ann_file=
        '/home/jaa/Work/Prog/BSU/Detectors/mmdetection/data/Padded_GroundingDINO.v1i.yolov5pytorch/valid_coco.json',
        backend_args=None,
        data_prefix=dict(
            img=
            '/home/jaa/Work/Prog/BSU/Detectors/mmdetection/data/Padded_GroundingDINO.v1i.yolov5pytorch/valid/images'
        ),
        data_root=
        '/home/jaa/Work/Prog/BSU/Detectors/mmdetection/data/Padded_GroundingDINO.v1i.yolov5pytorch/',
        metainfo=dict(classes=('cow', ), palette=[
            (
                220,
                20,
                60,
            ),
        ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                keep_empty=False,
                min_gt_bbox_wh=(
                    16,
                    16,
                ),
                type='FilterAnnotations'),
            dict(keep_ratio=True, scale=(
                1280,
                1280,
            ), type='Resize'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=6,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file=
    '/home/jaa/Work/Prog/BSU/Detectors/mmdetection/data/Padded_GroundingDINO.v1i.yolov5pytorch//valid_coco.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    proposal_nums=(
        100,
        1,
        10,
    ),
    type='CocoMetric')
vis_backends = [
    dict(
        init_kwargs=dict(
            group='yolo-like',
            name='RTMDet_m_30e_6b_new1',
            project='Сomparison of detectors'),
        type='WandbVisBackend'),
]
visualizer = dict(
    type='Visualizer',
    vis_backends=[
        dict(
            init_kwargs=dict(
                group='yolo-like',
                name='RTMDet_m_30e_6b_new1',
                project='Сomparison of detectors'),
            type='WandbVisBackend'),
    ])
weights = 'rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth'
work_dir = '/home/jaa/Work/Prog/BSU/Detectors/work_dirs/RTMDet/RTMDet_m_30e_new1_1_custom/'
