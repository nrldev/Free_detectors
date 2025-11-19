HOME = '/home/jaa/Work/Prog/BSU/Detectors'
MODEL_GROUP = '1stage'
MODEL_TYPE = 'RetinaNet'
WEIGHT_SIZE = 'r50'
additions = 'SGD'
auto_scale_lr = dict(base_batch_size=10, enable=False)
backend_args = None
batch_size = 10
checkpoint_config = dict(interval=1)
data_root = '/home/jaa/Work/Prog/BSU/Detectors/mmdetection/data/Padded_GroundingDINO.v1i.yolov5pytorch/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook'),
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
experiment = 'RetinaNet_r50_24e_SGD_custom'
img_norm_cfg = dict(
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    std=[
        58.395,
        57.12,
        57.375,
    ],
    to_rgb=True)
launcher = 'none'
load_from = '/home/jaa/Work/Prog/BSU/Detectors/work_dirs/RetinaNet/RetinaNet_r50_24e_SGD_custom/best_epoch_10.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
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
        depth=50,
        frozen_stages=2,
        init_cfg=dict(checkpoint='torchvision://resnet50', type='Pretrained'),
        norm_cfg=dict(requires_grad=True, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style='pytorch',
        type='ResNet'),
    bbox_head=dict(
        anchor_generator=dict(
            octave_base_scale=4,
            ratios=[
                0.5,
                1.0,
                2.0,
            ],
            scales_per_octave=3,
            strides=[
                8,
                16,
                32,
                64,
                128,
            ],
            type='AnchorGenerator'),
        bbox_coder=dict(
            target_means=[
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            target_stds=[
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            type='DeltaXYWHBBoxCoder'),
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(loss_weight=1.0, type='L1Loss'),
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            type='FocalLoss',
            use_sigmoid=True),
        num_classes=1,
        stacked_convs=4,
        type='RetinaHead'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    neck=dict(
        add_extra_convs='on_input',
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        num_outs=5,
        out_channels=256,
        start_level=1,
        type='FPN'),
    test_cfg=dict(
        max_per_img=100,
        min_bbox_size=0,
        nms=dict(iou_threshold=0.5, type='nms'),
        nms_pre=1000,
        score_thr=0.05),
    train_cfg=dict(
        allowed_border=-1,
        assigner=dict(
            ignore_iof_thr=-1,
            min_pos_iou=0,
            neg_iou_thr=0.4,
            pos_iou_thr=0.5,
            type='MaxIoUAssigner'),
        debug=False,
        pos_weight=-1,
        sampler=dict(type='PseudoSampler')),
    type='RetinaNet')
num_classes = 1
num_epochs = 24
num_workers = 6
optim_wrapper = dict(
    optimizer=dict(lr=0.001, momentum=0.9, type='SGD', weight_decay=0.0001))
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=500, start_factor=0.0001,
        type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=24,
        gamma=0.1,
        milestones=[
            8,
            11,
        ],
        type='MultiStepLR'),
]
resume = False
seed = 38
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=10,
    dataset=dict(
        ann_file=
        '/home/jaa/Work/Prog/BSU/Detectors/mmdetection/data/Padded_GroundingDINO.v1i.yolov5pytorch/test_coco.json',
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
    format_only=True,
    metric='bbox',
    outfile_prefix=
    '/home/jaa/Work/Prog/BSU/Detectors/mmdetection/data/Padded_GroundingDINO.v1i.yolov5pytorch/test/images',
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
train_cfg = dict(max_epochs=24, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_size=10,
    dataset=dict(
        ann_file=
        '/home/jaa/Work/Prog/BSU/Detectors/mmdetection/data/Padded_GroundingDINO.v1i.yolov5pytorch/train_coco.json',
        data_prefix=dict(
            img=
            '/home/jaa/Work/Prog/BSU/Detectors/mmdetection/data/Padded_GroundingDINO.v1i.yolov5pytorch/train/images'
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
    num_workers=6)
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
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=10,
    dataset=dict(
        ann_file=
        '/home/jaa/Work/Prog/BSU/Detectors/mmdetection/data/Padded_GroundingDINO.v1i.yolov5pytorch/valid_coco.json',
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
    num_workers=6)
val_evaluator = dict(
    ann_file=
    '/home/jaa/Work/Prog/BSU/Detectors/mmdetection/data/Padded_GroundingDINO.v1i.yolov5pytorch//valid_coco.json',
    format_only=False,
    metric='bbox',
    type='CocoMetric')
vis_backends = [
    dict(
        init_kwargs=dict(
            group='1stage',
            name='RetinaNet_r50_24e_10b_SGD',
            project='Сomparison of detectors'),
        type='WandbVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='Visualizer',
    vis_backends=[
        dict(
            init_kwargs=dict(
                group='1stage',
                name='RetinaNet_r50_24e_10b_SGD',
                project='Сomparison of detectors'),
            type='WandbVisBackend'),
    ])
weights = 'retinanet_r50_fpn_mstrain_3x_coco_20210718_220633-88476508.pth'
work_dir = '/home/jaa/Work/Prog/BSU/Detectors/work_dirs/RetinaNet/RetinaNet_r50_24e_SGD_custom/'
