HOME = '/home/jaa/Work/Prog/BSU/Detectors'
MODEL_GROUP = 'transformer'
MODEL_TYPE = 'RTDETR'
WEIGHT_SIZE = 'r50'
additions = 'New'
auto_scale_lr = dict(base_batch_size=5, enable=False)
backend_args = None
batch_size = 5
checkpoint_config = dict(interval=1)
custom_hooks = [
    dict(type='CustomFreezeHook', unfreeze_epoch=5),
]
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'mmdet.models.detectors.rtdetr',
        'custom_freeze_hook',
    ])
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
experiment = 'RTDETR_r50_36e_5b_New_custom'
image_size = (
    1280,
    1280,
)
interpolations = [
    'nearest',
    'bilinear',
    'bicubic',
    'area',
    'lanczos',
]
launcher = 'none'
load_from = '/home/jaa/Work/Prog/BSU/Detectors/work_dirs/RTDETR/RTDETR_r50_36e_5b_New_custom/best_epoch_32.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
max_epochs = 72
metainfo = dict(
    classes=('cow', ), palette=[
        (
            220,
            20,
            60,
        ),
    ])
model = dict(
    as_two_stage=True,
    backbone=dict(
        depth=50,
        frozen_stages=0,
        init_cfg=dict(
            checkpoint=
            'https://github.com/flytocc/mmdetection/releases/download/model_zoo/resnet50vd_ssld_v2_pretrained_d037e232.pth',
            type='Pretrained'),
        norm_cfg=dict(requires_grad=False, type='SyncBN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            1,
            2,
            3,
        ),
        style='pytorch',
        type='ResNetV1d'),
    bbox_head=dict(
        loss_bbox=dict(loss_weight=5.0, type='L1Loss'),
        loss_cls=dict(
            alpha=0.75,
            gamma=2.0,
            iou_weighted=True,
            loss_weight=1.0,
            type='RTDETRVarifocalLoss',
            use_sigmoid=True),
        loss_iou=dict(loss_weight=2.0, type='GIoULoss'),
        num_classes=1,
        sync_cls_avg_factor=True,
        type='RTDETRHead'),
    data_preprocessor=dict(
        batch_augments=[],
        bgr_to_rgb=True,
        mean=[
            0,
            0,
            0,
        ],
        pad_size_divisor=1,
        std=[
            255,
            255,
            255,
        ],
        type='DetDataPreprocessor'),
    decoder=dict(
        layer_cfg=dict(
            cross_attn_cfg=dict(dropout=0.0, embed_dims=256, num_levels=3),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=1024, ffn_drop=0.0),
            self_attn_cfg=dict(dropout=0.0, embed_dims=256, num_heads=8)),
        num_layers=6,
        post_norm_cfg=None,
        return_intermediate=True),
    dn_cfg=dict(
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True, num_dn_queries=100, num_groups=None),
        label_noise_scale=0.5),
    encoder=dict(
        fpn_cfg=dict(
            expansion=1.0,
            in_channels=[
                256,
                256,
                256,
            ],
            norm_cfg=dict(requires_grad=True, type='SyncBN'),
            out_channels=256,
            type='RTDETRFPN'),
        in_channels=[
            256,
            256,
            256,
        ],
        layer_cfg=dict(
            ffn_cfg=dict(
                act_cfg=dict(type='GELU'),
                embed_dims=256,
                feedforward_channels=1024,
                ffn_drop=0.0),
            self_attn_cfg=dict(dropout=0.0, embed_dims=256, num_heads=8)),
        num_encoder_layers=1,
        use_encoder_idx=[
            2,
        ]),
    neck=dict(
        act_cfg=None,
        in_channels=[
            512,
            1024,
            2048,
        ],
        kernel_size=1,
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_outs=3,
        out_channels=256,
        type='ChannelMapper'),
    num_queries=300,
    test_cfg=dict(max_per_img=300),
    train_cfg=dict(
        assigner=dict(
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(box_format='xywh', type='BBoxL1Cost', weight=5.0),
                dict(iou_mode='giou', type='IoUCost', weight=2.0),
            ],
            type='HungarianAssigner')),
    type='RTDETR',
    with_box_refine=True)
num_classes = 1
num_epochs = 36
num_workers = 6
optim_wrapper = dict(
    accumulative_counts=3,
    clip_grad=dict(max_norm=0.1, norm_type=2),
    optimizer=dict(lr=1e-06, type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(
        bypass_duplicate=True,
        custom_keys=dict(backbone=dict(lr_mult=0.1)),
        norm_decay_mult=0),
    type='OptimWrapper')
param_scheduler = [
    dict(begin=0, by_epoch=True, end=2, start_factor=1e-05, type='LinearLR'),
    dict(
        T_max=27,
        begin=9,
        by_epoch=True,
        end=36,
        eta_min=0.0,
        type='CosineAnnealingLR'),
]
pretrained = '/home/jaa/Work/Prog/BSU/Detectors/models/RTDETR/resnet50vd_ssld_v2_pretrained_d037e232.pth'
resume = False
seed = 654
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=5,
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
train_cfg = dict(max_epochs=36, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=5,
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
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
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
    type='CocoMetric')
vis_backends = [
    dict(
        init_kwargs=dict(
            group='transformer',
            name='RTDETR_r50_36e_b5_New',
            project='Сomparison of detectors'),
        type='WandbVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='Visualizer',
    vis_backends=[
        dict(
            init_kwargs=dict(
                group='transformer',
                name='RTDETR_r50_36e_b5_New',
                project='Сomparison of detectors'),
            type='WandbVisBackend'),
    ])
weights = 'rtdetr_r50vd_8xb2-72e_coco_ad2bdcfe.pth'
work_dir = '/home/jaa/Work/Prog/BSU/Detectors/work_dirs/RTDETR/RTDETR_r50_36e_5b_New_custom/'
