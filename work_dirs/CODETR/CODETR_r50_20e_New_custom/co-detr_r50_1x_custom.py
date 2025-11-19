HOME = '/home/jaa/Work/Prog/BSU/Detectors'
MODEL_GROUP = 'transformer'
MODEL_TYPE = 'CODETR'
WEIGHT_SIZE = 'r50'
additions = 'New'
auto_scale_lr = dict(base_batch_size=8, enable=True)
backend_args = None
batch_augments = []
batch_size = 1
checkpoint_config = dict(interval=1)
custom_imports = dict(
    allow_failed_imports=False, imports=[
        'projects.CODETR.codetr',
    ])
data_root = '/home/jaa/Work/Prog/BSU/Detectors/mmdetection/data/Padded_GroundingDINO.v1i.yolov5pytorch/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(
        _scope_='mmdet',
        by_epoch=True,
        interval=1,
        max_keep_ckpts=3,
        type='CheckpointHook'),
    early_stopping=dict(
        min_delta=0.001,
        monitor='coco/bbox_mAP',
        patience=3,
        type='EarlyStoppingHook'),
    logger=dict(
        _scope_='mmdet',
        interval=50,
        log_metric_by_epoch=True,
        type='LoggerHook'),
    param_scheduler=dict(_scope_='mmdet', type='ParamSchedulerHook'),
    sampler_seed=dict(_scope_='mmdet', type='DistSamplerSeedHook'),
    timer=dict(_scope_='mmdet', type='IterTimerHook'),
    visualization=dict(_scope_='mmdet', type='DetVisualizationHook'))
default_scope = 'mmdet'
deterministic = True
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
evaluation = dict(interval=1, metric='bbox')
experiment = 'CODETR_r50_20e_New_custom'
image_size = (
    1280,
    1280,
)
launcher = 'none'
load_from = '/home/jaa/Work/Prog/BSU/Detectors/work_dirs/CODETR/CODETR_r50_20e_New_custom/best_epoch_17.pth'
load_pipeline = [
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
]
log_level = 'INFO'
log_processor = dict(
    _scope_='mmdet', by_epoch=True, type='LogProcessor', window_size=50)
loss_lambda = 2.0
max_epochs = 12
max_iters = 270000
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
        frozen_stages=1,
        init_cfg=dict(checkpoint='torchvision://resnet50', type='Pretrained'),
        norm_cfg=dict(requires_grad=False, type='BN'),
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
    bbox_head=[
        dict(
            anchor_generator=dict(
                octave_base_scale=8,
                ratios=[
                    1.0,
                ],
                scales_per_octave=1,
                strides=[
                    4,
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
                    0.1,
                    0.1,
                    0.2,
                    0.2,
                ],
                type='DeltaXYWHBBoxCoder'),
            feat_channels=256,
            in_channels=256,
            loss_bbox=dict(loss_weight=24.0, type='GIoULoss'),
            loss_centerness=dict(
                loss_weight=12.0, type='CrossEntropyLoss', use_sigmoid=True),
            loss_cls=dict(
                alpha=0.25,
                gamma=2.0,
                loss_weight=12.0,
                type='FocalLoss',
                use_sigmoid=True),
            num_classes=1,
            stacked_convs=1,
            type='CoATSSHead'),
    ],
    data_preprocessor=dict(
        batch_augments=[],
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_mask=True,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    eval_module='detr',
    neck=dict(
        act_cfg=None,
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        kernel_size=1,
        norm_cfg=dict(num_groups=32, type='GN'),
        num_outs=5,
        out_channels=256,
        type='ChannelMapper'),
    query_head=dict(
        as_two_stage=True,
        dn_cfg=dict(
            box_noise_scale=1.0,
            group_cfg=dict(dynamic=True, num_dn_queries=100, num_groups=None),
            label_noise_scale=0.5),
        in_channels=2048,
        loss_bbox=dict(loss_weight=5.0, type='L1Loss'),
        loss_cls=dict(
            beta=2.0,
            loss_weight=1.0,
            type='QualityFocalLoss',
            use_sigmoid=True),
        loss_iou=dict(loss_weight=2.0, type='GIoULoss'),
        num_classes=1,
        num_query=900,
        positional_encoding=dict(
            normalize=True,
            num_feats=128,
            temperature=20,
            type='SinePositionalEncoding'),
        transformer=dict(
            decoder=dict(
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    attn_cfgs=[
                        dict(
                            dropout=0.0,
                            embed_dims=256,
                            num_heads=8,
                            type='MultiheadAttention'),
                        dict(
                            dropout=0.0,
                            embed_dims=256,
                            num_levels=5,
                            type='MultiScaleDeformableAttention'),
                    ],
                    feedforward_channels=2048,
                    ffn_dropout=0.0,
                    operation_order=(
                        'self_attn',
                        'norm',
                        'cross_attn',
                        'norm',
                        'ffn',
                        'norm',
                    ),
                    type='DetrTransformerDecoderLayer'),
                type='DinoTransformerDecoder'),
            encoder=dict(
                num_layers=6,
                transformerlayers=dict(
                    attn_cfgs=dict(
                        dropout=0.0,
                        embed_dims=256,
                        num_levels=5,
                        type='MultiScaleDeformableAttention'),
                    feedforward_channels=2048,
                    ffn_dropout=0.0,
                    operation_order=(
                        'self_attn',
                        'norm',
                        'ffn',
                        'norm',
                    ),
                    type='BaseTransformerLayer'),
                type='DetrTransformerEncoder',
                with_cp=4),
            num_co_heads=2,
            num_feature_levels=5,
            type='CoDinoTransformer',
            with_coord_feat=False),
        type='CoDINOHead'),
    roi_head=[
        dict(
            bbox_head=dict(
                bbox_coder=dict(
                    target_means=[
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    target_stds=[
                        0.1,
                        0.1,
                        0.2,
                        0.2,
                    ],
                    type='DeltaXYWHBBoxCoder'),
                fc_out_channels=1024,
                in_channels=256,
                loss_bbox=dict(loss_weight=120.0, type='GIoULoss'),
                loss_cls=dict(
                    loss_weight=12.0,
                    type='CrossEntropyLoss',
                    use_sigmoid=False),
                num_classes=1,
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                roi_feat_size=7,
                type='Shared2FCBBoxHead'),
            bbox_roi_extractor=dict(
                featmap_strides=[
                    4,
                    8,
                    16,
                    32,
                    64,
                ],
                finest_scale=56,
                out_channels=256,
                roi_layer=dict(
                    output_size=7, sampling_ratio=0, type='RoIAlign'),
                type='SingleRoIExtractor'),
            type='CoStandardRoIHead'),
    ],
    rpn_head=dict(
        anchor_generator=dict(
            octave_base_scale=4,
            ratios=[
                0.5,
                1.0,
                2.0,
            ],
            scales_per_octave=3,
            strides=[
                4,
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
        loss_bbox=dict(loss_weight=12.0, type='L1Loss'),
        loss_cls=dict(
            loss_weight=12.0, type='CrossEntropyLoss', use_sigmoid=True),
        type='RPNHead'),
    test_cfg=[
        dict(max_per_img=300, nms=dict(iou_threshold=0.8, type='soft_nms')),
        dict(
            rcnn=dict(
                max_per_img=100,
                nms=dict(iou_threshold=0.5, type='nms'),
                score_thr=0.0),
            rpn=dict(
                max_per_img=1000,
                min_bbox_size=0,
                nms=dict(iou_threshold=0.7, type='nms'),
                nms_pre=1000)),
        dict(
            max_per_img=100,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.6, type='nms'),
            nms_pre=1000,
            score_thr=0.0),
    ],
    train_cfg=[
        dict(
            assigner=dict(
                match_costs=[
                    dict(type='FocalLossCost', weight=2.0),
                    dict(box_format='xywh', type='BBoxL1Cost', weight=5.0),
                    dict(iou_mode='giou', type='IoUCost', weight=2.0),
                ],
                type='HungarianAssigner')),
        dict(
            rcnn=dict(
                assigner=dict(
                    ignore_iof_thr=-1,
                    match_low_quality=False,
                    min_pos_iou=0.5,
                    neg_iou_thr=0.5,
                    pos_iou_thr=0.5,
                    type='MaxIoUAssigner'),
                debug=False,
                pos_weight=-1,
                sampler=dict(
                    add_gt_as_proposals=True,
                    neg_pos_ub=-1,
                    num=512,
                    pos_fraction=0.25,
                    type='RandomSampler')),
            rpn=dict(
                allowed_border=-1,
                assigner=dict(
                    ignore_iof_thr=-1,
                    match_low_quality=True,
                    min_pos_iou=0.3,
                    neg_iou_thr=0.3,
                    pos_iou_thr=0.7,
                    type='MaxIoUAssigner'),
                debug=False,
                pos_weight=-1,
                sampler=dict(
                    add_gt_as_proposals=False,
                    neg_pos_ub=-1,
                    num=256,
                    pos_fraction=0.5,
                    type='RandomSampler')),
            rpn_proposal=dict(
                max_per_img=1000,
                min_bbox_size=0,
                nms=dict(iou_threshold=0.7, type='nms'),
                nms_pre=4000)),
        dict(
            allowed_border=-1,
            assigner=dict(topk=9, type='ATSSAssigner'),
            debug=False,
            pos_weight=-1),
    ],
    type='CoDETR',
    use_lsj=True)
model_backbone = dict(frozen_stages=1, norm_eval=True)
model_bbox_head = dict(bbox_head=[
    dict(
        anchor_generator=dict(
            octave_base_scale=8,
            ratios=[
                1.0,
            ],
            scales_per_octave=1,
            strides=[
                4,
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
                0.1,
                0.1,
                0.2,
                0.2,
            ],
            type='DeltaXYWHBBoxCoder'),
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(loss_weight=24.0, type='GIoULoss'),
        loss_centerness=dict(
            loss_weight=12.0, type='CrossEntropyLoss', use_sigmoid=True),
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=12.0,
            type='FocalLoss',
            use_sigmoid=True),
        num_classes=1,
        stacked_convs=1,
        type='CoATSSHead'),
])
model_roi_head = dict()
num_classes = 1
num_dec_layer = 6
num_epochs = 20
num_workers = 6
optim_wrapper = dict(
    clip_grad=dict(max_norm=0.1, norm_type=2),
    optimizer=dict(lr=2.5e-05, type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(custom_keys=dict(backbone=dict(lr_mult=0.1))),
    type='OptimWrapper')
param_scheduler = [
    dict(begin=0, by_epoch=False, end=500, start_factor=0.05, type='LinearLR'),
    dict(
        T_max=12,
        begin=0,
        by_epoch=True,
        end=20,
        eta_min=1e-06,
        type='CosineAnnealingLR'),
]
resume = True
seed = 11
test_cfg = dict(_scope_='mmdet', type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        _scope_='mmdet',
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
    sampler=dict(_scope_='mmdet', shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    _scope_='mmdet',
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
train_cfg = dict(max_epochs=20, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_size=1,
    dataset=dict(
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
            ],
            type='CocoDataset'),
        pipeline=[
            dict(type='PackDetInputs'),
        ],
        type='MultiImageMixDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(_scope_='mmdet', shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='PackDetInputs'),
]
val_cfg = dict(_scope_='mmdet', type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        _scope_='mmdet',
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
    sampler=dict(_scope_='mmdet', shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    _scope_='mmdet',
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
            name='CODETR_r50_20e_1b_New',
            project='Сomparison of detectors'),
        type='WandbVisBackend'),
]
visualizer = dict(
    _scope_='mmdet',
    name='visualizer',
    type='Visualizer',
    vis_backends=[
        dict(
            init_kwargs=dict(
                group='transformer',
                name='CODETR_r50_20e_1b_New',
                project='Сomparison of detectors'),
            type='WandbVisBackend'),
    ])
weights = 'co_dino_5scale_lsj_r50_3x_coco-fe5a6829.pth'
work_dir = '/home/jaa/Work/Prog/BSU/Detectors/work_dirs/CODETR/CODETR_r50_20e_New_custom/'
