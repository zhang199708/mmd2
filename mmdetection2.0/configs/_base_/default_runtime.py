checkpoint_config = dict(interval=5)
# yapf:disable
log_config = dict(
    interval=1139,  # 验证集图片个数
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'cascade_rcnn_r50_coco_pretrained_weights_classes_21.pth'
resume_from = None
workflow = [('train', 1)]
