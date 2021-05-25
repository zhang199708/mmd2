from mmdet.apis import init_detector
from mmdet.apis import inference_detector
from mmdet.apis import show_result_pyplot

# 模型配置文件
config_file = 'configs/cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco.py'

# 预训练模型文件
checkpoint_file = 'checkpoints/cascade_mask_rcnn_r50_fpn_1x_coco_20200203-9d4dcb24.pth'

# 通过模型配置文件与预训练文件构建模型
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# 测试单张图片
img = 'demo/demo.jpg'
result = inference_detector(model, img)
show_result_pyplot(model,img, result)