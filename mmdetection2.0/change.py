def main():
    # gen coco pretrained weight
    import torch
    num_classes = 20
    pretrained_weights = torch.load("checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth")

    # # weight
    # model_coco["state_dict"]["bbox_head.0.fc_cls.weight"] = model_coco["state_dict"]["bbox_head.0.fc_cls.weight"][
    #                                                         :num_classes, :]
    # model_coco["state_dict"]["bbox_head.1.fc_cls.weight"] = model_coco["state_dict"]["bbox_head.1.fc_cls.weight"][
    #                                                         :num_classes, :]
    # model_coco["state_dict"]["bbox_head.2.fc_cls.weight"] = model_coco["state_dict"]["bbox_head.2.fc_cls.weight"][
    #                                                         :num_classes, :]
    # # bias
    # model_coco["state_dict"]["bbox_head.0.fc_cls.bias"] = model_coco["state_dict"]["bbox_head.0.fc_cls.bias"][
    #                                                       :num_classes]
    # model_coco["state_dict"]["bbox_head.1.fc_cls.bias"] = model_coco["state_dict"]["bbox_head.1.fc_cls.bias"][
    #                                                       :num_classes]
    # model_coco["state_dict"]["bbox_head.2.fc_cls.bias"] = model_coco["state_dict"]["bbox_head.2.fc_cls.bias"][
    #                                                       :num_classes]
    pretrained_weights["state_dict"]['roi_head.bbox_head.fc_cls.weight'].resize_(num_classes + 1, 1024)
    pretrained_weights["state_dict"]["roi_head.bbox_head.fc_cls.bias"].resize_(num_classes + 1)
    pretrained_weights["state_dict"]["roi_head.bbox_head.fc_reg.weight"].resize_(num_classes * 4, 1024)
    pretrained_weights["state_dict"]["roi_head.bbox_head.fc_reg.bias"].resize_(num_classes * 4)

    # save new model
    torch.save(pretrained_weights, "faster_rcnn_r50_fpn_1x_%d.pth" % num_classes)


if __name__ == "__main__":
    main()
