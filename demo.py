import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import math
from utils import compute_iou, plot_boxes_on_image, wandhG, load_gt_boxes, compute_regression, decode_output

pos_thresh = 0.5
neg_thresh = 0.1
iou_thresh = 0.5
grid_width = 16  # 网格的长宽都是16，因为从原始图片到 feature map 经历了16倍的缩放
grid_height = 16
image_height = 900
image_width = 900
wnum,hnum=math.floor(image_width/grid_width),math.floor(image_height/grid_height)
image_path = "IMG_1278.jpg"
label_path = "test.txt"
gt_boxes = load_gt_boxes(label_path)  # 把 ground truth boxes 的坐标读取出来
raw_image = cv2.imread(image_path)  # 将图片读取出来 (高，宽，通道数)
image_with_gt_boxes = np.copy(raw_image)  # 复制原始图片
plot_boxes_on_image(image_with_gt_boxes, gt_boxes)  # 将 ground truth boxes 画在图片上
Image.fromarray(image_with_gt_boxes).show()  # 展示画了 ground truth boxes 的图片
## 因为得到的 feature map 的长宽都是原始图片的 1/16，所以这里 wnum=720/16，hnum=960/16。
target_scores = np.zeros(shape=[wnum, hnum, 9, 2]) # 0: background, 1: foreground, ,
target_bboxes = np.zeros(shape=[wnum, hnum, 9, 4]) # t_x, t_y, t_w, t_h
target_masks  = np.zeros(shape=[wnum, hnum, 9]) # negative_samples: -1, positive_samples: 1
################################### ENCODE INPUT #################################
## 将 feature map 分成 wnum*hnum 个小块
for i in range(wnum):
    for j in range(hnum):
        for k in range(9):
            center_x = j * grid_width + grid_width * 0.5  # 计算此小块的中心点横坐标
            center_y = i * grid_height + grid_height * 0.5  # 计算此小块的中心点纵坐标
            xmin = center_x - wandhG[k][0] * 0.5  # wandhG 是预测框的宽度和长度，xmin 是预测框在图上的左上角的横坐标
            ymin = center_y - wandhG[k][1] * 0.5  # ymin 是预测框在图上的左上角的纵坐标
            xmax = center_x + wandhG[k][0] * 0.5  # xmax 是预测框在图上的右下角的纵坐标
            ymax = center_y + wandhG[k][1] * 0.5  # ymax 是预测框在图上的右下角的纵坐标
            # ignore cross-boundary anchors
            if (xmin > -5) & (ymin > -5) & (xmax < (image_width+5)) & (ymax < (image_height+5)):
                anchor_boxes = np.array([xmin, ymin, xmax, ymax])
                anchor_boxes = np.expand_dims(anchor_boxes, axis=0)
                # compute iou between this anchor and all ground-truth boxes in image.
                ious = compute_iou(anchor_boxes, gt_boxes)
                positive_masks = ious > pos_thresh
                negative_masks = ious < neg_thresh

                if np.any(positive_masks):
                    plot_boxes_on_image(image_with_gt_boxes, anchor_boxes, thickness=1)
                    print("=> Encoding positive sample: %d, %d, %d" %(i, j, k))
                    cv2.circle(image_with_gt_boxes, center=(int(0.5*(xmin+xmax)), int(0.5*(ymin+ymax))),
                                    radius=1, color=[255,0,0], thickness=4)  # 正预测框的中心点用红圆表示

                    target_scores[i, j, k, 1] = 1.  # 表示检测到物体
                    target_masks[i, j, k] = 1 # labeled as a positive sample
                    # find out which ground-truth box matches this anchor
                    max_iou_idx = np.argmax(ious)
                    selected_gt_boxes = gt_boxes[max_iou_idx]
                    target_bboxes[i, j, k] = compute_regression(selected_gt_boxes, anchor_boxes[0])

                if np.all(negative_masks):
                    target_scores[i, j, k, 0] = 1.  # 表示是背景
                    target_masks[i, j, k] = -1 # labeled as a negative sample
                    cv2.circle(image_with_gt_boxes, center=(int(0.5*(xmin+xmax)), int(0.5*(ymin+ymax))),
                                    radius=1, color=[0,0,0], thickness=4)  # 负预测框的中心点用黑圆表示
Image.fromarray(image_with_gt_boxes).show()
############################## FASTER DECODE OUTPUT ###############################
faster_decode_image = np.copy(raw_image)
pred_bboxes = np.expand_dims(target_bboxes, 0).astype(np.float32)
pred_scores = np.expand_dims(target_scores, 0).astype(np.float32)
pred_scores, pred_bboxes = decode_output(pred_bboxes, pred_scores)
plot_boxes_on_image(faster_decode_image, pred_bboxes, color=[255, 0, 0]) # red boundig box
Image.fromarray(np.uint8(faster_decode_image)).show()
