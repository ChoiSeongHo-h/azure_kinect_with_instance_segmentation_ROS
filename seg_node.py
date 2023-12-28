#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import numpy as np
import torch

model = YOLO('yolo-Weights/best_a.pt')

class ImageSubscriber:
    def __init__(self):
        rospy.init_node('image_subscriber', anonymous=True)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/rgb/image_raw', Image, self.image_callback)
        self.clss_seg_pub = rospy.Publisher('/seg/clss_seg', Image, queue_size=10)
        self.instance_seg_pub = rospy.Publisher('/seg/instance_seg', Image, queue_size=10)

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        cv_image_resized = cv2.resize(cv_image, (640, 480))
        print(msg.header.stamp)
        self.pub(cv_image_resized, msg.header.stamp)

    def pub(self, img, stamp):
        model_out = next(model.predict(img, stream=True, verbose=False))
        if len(model_out.boxes) == 0:
            clss_img = np.zeros((480, 640), dtype=np.uint8)
            clss_seg_msg = self.bridge.cv2_to_imgmsg(clss_img, "mono8")
            clss_seg_msg.header.stamp = stamp
            self.clss_seg_pub.publish(clss_seg_msg)
            instance_img = np.zeros((480, 640), dtype=np.uint8)
            instance_seg_msg = self.bridge.cv2_to_imgmsg(instance_img, "mono8")
            instance_seg_msg.header.stamp = stamp
            self.instance_seg_pub.publish(instance_seg_msg)
            return
        
        masks = model_out.masks.data
        boxes = model_out.boxes.data

        clsses = np.array(boxes[:, 5].cpu(), dtype=np.uint8).flatten()
        probs = np.array(boxes[:, 4].cpu(), dtype=np.float64).flatten()
        sorted_idcs = np.argsort(probs)

        clss_mask = torch.zeros(480, 640).cuda()
        for idx in sorted_idcs:
            if probs[idx] < 0.2:
                continue
            clss = clsses[idx] + 1
            local_mask = (masks[idx] != 0)
            clss_mask[local_mask] = masks[idx][local_mask] * clss * 3

        clss_img = np.array(clss_mask.cpu().numpy(), dtype=np.uint8)

        color = 1
        instance_mask = torch.zeros(480, 640).cuda()
        for idx in sorted_idcs:
            if probs[idx] < 0.2:
                continue
            local_mask = (masks[idx] != 0)
            instance_mask[local_mask] = masks[idx][local_mask] * color * 20
            color = color + 1

        instance_img = np.array(instance_mask.cpu().numpy(), dtype=np.uint8)
        _, binary = cv2.threshold(instance_img, 1, 1, cv2.THRESH_BINARY)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        erosion = cv2.erode(binary, k)
        instance_img = instance_img * erosion

        clss_seg_msg = self.bridge.cv2_to_imgmsg(clss_img, "mono8")
        clss_seg_msg.header.stamp = stamp
        self.clss_seg_pub.publish(clss_seg_msg)

        instance_seg_msg = self.bridge.cv2_to_imgmsg(instance_img, "mono8")
        instance_seg_msg.header.stamp = stamp
        self.instance_seg_pub.publish(instance_seg_msg)

        cv2.imshow("plot", cv2.resize(model_out.plot(), (0, 0), fx=2.0 ,fy=2.0, interpolation = cv2.INTER_AREA))
        cv2.waitKey(1)

def main():
    image_subscriber = ImageSubscriber()
    rospy.spin()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
