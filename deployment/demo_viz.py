#!/usr/bin/env python3
import rospy
import numpy as np
import math
import torch
import torch.nn as nn
import cv2
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import threading

class Actor(nn.Module):
    def __init__(self, s, a):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s, 400), nn.ReLU(),
            nn.Linear(400, 300), nn.ReLU(),
            nn.Linear(300, a), nn.Tanh())
    def forward(self, x):
        return self.net(x)

model = Actor(26, 2)
model.load_state_dict(torch.load('/root/best_actor.pth',
    map_location='cpu', weights_only=False))
model.eval()

rospy.init_node('demo_viz')
cmd_pub  = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
scan_obs  = np.ones(24) * 3.5
depth_img = None
lin_vel   = 0.0
ang_vel   = 0.0

def depth_cb(msg):
    global scan_obs, depth_img
    frame = np.frombuffer(msg.data, dtype=np.uint16).reshape(
        msg.height, msg.width).astype(float)
    frame[frame == 0]    = 10000
    frame[frame > 10000] = 10000
    frame = frame / 1000.0
    mid  = msg.height // 2
    roi  = frame[mid-1:mid+2, :]
    sector_width = msg.width // 24
    sectors = []
    for i in range(24):
        val = np.min(roi[:, i*sector_width:(i+1)*sector_width])
        val = np.clip(val, 0.15, 3.5)
        sectors.append(val)
    scan_obs = np.array(sectors)
    disp = np.clip(frame * 255 / 5.0, 0, 255).astype(np.uint8)
    depth_img = cv2.applyColorMap(disp, cv2.COLORMAP_JET)

rospy.Subscriber('/camera/depth/image_rect_raw', Image, depth_cb)
rate = rospy.Rate(10)

def inference_loop():
    global lin_vel, ang_vel
    while not rospy.is_shutdown():
        obs = np.concatenate([
            scan_obs / 3.5,
            [0.5],
            [0.0]
        ]).astype(np.float32)
        with torch.no_grad():
            s = torch.FloatTensor(obs).unsqueeze(0)
            a = model(s).numpy()[0]
        lin_vel = float(np.clip(abs(a[0]) * 0.25, 0.0, 0.25))
        ang_vel = float(np.clip(a[1], -1.0, 1.0))
        twist = Twist()
        twist.linear.x  = lin_vel
        twist.angular.z = ang_vel
        cmd_pub.publish(twist)
        rate.sleep()

t = threading.Thread(target=inference_loop)
t.daemon = True
t.start()

rospy.loginfo("Demo running — press Q to quit")

while not rospy.is_shutdown():
    if depth_img is not None:
        display = depth_img.copy()
        h, w    = display.shape[:2]
        bar_h = 60
        for i in range(24):
            x1    = int(i * w / 24)
            x2    = int((i+1) * w / 24)
            fill  = int((1 - scan_obs[i]/3.5) * bar_h)
            color = (0,255,0) if scan_obs[i] > 0.5 else (0,0,255)
            cv2.rectangle(display, (x1, h-bar_h), (x2, h-bar_h+fill), color, -1)
            cv2.rectangle(display, (x1, h-bar_h), (x2, h), (255,255,255), 1)
        cv2.putText(display, f"Linear:  {lin_vel:.2f} m/s",
            (10,30),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(display, f"Angular: {ang_vel:.2f} rad/s",
            (10,65),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(display, f"Min obstacle: {np.min(scan_obs):.2f} m",
            (10,100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        cv2.putText(display, "DRL Obstacle Avoidance | D415 + Jetson Orin",
            (10, h-bar_h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
        cv2.imshow("Depth Camera + DRL Inference", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()