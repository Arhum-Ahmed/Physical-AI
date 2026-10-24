#!/usr/bin/env python3
import rospy
import numpy as np
import math
import torch
import torch.nn as nn
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

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

rospy.init_node('drl_inference')
cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
scan_obs = np.ones(24) * 3.5
pos      = [0.0, 0.0]
yaw      = 0.0

GOALS = [[1.5,1.5],[-1.5,1.5],[1.5,-1.5],[-1.5,-1.5],[0.0,1.8],[1.8,0.0]]
goal_idx = 0
goal = np.array(GOALS[goal_idx])

def scan_cb(msg):
    global scan_obs
    r = np.array(msg.ranges)
    r = np.clip(r, 0.0, 3.5)
    r = np.nan_to_num(r, nan=3.5)
    idx = np.linspace(0, len(r)-1, 24, dtype=int)
    scan_obs = r[idx]

def odom_cb(msg):
    global pos, yaw
    pos = [msg.pose.pose.position.x, msg.pose.pose.position.y]
    q   = msg.pose.pose.orientation
    yaw = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))

rospy.Subscriber('/scan', LaserScan, scan_cb)
rospy.Subscriber('/odom', Odometry,  odom_cb)
rate = rospy.Rate(10)
rospy.loginfo("Inference node running...")

while not rospy.is_shutdown():
    global goal_idx, goal
    gv    = goal - np.array(pos)
    gdist = float(np.linalg.norm(gv))
    gang  = float(math.atan2(gv[1], gv[0]) - yaw)
    gang  = math.atan2(math.sin(gang), math.cos(gang))
    if gdist < 0.4:
        goal_idx = (goal_idx + 1) % len(GOALS)
        goal = np.array(GOALS[goal_idx])
        rospy.loginfo(f"Goal reached! New goal: {goal}")
    obs = np.concatenate([
        scan_obs / 3.5,
        [min(gdist/4.0, 1.0)],
        [gang/math.pi]
    ]).astype(np.float32)
    with torch.no_grad():
        s = torch.FloatTensor(obs).unsqueeze(0)
        a = model(s).numpy()[0]
    twist = Twist()
    twist.linear.x  = float(np.clip(abs(a[0]) * 0.25, 0.0, 0.25))
    twist.angular.z = float(np.clip(a[1], -1.0, 1.0))
    cmd_pub.publish(twist)
    rospy.loginfo(f"Dist:{gdist:.2f} lin:{twist.linear.x:.2f} ang:{twist.angular.z:.2f}")
    rate.sleep()