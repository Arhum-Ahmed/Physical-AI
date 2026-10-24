#!/usr/bin/env python3
import rospy
import numpy as np
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState

NUM_SECTORS  = 24
MAX_DEPTH    = 3.5
MIN_DEPTH    = 0.15
MAX_STEPS    = 300_000
MAX_EP_STEPS = 500
BATCH_SIZE   = 256
WARMUP       = 3_000
SAVE_EVERY   = 10_000
SAVE_PATH    = "/root/stage4_td3"

SPAWN_POSITIONS = [
    [ 0.0,  0.0], [ 0.5,  0.5], [-0.5,  0.5],
    [ 0.5, -0.5], [-0.5, -0.5], [ 0.0,  0.8],
    [ 0.8,  0.0], [-0.8,  0.0], [ 0.0, -0.8],
]

GOAL_POSITIONS = [
    [ 1.5,  1.5], [-1.5,  1.5], [ 1.5, -1.5],
    [-1.5, -1.5], [ 0.0,  1.8], [ 1.8,  0.0],
    [-1.8,  0.0], [ 0.0, -1.8], [ 1.0,  1.0],
    [-1.0,  1.0], [ 1.0, -1.0], [-1.0, -1.0],
]

class Actor(nn.Module):
    def __init__(self, s, a):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s, 400), nn.ReLU(),
            nn.Linear(400, 300), nn.ReLU(),
            nn.Linear(300, a), nn.Tanh())
    def forward(self, x):
        return self.net(x)

class Critic(nn.Module):
    def __init__(self, s, a):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(s+a, 400), nn.ReLU(),
            nn.Linear(400, 300), nn.ReLU(),
            nn.Linear(300, 1))
        self.q2 = nn.Sequential(
            nn.Linear(s+a, 400), nn.ReLU(),
            nn.Linear(400, 300), nn.ReLU(),
            nn.Linear(300, 1))
    def forward(self, s, a):
        sa = torch.cat([s, a], 1)
        return self.q1(sa), self.q2(sa)
    def q1_only(self, s, a):
        return self.q1(torch.cat([s, a], 1))

class ReplayBuffer:
    def __init__(self, size=200_000):
        self.buf, self.ptr, self.size = [], 0, size
    def add(self, s, a, r, s2, d):
        if len(self.buf) < self.size: self.buf.append(None)
        self.buf[self.ptr] = (s, a, r, s2, d)
        self.ptr = (self.ptr + 1) % self.size
    def sample(self, n):
        idx = np.random.randint(0, len(self.buf), n)
        s,a,r,s2,d = zip(*[self.buf[i] for i in idx])
        return (torch.FloatTensor(np.array(s)),
                torch.FloatTensor(np.array(a)),
                torch.FloatTensor(np.array(r)).reshape(-1,1),
                torch.FloatTensor(np.array(s2)),
                torch.FloatTensor(np.array(d)).reshape(-1,1))
    def __len__(self): return len(self.buf)

class TD3Agent:
    def __init__(self, sd, ad):
        self.actor   = Actor(sd, ad)
        self.actor_t = Actor(sd, ad)
        self.actor_t.load_state_dict(self.actor.state_dict())
        self.critic   = Critic(sd, ad)
        self.critic_t = Critic(sd, ad)
        self.critic_t.load_state_dict(self.critic.state_dict())
        self.a_opt = torch.optim.Adam(self.actor.parameters(),  3e-4)
        self.c_opt = torch.optim.Adam(self.critic.parameters(), 3e-4)
        self.it = 0

    def act(self, s, noise=0.1):
        s = torch.FloatTensor(s).unsqueeze(0)
        a = self.actor(s).detach().numpy()[0]
        return np.clip(a + np.random.normal(0, noise, a.shape), -1, 1)

    def train(self, buf):
        if len(buf) < BATCH_SIZE: return
        self.it += 1
        s,a,r,s2,d = buf.sample(BATCH_SIZE)
        with torch.no_grad():
            n  = (torch.randn_like(a)*0.2).clamp(-0.5, 0.5)
            a2 = (self.actor_t(s2)+n).clamp(-1, 1)
            q1,q2 = self.critic_t(s2, a2)
            qt = r + 0.99*(1-d)*torch.min(q1, q2)
        q1,q2 = self.critic(s, a)
        cl = F.mse_loss(q1, qt) + F.mse_loss(q2, qt)
        self.c_opt.zero_grad(); cl.backward(); self.c_opt.step()
        if self.it % 2 == 0:
            al = -self.critic.q1_only(s, self.actor(s)).mean()
            self.a_opt.zero_grad(); al.backward(); self.a_opt.step()
            for p,pt in zip(self.critic.parameters(), self.critic_t.parameters()):
                pt.data.copy_(0.005*p + 0.995*pt)
            for p,pt in zip(self.actor.parameters(), self.actor_t.parameters()):
                pt.data.copy_(0.005*p + 0.995*pt)

    def save(self, path):
        torch.save(self.actor.state_dict(),  f"{path}_actor.pth")
        torch.save(self.critic.state_dict(), f"{path}_critic.pth")
        print(f"Saved: {path}")

    def load(self, path):
        self.actor.load_state_dict(torch.load(f"{path}_actor.pth", map_location="cpu"))
        self.actor_t.load_state_dict(self.actor.state_dict())
        self.critic.load_state_dict(torch.load(f"{path}_critic.pth", map_location="cpu"))
        self.critic_t.load_state_dict(self.critic.state_dict())
        print("Resumed from checkpoint!")

class Stage4Env:
    def __init__(self):
        rospy.init_node('td3_stage4', anonymous=True)
        self.cmd_pub   = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.scan_sub  = rospy.Subscriber('/scan', LaserScan, self._scan_cb)
        self.odom_sub  = rospy.Subscriber('/odom', Odometry,  self._odom_cb)
        self.unpause   = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause     = rospy.ServiceProxy('/gazebo/pause_physics',   Empty)
        self.set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.scan_obs  = np.ones(NUM_SECTORS) * MAX_DEPTH
        self.pos       = [0.0, 0.0]
        self.yaw       = 0.0
        self.collision = False
        self.goal      = np.array([1.5, 1.5])
        self.obs_size  = NUM_SECTORS + 2
        self.act_size  = 2
        self.goals_reached = 0
        rospy.sleep(2.0)
        print("Stage4 env ready")

    def _scan_cb(self, msg):
        r = np.array(msg.ranges)
        r = np.clip(r, MIN_DEPTH, MAX_DEPTH)
        r = np.nan_to_num(r, nan=MAX_DEPTH, posinf=MAX_DEPTH)
        idx = np.linspace(0, len(r)-1, NUM_SECTORS, dtype=int)
        self.scan_obs  = r[idx]
        self.collision = bool(np.min(self.scan_obs) < 0.2)

    def _odom_cb(self, msg):
        self.pos = [msg.pose.pose.position.x,
                    msg.pose.pose.position.y]
        q = msg.pose.pose.orientation
        self.yaw = math.atan2(
            2*(q.w*q.z + q.x*q.y),
            1 - 2*(q.y*q.y + q.z*q.z))

    def _get_obs(self):
        gv    = self.goal - np.array(self.pos)
        gdist = float(np.linalg.norm(gv))
        gang  = float(math.atan2(gv[1], gv[0]) - self.yaw)
        gang  = math.atan2(math.sin(gang), math.cos(gang))
        return np.concatenate([
            self.scan_obs / MAX_DEPTH,
            [min(gdist / 4.0, 1.0)],
            [gang / math.pi]
        ]).astype(np.float32)

    def _new_goal(self):
        spawn = np.array(self.pos)
        goal  = spawn.copy()
        while np.linalg.norm(goal - spawn) < 1.5:
            goal = np.array(GOAL_POSITIONS[np.random.randint(len(GOAL_POSITIONS))])
        self.goal = goal
        rospy.set_param('/current_goal', self.goal.tolist())

    def reset(self):
        self.unpause()
        sp = SPAWN_POSITIONS[np.random.randint(len(SPAWN_POSITIONS))]
        state = ModelState()
        state.model_name = 'turtlebot3_burger'
        state.pose.position.x = sp[0]
        state.pose.position.y = sp[1]
        state.pose.position.z = 0.0
        yaw = np.random.uniform(-math.pi, math.pi)
        state.pose.orientation.z = math.sin(yaw/2)
        state.pose.orientation.w = math.cos(yaw/2)
        try: self.set_state(state)
        except: pass
        self._new_goal()
        self.collision = False
        time.sleep(0.3)
        self.pause()
        return self._get_obs()

    def step(self, action):
        self.unpause()
        t = Twist()
        t.linear.x  = float(np.clip((action[0]+1)/2 * 0.25, 0.0, 0.25))
        t.angular.z = float(np.clip(action[1], -1.0, 1.0))
        self.cmd_pub.publish(t)
        time.sleep(0.1)
        self.pause()
        obs      = self._get_obs()
        gdist    = float(np.linalg.norm(self.goal - np.array(self.pos)))
        min_obs  = float(np.min(self.scan_obs))
        if self.collision:
            return obs, -100.0, True, {}
        if gdist < 0.3:
            self.goals_reached += 1
            print(f"Goal reached! Total: {self.goals_reached}")
            self._new_goal()
            return self._get_obs(), 100.0, False, {}
        reward  =  10.0 * (1.0 - obs[NUM_SECTORS])
        reward +=  t.linear.x * 2.0
        reward -=  abs(t.angular.z) * 0.5
        if min_obs < 0.5:
            reward -= (0.5 - min_obs) * 20.0
        if min_obs < 0.3:
            reward -= (0.3 - min_obs) * 50.0
        return obs, reward, False, {}

    def stop(self):
        self.cmd_pub.publish(Twist())

env    = Stage4Env()
agent  = TD3Agent(env.obs_size, env.act_size)
replay = ReplayBuffer()

import os
if os.path.exists("/root/best_actor.pth"):
    agent.load("/root/best_actor.pth")

best  = -np.inf
total = 0
ep    = 0

print(f"Training on Stage4 | obs={env.obs_size} act={env.act_size}")

while total < MAX_STEPS:
    s   = env.reset()
    ep += 1
    er  = 0
    for _ in range(MAX_EP_STEPS):
        a = np.random.uniform(-1, 1, 2) if total < WARMUP else agent.act(s)
        s2, r, done, _ = env.step(a)
        replay.add(s, a, r, s2, float(done))
        agent.train(replay)
        er += r; total += 1; s = s2
        if total % SAVE_EVERY == 0:
            agent.save(SAVE_PATH)
            print(f"[{total}] Checkpoint saved")
        if done: break
    print(f"Ep {ep} | Steps {total}/{MAX_STEPS} | Reward {er:.1f} | Goals {env.goals_reached} | Buf {len(replay)}")
    if er > best:
        best = er
        agent.save(SAVE_PATH + "_best")

env.stop()
print(f"Training complete! Goals: {env.goals_reached}")