# Physical-AI

This repo dives into a small project to explore Physical-AI. For this purpose, a DRL (Deep Reinforcement Learning) model is trained on Gazebo simulator and is deployed on an edge device. This also relates to the idea of zero-shot sim-to-real transfer.

## Hardware
For training a Linux machine with 16 GB RAM, i5 10th Gen, and GTX 1660 Ti was used. For the deployment, a NVIDIA Jetson Orin used as the edge device and Intel Realsense Depth Camera was used as a visual device.

## Software
The setup requires the following:


1. Docker

2. Python

3. ROS (noetic)

4. Gazebo (for training)

## Docker Setup

Both the training/inference and deployment stages use Docker. The setup for them is below.

### Training and Inference
**1. First time setup only**
```bash
xhost +local:docker
sudo docker run -it \
  --name ros_noetic \
  --env="DISPLAY=$DISPLAY" \
  --env="QT_X11_NO_MITSHM=1" \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  --network=host \
  osrf/ros:noetic-desktop-full
```

**2. Inside container install everything**
```bash
apt update && apt install -y \
  ros-noetic-turtlebot3 \
  ros-noetic-turtlebot3-simulations \
  ros-noetic-gazebo-ros-pkgs \
  ros-noetic-gazebo-plugins \
  python3-pip
pip3 install torch stable-baselines3 numpy
echo "export TURTLEBOT3_MODEL=burger" >> ~/.bashrc
source ~/.bashrc
```

### Deployment
**1. First time setup only**
``` bash
sudo docker run -it \
  --name ros_noetic \
  --env="DISPLAY=$DISPLAY" \
  --env="QT_X11_NO_MITSHM=1" \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  --volume="/dev:/dev" \
  --network=host \
  --privileged \
  arm64v8/ros:noetic-ros-base
```

**2. Inside container install everything**
```bash
apt update && apt install -y \
  ros-noetic-realsense2-camera \
  ros-noetic-cv-bridge \
  python3-pip
pip3 install torch numpy opencv-python
source /opt/ros/noetic/setup.bash
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
```

## Training in Gazebo
After the setup has been complete. Follow the steps below to start the docker conatiner, launch Gazebo and train the model.

**1. Start Docker**
```bash
xhost +local:docker
sudo docker start ros_noetic
sudo docker exec -it ros_noetic bash
```


**2. Terminal 1 - Gazebo**
``` bash
source ~/ros/noetic/setup.bash
export TURTLEBOT3_MODEL=burger
roslaunch turtlebot3_gazebo turtlebot3_stage_4.launch
```
**3. Terminal 2 - Training**
```bash
source ~/ros/noetic/setup.bash
python3 ~/train_stage.py
```

**4. Terminal 3 - Goal marker (optional)**
```bash
source ~/ros/noetic/setup.bash
python3 ~/goal_marker.py
```

**5. After training copy model out (Optional)**
``` bash
sudo docker cp ros_noetic:/root/best_actor.pth ~/Desktop/
sudo docker cp ros_noetic:/root/best_critic.pth ~/Desktop/
```

## Infernece in Gazebo

**1. Enter Docker**
```bash
sudo docker start ros_noetic
xhost +local:docker
sudo docker exec -it --env="DISPLAY=$DISPLAY" --env="QT_X11_NO_MITSHM=1" ros_noetic bash
```

**2. Copy model in (run outside Docker)**
```bash
sudo docker cp ~/best_actor.pth ros_noetic:/root/
```

**3. Terminal 1 - Gazebo**
``` bash
source ~/ros/noetic/setup.bash
export TURTLEBOT3_MODEL=burger
roslaunch turtlebot3_gazebo turtlebot3_stage_4.launch
```

**4. Terminal 2 - Demo visualization**
```bash
source ~/.bashrc
python3 /root/run_inference.py
```

## Deployment on Edge Device (NVIDIA Jetson Orin + Depth Camera)

**1. Enter Docker**
```bash
sudo docker start ros_noetic
xhost +local:docker
sudo docker exec -it --env="DISPLAY=$DISPLAY" --env="QT_X11_NO_MITSHM=1" ros_noetic bash
```

**2. Copy model in (run outside Docker)**
```bash
sudo docker cp ~/best_actor.pth ros_noetic:/root/
```
**3. Terminal 1 - RealSense camera**
```bash
source ~/.bashrc
roslaunch realsense2_camera rs_camera.launch
```
**4. Terminal 2 - Demo visualization**
```bash
source ~/.bashrc
python3 /root/run_inference.py
```