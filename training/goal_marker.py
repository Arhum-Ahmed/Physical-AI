#!/usr/bin/env python3
import rospy
from visualization_msgs.msg import Marker

rospy.init_node('goal_marker')
pub = rospy.Publisher('/goal_marker', Marker, queue_size=1)
rate = rospy.Rate(10)

while not rospy.is_shutdown():
    try:
        goal = rospy.get_param('/current_goal')
        m = Marker()
        m.header.frame_id = "odom"
        m.header.stamp = rospy.Time.now()
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = goal[0]
        m.pose.position.y = goal[1]
        m.pose.position.z = 0.1
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.3
        m.color.r = 1.0
        m.color.g = 0.0
        m.color.b = 0.0
        m.color.a = 1.0
        pub.publish(m)
    except: pass
    rate.sleep()