#!/usr/bin/env python
import pdb
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from geometry_msgs.msg import Point
import sys
import baxter_interface
from baxter_interface import CHECK_VERSION
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import threading
import math
from baxter_core_msgs.msg import (
    JointCommand,
    EndpointState,
)
from std_msgs.msg import String
import numpy as np
from limb import Limb, BothLimb
import collections
import tf


hand_position = [0.632, -0.577, 0.012]  # pre-defined
elbow_position = [0.617, -0.335, -0.158]  # pre-defined
shd_position = [0.588, -0.121, 0.011]  # pre-defined
shd_offset = [0, 0.12, 0.05]
orientation = [0.694, 0.671, 0.180, -0.192]

class PoseMonitor: 

    def __init__(self):
        self.subscriber = rospy.Subscriber('/pose', String, self.callback)
        self.pose = {  # /kinect2_link
            'r_wri': {'x': 0, 'y': 0, 'z': 0},
            'r_elb': {'x': 0, 'y': 0, 'z': 0},
            'r_shd': {'x': 0, 'y': 0, 'z': 0},
            'l_wri': {'x': 0, 'y': 0, 'z': 0},
            'l_elb': {'x': 0, 'y': 0, 'z': 0},
            'l_shd': {'x': 0, 'y': 0, 'z': 0},
        }
        self.transformed_pose = {  # /base
            'r_wri': {'x': 0, 'y': 0, 'z': 0},
            'r_elb': {'x': 0, 'y': 0, 'z': 0},
            'r_shd': {'x': 0, 'y': 0, 'z': 0},
            'l_wri': {'x': 0, 'y': 0, 'z': 0},
            'l_elb': {'x': 0, 'y': 0, 'z': 0},
            'l_shd': {'x': 0, 'y': 0, 'z': 0},
        }
        self.joints = ['r_wri', 'r_elb', 'r_shd', 'l_wri', 'l_elb', 'l_shd']
        self.tf_base2kinect = self.lookupTransform('/base', '/kinect2_link')

        rospy.loginfo('Pose mornitor initializing')
        rospy.sleep(5)


    def callback(self, data):
        pose = eval(data.data) # {joint:{'x':x, 'y':y, 'z':d} for joint in ['r_wri', 'r_elb', 'r_shd']}
        # if pose['r_wri']['z'] == 0 and pose['r_elb']['z'] == 0 and pose['r_shd']['z'] == 0:  # invalid data
        #     return
        for joint in self.joints:
            if self.pose[joint]['z'] == 0:
                self.pose[joint] = pose[joint]
            else:
                if self.distance(self.pose[joint], pose[joint]) > 0.5:  # may result from noise
                    for i in ['x', 'y', 'z']:
                        self.pose[joint][i] = 0.8 * self.pose[joint][i] + 0.2 * pose[joint][i]
                else:
                    for i in ['x', 'y', 'z']: 
                        self.pose[joint][i] = 0.2 * self.pose[joint][i] + 0.8 * pose[joint][i]  # move smoothly
            self.transformed_pose[joint] = self.transform2base(self.pose[joint])

    def distance(self, joint1, joint2):
        x1, y1, z1 = joint1['x'], joint1['y'], joint1['z']
        x2, y2, z2 = joint2['x'], joint2['y'], joint2['z']
        return math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)

    def lookupTransform(self, source, target):
        tf_listener = tf.TransformListener()
        tf_listener.waitForTransform(source, target, rospy.Time(), rospy.Duration(4.0))

        trans, rot = tf_listener.lookupTransform(source, target, rospy.Time())
        euler = tf.transformations.euler_from_quaternion(rot)

        source_target = tf.transformations.compose_matrix(translate = trans, angles = euler)
        return source_target

    def transform2base(self, point):
        """
        transform from frame kinect to frame base
        """
        point = np.array([point['x'], point['y'], point['z'], 1])
        transformed_point = self.tf_base2kinect.dot(point)
        return {'x': transformed_point[0], 'y': transformed_point[1], 'z': transformed_point[2]}

    def check_valid(self):
        for joint in self.joints:
            if self.transformed_pose[joint]['z'] == 0:
                return False
        return True

def get_pose_msg(pose):
    """ pose = limb.endpoint_pose() """
    p = geometry_msgs.msg.Pose()
    ori = pose['orientation']
    pos = pose['position']
    p.orientation.x = ori.x
    p.orientation.y = ori.y
    p.orientation.z = ori.z
    p.orientation.w = ori.w
    p.position.x = pos.x
    p.position.y = pos.y
    p.position.z = pos.z
    return p

def get_orientation_constraint(link_name='right_gripper', frame_id='base',
    x_toler=0.3, y_toler=0.3, z_toler=0.3, weight=1):
    ocm = moveit_msgs.msg.OrientationConstraint()
    ocm.link_name = link_name
    ocm.header.frame_id = frame_id
    ocm.absolute_x_axis_tolerance = x_toler
    ocm.absolute_y_axis_tolerance = y_toler
    ocm.absolute_z_axis_tolerance = z_toler
    ocm.weight = weight

    ocm.orientation = geometry_msgs.msg.Quaternion(*orientation)
    return ocm


if __name__ == '__main__':
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('dress')
    rospy.loginfo('===Initializing===')

    robot = moveit_commander.RobotCommander()
    move_group = moveit_commander.MoveGroupCommander('right_arm')
    move_group.set_planning_time(10)
    move_group.allow_replanning(True)
    move_group.set_goal_position_tolerance(0.01)
    move_group.set_goal_orientation_tolerance(0.05)
    
    enable = baxter_interface.RobotEnable(CHECK_VERSION)
    enable.enable()

    rospy.loginfo('===Moving to neutral===')
    limb = Limb('right')

    #limb.force_cali_mode = False

    limb.calibrate = True
    limb.move_to_neutral()
    
    pm = PoseMonitor()
    while not pm.check_valid() and not rospy.is_shutdown():
        rospy.loginfo('Invalid pose: {}\nSleep 3s'.format(pm.transformed_pose))
        rospy.sleep(3)
    hand_position = [pm.transformed_pose['r_wri']['x'], pm.transformed_pose['r_wri']['y'], pm.transformed_pose['r_wri']['z']]
    elbow_position = [pm.transformed_pose['r_elb']['x'], pm.transformed_pose['r_elb']['y'], pm.transformed_pose['r_elb']['z']]
    shd_position = [pm.transformed_pose['r_shd']['x']+shd_offset[0], pm.transformed_pose['r_shd']['y']+shd_offset[1], pm.transformed_pose['r_shd']['z']+shd_offset[2]]
    print(hand_position)
    print(elbow_position)
    print(shd_position)
    
    #pose = get_pose_msg(limb.endpoint_pose())
    #pose.position.y -= 0.4
    #pose.orientation = geometry_msgs.msg.Quaternion(*orientation)
    
    hand_pose = geometry_msgs.msg.Pose()
    hand_pose.position = geometry_msgs.msg.Point(*hand_position)
    hand_pose.orientation = geometry_msgs.msg.Quaternion(*orientation)

    elbow_pose = geometry_msgs.msg.Pose()
    elbow_pose.position = geometry_msgs.msg.Point(*elbow_position)
    elbow_pose.orientation = geometry_msgs.msg.Quaternion(*orientation)

    shd_pose = geometry_msgs.msg.Pose()
    shd_pose.position = geometry_msgs.msg.Point(*shd_position)
    shd_pose.orientation = geometry_msgs.msg.Quaternion(*orientation)


    move_group.set_pose_target(hand_pose)
    plan = move_group.plan()
    #pdb.set_trace()
    limb.execute_plan(plan, hand_position, elbow_position, shd_position)
    limb.calibrate = False
    limb.calibrate_force()

    ocm = get_orientation_constraint()
    constraints = moveit_msgs.msg.Constraints()
    constraints.orientation_constraints.append(ocm)
    move_group.set_path_constraints(constraints)
    
    move_group.set_start_state_to_current_state()

    waypoints = [hand_pose, elbow_pose, shd_pose]
    print(waypoints)
    plan, fraction = move_group.compute_cartesian_path(waypoints, 0.01, 0.0) # eef_step = 1cm, jump_threshold is disabled

    #pdb.set_trace()

    limb.start = True
    rospy.loginfo('start')
    #move_group.execute(plan, wait=True)
    print(len(plan.joint_trajectory.points))
    limb.execute_plan(plan, hand_position, elbow_position, shd_position)

    moveit_commander.roscpp_shutdown()
    moveit_commander.os._exit(0)
