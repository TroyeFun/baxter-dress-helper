#!/usr/bin/env python
import rospy
import pdb
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

position_offset = {
    'r_wri': [0, -0.03, 0],
    'r_elb': [0, -0.03, 0],
    'r_shd': [0, -0.03, 0.1],
    'r_shd2': [0.05, 0.2, 0.05],
    'l_wri': [0, 0.03, 0],
    'l_elb': [0, 0.03, 0],
    'l_shd': [0, 0.03, 0.1],
    'l_shd2': [0.05, -0.2, 0.05],
}

r_orientation = [0.694, 0.671, 0.180, -0.192]
l_orientation = [0.694, -0.671, 0.180, 0.192]
l_shd_orientation = [-0.6, 0.6, -0.3, -0.35]
r_shd_orientation = [0.6, 0.6, 0.3, -0.35]

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
        for i in range(3):
            rospy.loginfo('Start after {} seconds'.format(3-i))
            rospy.sleep(1)


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

def pose2msg(pose):
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

def list2msg(pose_list):
    """
    pose_list = [x, y, z, qx, qy, qz, qw]
    """
    p = geometry_msgs.msg.Pose()
    p.position.x = pose_list[0]
    p.position.y = pose_list[1]
    p.position.z = pose_list[2]
    p.orientation.x = pose_list[3]
    p.orientation.y = pose_list[4]
    p.orientation.z = pose_list[5]
    p.orientation.w = pose_list[6]
    return p

def get_orientation_constraint(link_name, frame_id='base',
    x_toler=0.6, y_toler=0.6, z_toler=0.6, weight=1):
    assert link_name in ['right_gripper', 'left_gripper']

    ocm = moveit_msgs.msg.OrientationConstraint()
    ocm.link_name = link_name
    ocm.header.frame_id = frame_id
    ocm.absolute_x_axis_tolerance = x_toler
    ocm.absolute_y_axis_tolerance = y_toler
    ocm.absolute_z_axis_tolerance = z_toler
    ocm.weight = weight

    if link_name == 'right_gripper':
        ocm.orientation = geometry_msgs.msg.Quaternion(*r_orientation)
    elif link_name == 'left_gripper':
        ocm.orientation = geometry_msgs.msg.Quaternion(*l_orientation)

    return ocm


if __name__ == '__main__':
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('dress')
    rospy.loginfo('===Initializing===')

    robot = moveit_commander.RobotCommander()
    r_move_group = moveit_commander.MoveGroupCommander('right_arm')
    r_move_group.set_planning_time(20)
    r_move_group.allow_replanning(True)
    r_move_group.set_goal_position_tolerance(0.01)
    r_move_group.set_goal_orientation_tolerance(0.05)

    l_move_group = moveit_commander.MoveGroupCommander('left_arm')
    l_move_group.set_planning_time(20)
    l_move_group.allow_replanning(True)
    l_move_group.set_goal_position_tolerance(0.01)
    l_move_group.set_goal_orientation_tolerance(0.05)
    
    enable = baxter_interface.RobotEnable(CHECK_VERSION)
    enable.enable()

    rospy.loginfo('===Moving to neutral===')
    both_limb = BothLimb()
    
    rospy.loginfo('===Checking pose monitor===')
    pm = PoseMonitor()
    while not pm.check_valid() and not rospy.is_shutdown():
        rospy.loginfo('Invalid pose: {}\nSleep 3s'.format(pm.transformed_pose))
        rospy.sleep(3)
        
    rospy.loginfo('===Moving to hand===')
    r_hand_position = [pm.transformed_pose['r_wri']['x']+position_offset['r_wri'][0], pm.transformed_pose['r_wri']['y']+position_offset['r_wri'][1], pm.transformed_pose['r_wri']['z']+position_offset['r_wri'][2]]
    l_hand_position = [pm.transformed_pose['l_wri']['x']+position_offset['l_wri'][0], pm.transformed_pose['l_wri']['y']+position_offset['l_wri'][1], pm.transformed_pose['l_wri']['z']+position_offset['l_wri'][2]]

    r_hand_pose = list2msg(r_hand_position + r_orientation)
    l_hand_pose = list2msg(l_hand_position + l_orientation)
    print('right: \n', r_hand_pose)
    print('left: \n', l_hand_pose)

    #both_limb.r_limb.move_to_neutral()
    print(1)
    r_move_group.set_pose_target(r_hand_pose)
    r_plan = r_move_group.plan()
    print(2)
    both_limb.r_limb.execute_plan(r_plan)
    print(3)
     
    #both_limb.l_limb.move_to_neutral()
    print(4)
    l_move_group.set_pose_target(l_hand_pose)
    l_plan = l_move_group.plan()
    print(5)
    both_limb.l_limb.execute_plan(l_plan)
    print(6)


    #pdb.set_trace()

    rospy.loginfo('===Add moveit constraint===')
    r_ocm = get_orientation_constraint('right_gripper')
    r_constraints = moveit_msgs.msg.Constraints()
    r_constraints.orientation_constraints.append(r_ocm)
    r_move_group.set_path_constraints(r_constraints)

    l_ocm = get_orientation_constraint('left_gripper')
    l_constraints = moveit_msgs.msg.Constraints()
    l_constraints.orientation_constraints.append(l_ocm)
    l_move_group.set_path_constraints(l_constraints)
    
    l_move_group.set_start_state_to_current_state()
    r_move_group.set_start_state_to_current_state()

    rospy.loginfo('===Planning===')
    poses = {}
    for joint in pm.transformed_pose:
        print(joint, pm.transformed_pose[joint])
        if joint.startswith('l'):
            if joint == 'l_shd':
                poses[joint] = [pm.transformed_pose[joint]['x']+position_offset[joint][0], pm.transformed_pose[joint]['y']+position_offset[joint][1], pm.transformed_pose[joint]['z']+position_offset[joint][2]] + l_shd_orientation
            else:
                poses[joint] = [pm.transformed_pose[joint]['x']+position_offset[joint][0], pm.transformed_pose[joint]['y']+position_offset[joint][1], pm.transformed_pose[joint]['z']+position_offset[joint][2]] + l_orientation
            poses[joint] = list2msg(poses[joint])
        elif joint.startswith('r'):
            if joint == 'r_shd':
                poses[joint] = [pm.transformed_pose[joint]['x']+position_offset[joint][0], pm.transformed_pose[joint]['y']+position_offset[joint][1], pm.transformed_pose[joint]['z']+position_offset[joint][2]] + r_shd_orientation
            else:
                poses[joint] = [pm.transformed_pose[joint]['x']+position_offset[joint][0], pm.transformed_pose[joint]['y']+position_offset[joint][1], pm.transformed_pose[joint]['z']+position_offset[joint][2]] + r_orientation
            poses[joint] = list2msg(poses[joint])

    poses['r_shd2'] =  [pm.transformed_pose['r_shd']['x']+position_offset['r_shd2'][0], pm.transformed_pose['r_shd']['y']+position_offset['r_shd2'][1], pm.transformed_pose['r_shd']['z']+position_offset['r_shd2'][2]] + r_shd_orientation
    poses['r_shd2'] = list2msg(poses['r_shd2'])
    poses['l_shd2'] =  [pm.transformed_pose['l_shd']['x']+position_offset['l_shd2'][0], pm.transformed_pose['l_shd']['y']+position_offset['l_shd2'][1], pm.transformed_pose['l_shd']['z']+position_offset['l_shd2'][2]] + l_shd_orientation
    poses['l_shd2'] = list2msg(poses['l_shd2'])

    r_waypoints = [poses['r_wri'], poses['r_elb'], poses['r_shd'], poses['r_shd2']]
    l_waypoints = [poses['l_wri'], poses['l_elb'], poses['l_shd'], poses['l_shd2']]
    r_plan, fraction = r_move_group.compute_cartesian_path(r_waypoints, 0.01, 0.0) # eef_step = 1cm, jump_threshold is disabled
    l_plan, fraction = l_move_group.compute_cartesian_path(l_waypoints, 0.01, 0,0)

    #pdb.set_trace()

    rospy.loginfo('===Executing Plan===')

    rospy.loginfo('start')
    print(len(r_plan.joint_trajectory.points), len(l_plan.joint_trajectory.points))

    both_limb.execute_plan(l_plan, r_plan)

    moveit_commander.roscpp_shutdown()
    moveit_commander.os._exit(0)
