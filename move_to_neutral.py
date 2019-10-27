#!/usr/bin/env python
# coding=utf-8
import rospy
import baxter_interface

rospy.init_node('neutral')
l_limb = baxter_interface.Limb('left')
r_limb = baxter_interface.Limb('right')
l_joint_angles = {'left_w0': -1.0, 
                  'left_w1': 0.5, 
                  'left_w2': -0.1, 
                  'left_e0': -0.4, 
                  'left_e1': 2.4, 
                  'left_s0': -0.1, 
                  'left_s1': -0.7}
r_joint_angles = {'right_s0': 0.1, 
                  'right_s1': -0.7, 
                  'right_w0': 1.0, 
                  'right_w1': 0.5, 
                  'right_w2': 0.1, 
                  'right_e0': 0.4, 
                  'right_e1': 2.4}
l_limb.move_to_joint_positions(l_joint_angles)
r_limb.move_to_joint_positions(r_joint_angles)

