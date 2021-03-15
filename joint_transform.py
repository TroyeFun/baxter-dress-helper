#!/usr/bin/env python
# coding=utf-8
from math import pi, cos, sin
import numpy as np

class JointTransform(object):
    # for right arm only

    def __init__(self):
        self.xyz = [
              [0.024645, -0.219645, 0.118588], # right_torso_arm_mount
              [0.055695, 0, 0.011038], # s0
              [0.069, 0, 0.27035], # s1
              [0.102, 0, 0], # e0
              [0.069, 0, 0.26242], # e1
              [0.10359, 0, 0], # w0
              [0.01, 0, 0.2707], # w1
              [0.115975, 0, 0], # w2
              [0, 0, 0.11355], # hand
              [0, 0, 0.025], # right_gripper_base
              [0, 0, 0.1327], # right_endpoint
        ]

        self.rpy = [
              [0, 0, -0.7854], # right_torso_arm_mount
              [0, 0, 0], # s0
              [-pi/2, 0, 0], # s1
              [pi/2, 0, pi/2], # e0
              [-pi/2, -pi/2, 0], # e1
              [pi/2, 0, pi/2], # w0
              [-pi/2, -pi/2, 0], # w1
              [pi/2, 0, pi/2], # w2
              [0, 0, 0], # hand
              [0, 0, 0],  # right_gripper_base
              [0, 0, 0],  # right_endpoint
        ]

    def transform_single_link(self, i, theta):
        """
        transform matrix from frame i-1 to i, i=0 is arm_mount
        """
        thift = np.array([[1, 0, 0, self.xyz[i][0]],
                         [0, 1, 0, self.xyz[i][1]],
                         [0, 0, 1, self.xyz[i][2]],
                         [0, 0, 0, 1]])
        r, p, y = self.rpy[i]
        Ty = np.array([[cos(y), -sin(y), 0, 0],
                      [sin(y), cos(y), 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        Tp = np.array([[cos(p), 0, sin(p), 0],
                      [0, 1, 0, 0],
                      [-sin(p), 0, cos(p), 0],
                      [0, 0, 0, 1]])
        Tr = np.array([[1, 0, 0, 0],
                      [0, cos(r), -sin(r), 0],
                      [0, sin(r), cos(r), 0],
                      [0, 0, 0, 1]])
        rotate = np.array([[cos(theta), -sin(theta), 0, 0],
                          [sin(theta), cos(theta), 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])
        return thift.dot(Ty.dot(Tp.dot(Tr.dot(rotate))))

    def deviate_single_link(self, i, theta):
        """
        deviation of theta_i
        """
        thift = np.array([[1, 0, 0, self.xyz[i][0]],
                         [0, 1, 0, self.xyz[i][1]],
                         [0, 0, 1, self.xyz[i][2]],
                         [0, 0, 0, 1]])
        r, p, y = self.rpy[i]
        Ty = np.array([[cos(y), -sin(y), 0, 0],
                      [sin(y), cos(y), 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        Tp = np.array([[cos(p), 0, sin(p), 0],
                      [0, 1, 0, 0],
                      [-sin(p), 0, cos(p), 0],
                      [0, 0, 0, 1]])
        Tr = np.array([[1, 0, 0, 0],
                      [0, cos(r), -sin(r), 0],
                      [0, sin(r), cos(r), 0],
                      [0, 0, 0, 1]])
        rotate = np.array([[-sin(theta), -cos(theta), 0, 0],
                          [cos(theta), -sin(theta), 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])
        return thift.dot(Ty.dot(Tp.dot(Tr.dot(rotate))))


    #print(transform(1, pi/2))

    def gradient(self, force, joint_angles):
        """
        param:
        force: 3-d numpy vector
        joint_angles: {'right_s0': s0, 'right_s1': s1, 'right_e0': e0, 'right_e1': e1, 
            'right_w0': w0, 'right_w1': w1, 'right_w2': w2}
        return:
        gradient = {'s0': ds0, 's1', ds1, ..., 'w2', dw2}
        gradient(theta_i) = -2*force.T* d(transform)/d(theta_i)
        """
        prefix = joint_angles.keys()[0].split('_')[0] # 'left' or 'right'
        joints = ['s0', 's1', 'e0', 'e1', 'w0', 'w1', 'w2']
        joints = [prefix + '_' + joint for joint in joints]
        T = []
        D = []
        T.append(self.transform_single_link(0, 0))
        D.append(self.deviate_single_link(0, 0))
        for i in range(1, 8):
            T.append(self.transform_single_link(i, joint_angles[joints[i]]))
            D.append(self.deviate_single_link(i, joint_angles[joints[i]]))
        T.append(self.transform_single_link(8, 0))
        T.append(self.transform_single_link(9, 0))

        G = {}
        for i, key in enumerate(joints):
            trans = np.eye(4)
            for j in range(i+1):
                trans = trans.dot(T[j])
            trans = trans.dot(D[i+1])
            for j in range(i+2, 10):
                trans = trans.dot(T[j])
            trans = trans[:, 3]
            grad = -2 * force.dot(trans)
            G[joints[i]] = grad
        return G

    def transition(self, joint_angles):
        prefix = joint_angles.keys()[0].split('_')[0] # 'left' or 'right'
        joints = ['s0', 's1', 'e0', 'e1', 'w0', 'w1', 'w2']
        joints = [prefix + '_' + joint for joint in joints]
        trans = np.eye(4)
    
        positions = {}

        trans = self.transform_single_link(0, 0)
        positions['right_torso_arm_mount'] = trans[0:3, 3]
        for i, key in enumerate(joints):
            trans = trans.dot(self.transform_single_link(i+1, joint_angles[key]))
            positions[key] = trans[0:3, 3]
        trans = trans.dot(self.transform_single_link(8, 0))
        positions['right_hand'] = trans[0:3, 3]
        trans = trans.dot(self.transform_single_link(9, 0))
        positions['right_gripper_base'] = trans[0:3, 3]
        trans = trans.dot(self.transform_single_link(10, 0))
        positions['right_endpoint'] = trans[0:3, 3]

        #return trans
        return trans, positions


if __name__ == '__main__':
    # import rospy 
    # import baxter_interface
    # rospy.init_node('test')
    # limb = baxter_interface.Limb('right')

    # ja = limb.joint_angles()
    # ep = limb.endpoint_pose()
    ja = {'right_s0': -0.0011346619239320788, 'right_s1': -0.541417568261469, 'right_w0': -0.0006294630012320113, 'right_w1': 1.253691393219679, 'right_w2': 0.00022029739740325738, 'right_e0': 3.163334146893959e-05, 'right_e1': 0.7493426419938896}

    jf = JointTransform()
    tr, pos = jf.transition(ja)
    import ipdb; ipdb.set_trace()

"""
endpoint_pose:
{'position': Point(x=0.6453143590444091, y=-0.8419434911410008, z=0.05668012652021581), 'orientation': Quaternion(x=0.38265681087481757, y=0.9222779296619428, z=-0.02117122420080287, w=0.05028881401896333)}
joint_angles:
{'right_s0': -0.0011346619239320788, 'right_s1': -0.541417568261469, 'right_w0': -0.0006294630012320113, 'right_w1': 1.253691393219679, 'right_w2': 0.00022029739740325738, 'right_e0': 3.163334146893959e-05, 'right_e1': 0.7493426419938896}
right arm mount:
0.024645; -0.21964; 0.11859
0; 0; -0.38268; 0.92388
right_s0:
0.064027; -0.25903; 0.12963
0; 0; -0.38321; 0.92366
right_s1:
0.11276; -0.30787; 0.39998
-0.7018; 0.086445; -0.43576; 0.55688
right_e0:
0.1745; -0.36975; 0.45254
0.18865; 0.45468; -0.33355; 0.804
right_e1:
0.35846; -0.55412; 0.52865
-0.62148; 0.33728; -0.20172; 0.67773
right_w0:
0.43005; -0.62588; 0.50726
0.2974; 0.71747; -0.24155; 0.58175
right_w1:
0.61567; -0.81192; 0.4416
-0.30561; 0.63788; 0.23415; 0.66699
right_w2:
0.62455; -0.82091; 0.32632
0.38266; 0.92228; -0.021171; 0.050289
right_hand:
0.63324; -0.82972; 0.21344
0.38266; 0.92228; -0.021171; 0.050289
right_gripper:
0.64531; -0.84194; 0.05668
0.38266; 0.92228; -0.021171; 0.050289
"""
