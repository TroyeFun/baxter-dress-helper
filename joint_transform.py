#!/usr/bin/env python
# coding=utf-8
from math import pi, cos, sin
import numpy as np

class JointTransform(object):

    def __init__(self):
        self.xyz = [[0.055695, 0, 0.011038], # s0
              [0.069, 0, 0.27035], # s1
              [0.102, 0, 0], # e0
              [0.069, 0, 0.26242], # e1
              [0.10359, 0, 0], # w0
              [0.01, 0, 0.2707], # w1
              [0.115975, 0, 0], # w2
              [0, 0, 0.11355]] # hand

        self.rpy = [[0, 0, 0], # s0
              [-pi/2, 0, 0], # s1
              [pi/2, 0, pi/2], # e0
              [-pi/2, -pi/2, 0], # e1
              [pi/2, 0, pi/2], # w0
              [-pi/2, -pi/2, 0], # w1
              [pi/2, 0, pi/2], # w2
              [0, 0, 0]] # hand

    def transform(self, i, theta):
        """
        transform matrix from frame i-1 to i, i=0 is s0
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

    def deviate(self, i, theta):
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
        for i in range(7):
            T.append(self.transform(i, joint_angles[joints[i]]))
            D.append(self.deviate(i, joint_angles[joints[i]]))
        T.append(self.transform(7, 0))

        G = {}
        for i, key in enumerate(joints):
            trans = np.eye(4)
            for j in range(i):
                trans = trans.dot(T[j])
            trans = trans.dot(D[i])
            for j in range(i+1, 7):
                trans = trans.dot(T[j])
            trans = trans.dot(T[7])
            trans = trans[:, 3]
            grad = -2 * force.dot(trans)
            G[joints[i]] = grad
        return G

    def transition(self, joint_angles):
        prefix = joint_angles.keys()[0].split('_')[0] # 'left' or 'right'
        joints = ['s0', 's1', 'e0', 'e1', 'w0', 'w1', 'w2']
        joints = [prefix + '_' + joint for joint in joints]
        trans = np.eye(4)
        for i, key in enumerate(joints):
            trans = trans.dot(self.transform(i, joint_angles[key]))
        trans = trans.dot(self.transform(7, 0))
        return trans
