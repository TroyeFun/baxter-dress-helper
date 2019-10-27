#!/usr/bin/env python
# coding=utf-8
import sys
import os
import matplotlib.pyplot as plt
root = sys.argv[1]
modes = ['no-Force', 'Force_thrd1.6_2']

for mode in modes:
    idx = 0
    cnt = []
    forces = []

    while os.path.exists(os.path.join(root, mode, '{}_{}.log'.format(mode, idx))):
        logfile = open(os.path.join(root, mode, '{}_{}.log'.format(mode, idx)), 'r')
        for line in logfile.readlines():
            line = line.strip()
            idx = int(line.split(' ')[0])
            force = float(line.split(' ')[1])
            if len(cnt) < idx + 1:
                cnt.append(1)
                forces.append(force)
            else:
                cnt[idx] += 1
                forces[idx] += force
        idx += 1

    for idx in range(len(cnt)):
        forces[idx] /= cnt[idx]

    plt.xlim(0, 80)
    plt.ylim(0, 10)
    step = [i for i in range(len(cnt))]
    plt.plot(step, forces, label=mode)
    print('{} average force: {}'.format(mode, (sum(forces)/len(forces))))
save_name = modes[0] + '_' + modes[1] + '_average.png'
plt.savefig(os.path.join(root, 'average', save_name))
plt.show()

