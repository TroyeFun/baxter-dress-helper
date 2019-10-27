#!/usr/bin/env python
# coding=utf-8

import matplotlib.pyplot as plt
import sys
import os

record = open('force.log', 'r')
idx = 0
mode = sys.argv[1]
while os.path.exists('../record/{}_{}.png'.format(mode, idx)):
    idx += 1
new_record = open('../record/{}_{}.log'.format(mode, idx), 'w')

plt.xlim(0, 80)
plt.ylim(0, 10)
steps = []
forces = []
for line in record.readlines():
    new_record.write(line)
    line = line.strip()
    step = int(line.split(' ')[0])
    force = float(line.split(' ')[1])
    steps.append(step)
    forces.append(force)

plt.title(sys.argv[1])
plt.plot(steps, forces)
plt.savefig('../record/{}_{}.png'.format(mode, idx))
plt.show()
