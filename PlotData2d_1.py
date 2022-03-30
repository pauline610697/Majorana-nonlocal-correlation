#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 14:09:19 2018

@author: laiyihua
"""

import numpy as np
import matplotlib.pyplot as plt

rankNumber = 1360; # 400
VzStep = 0.0075; # 0.0075 or 0.01
Vpoints = 3001;
colorbarL = 0;
colorbarU = 1.8;
Gmatrix = []

y = np.linspace(-1.5,1.5,Vpoints)
x = np.linspace(0,VzStep*(rankNumber-1),rankNumber)

rows = rankNumber;
cols = Vpoints;
for i in range(rows):
    row = []
    y1 = np.loadtxt('G_T_5.0_rank'+ str(i)+'.txt', delimiter=',',dtype='str')
    for j in range(cols):
        row.append(float(y1[j]))
    Gmatrix.append(row)

# Transpose the G matrix
result = [[0]*rows]*cols;
result = [[Gmatrix[j][i] for j in range(len(Gmatrix))] for i in range(len(Gmatrix[0]))]

fig, ax = plt.subplots()
im = ax.pcolor(x,y,result, cmap='RdBu_r')
#im = ax.pcolor(x,y,result,vmin=0,vmax=2, cmap='RdBu_r')
fig.colorbar(im,ax=ax)
#plt.pcolor(x,y,result)
im.set_clim(colorbarL, colorbarU) # Set the axis scale

plt.xlabel('$V_z(meV)$')
plt.ylabel('$V(mV)$')
plt.axvline(x=1.5, color='cyan', linestyle='--')
plt.axvline(x=5.08, color='black', linestyle='--')
plt.axvline(x=1.9, color='yellow', linestyle='--')
plt.axvline(x=6.0, color='magenta', linestyle='--')
plt.title('$G(e^2/h)$, probed from Right')
plt.savefig('/Users/laiyihua/Google Drive/UMD/Research/Majorana/Two ABS/Paper/Plot/G_right_L=5um_mu=5.0_noTitle.jpg')
plt.show()