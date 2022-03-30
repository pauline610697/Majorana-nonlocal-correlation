#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 20:10:19 2019

@author: laiyihua
"""

import numpy as np
import matplotlib.pyplot as plt

VzNumber = 201; # 400
VzStep = 0.005; # 0.0075 or 0.01
VzMin = 2.5;
VzMax = 3.5
Vpoints = 1001;
colorbarL = -0.001;
colorbarU = 0.04;
Gmatrix = []

x = np.linspace(-0.5,0.5,Vpoints)
y = np.linspace(VzMin,VzMax,VzNumber)

rows = VzNumber;
cols = Vpoints;
datafile=open('GT_LR_3.0_QD.txt','r')
for line in datafile:
        row = [];
        y1 = line.split(','); #array: string of numbers
        for j in range(cols):
                row.append(float(y1[j]))
        Gmatrix.append(row)

       
fig, ax = plt.subplots()
im = ax.pcolor(x,y,Gmatrix, cmap='RdBu_r')
#im = ax.pcolor(x,y,result,vmin=0,vmax=2, cmap='RdBu_r')
fig.colorbar(im,ax=ax)
#plt.pcolor(x,y,result)
im.set_clim(colorbarL, colorbarU) # Set the axis scale

plt.xlabel('$V_R(meV)$')
plt.ylabel('$V_z(mV)$')
plt.title('$G_{LR}(e^2/h)$:$\mu=$3 meV, T=0 meV, $V_{Z_c}=3.13$ meV, QD on the left.')
plt.savefig('/Users/laiyihua/Google Drive/UMD/Research/Majorana/Two ABS/CrossG_2D_QD.jpg')
plt.show()