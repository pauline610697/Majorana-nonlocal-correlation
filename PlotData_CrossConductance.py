#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 13:48:33 2018

@author: laiyihua
"""

import numpy as np
import matplotlib.pyplot as plt
#from pylab import *
T = 0;
mu = 3.0;
Vz = 2.8;
V = np.linspace(-0.5,0.5,1001);

#w = np.loadtxt('T_LR_ee_3.0_ZM.txt', delimiter=',',dtype='str');
#w = np.loadtxt('Gr_LL_3.0_ZM.txt', delimiter=',',dtype='str');
#w = ','.join(w);
#w1 = np.fromstring(w, dtype = np.float, sep = ',');

#x = np.loadtxt('T_LR_he_3.0_ZM.txt', delimiter=',',dtype='str');
#x = np.loadtxt('Gr_RR_3.0_ZM.txt', delimiter=',',dtype='str');
#x = ','.join(x);
#x1 = np.fromstring(x, dtype = np.float, sep = ',');

#y = np.loadtxt('T_RL_ee_3.0_ZM.txt', delimiter=',',dtype='str');
#y = np.loadtxt('GT_LR_3.0_Vz='+str(Vz)+'_ZM.txt', delimiter=',',dtype='str');
#y = ','.join(y);
#y1 = np.fromstring(y, dtype = np.float, sep = ',');

y = np.loadtxt('GT_LR_3.0_Vz=2.8_no_ZM.txt', delimiter=',',dtype='str');
y = ','.join(y);
y1 = np.fromstring(y, dtype = np.float, sep = ',');

y = np.loadtxt('GT_LR_3.0_Vz=2.95_no_ZM.txt', delimiter=',',dtype='str');
y = ','.join(y);
y2 = np.fromstring(y, dtype = np.float, sep = ',');

y = np.loadtxt('GT_LR_3.0_Vz=3.13_no_ZM.txt', delimiter=',',dtype='str');
y = ','.join(y);
y3 = np.fromstring(y, dtype = np.float, sep = ',');

y = np.loadtxt('GT_LR_3.0_Vz=3.2_no_ZM.txt', delimiter=',',dtype='str');
y = ','.join(y);
y4 = np.fromstring(y, dtype = np.float, sep = ',');

y = np.loadtxt('GT_LR_3.0_Vz=3.3_no_ZM.txt', delimiter=',',dtype='str');
y = ','.join(y);
y5 = np.fromstring(y, dtype = np.float, sep = ',');

#z = np.loadtxt('T_RL_he_3.0_ZM.txt', delimiter=',',dtype='str');
#z = np.loadtxt('GT_RL_3.0_ZM.txt', delimiter=',',dtype='str');
#z = ','.join(z);
#z1 = np.fromstring(z, dtype = np.float, sep = ',');

plt.figure(dpi = 400)
#plt.plot(V,w1,label='$T_{LR}^{ee}$',color='blue')
#plt.plot(V,x1,label='$T_{LR}^{he}$',color='orange',linestyle=':')
#plt.plot(V,y1,label='$T_{RL}^{ee}$',color='yellow',linestyle='--')
#plt.plot(V,z1,label='$T_{RL}^{he}$',color='magenta',linestyle='-.')
#plt.plot(V,w1,label='$G_{LL}$',color='blue')
#plt.plot(V,x1,label='$G_{RR}$',color='yellow',linestyle='--')
plt.plot(V,y1,label='$V_z=2.8$',color='green',linestyle='--')
plt.plot(V,y2,label='$V_z=2.95$',color='red',linestyle='--')
plt.plot(V,y3,label='$V_z=3.13$',color='blue',linestyle='--')
plt.plot(V,y4,label='$V_z=3.2$',color='magenta',linestyle='--')
plt.plot(V,y5,label='$V_z=3.3$',color='orange',linestyle='--')
##plt.plot(V,z1,label='$G_{RL}$',color='magenta',linestyle='-.')
plt.xlabel('$V_R$ $(meV)$')
plt.ylabel('$G_{LR}(e^2/h)$')
#plt.title('$G(e^2/h)$, $\mu=$' +str(mu)+'meV, $V_Z=$' +str(Vz)+ 'meV, T=' + str(T)+ 'meV, QD on Left.')
plt.title('$\mu=$' +str(mu)+ ' meV, T=' + str(T)+ ' meV, $V_{Z_c}=3.13$ meV, no QD.')
plt.legend()
#plt.savefig('/Users/laiyihua/Google Drive/UMD/Research/Majorana/Two ABS/Conductance plots/without self-energy/Gt_left_L=5um_mu=3.0_Vz=' +str(Vz)+ '_T=' + str(T)+ '_Line.jpg')
#plt.savefig('/Users/laiyihua/Google Drive/UMD/Research/Majorana/Two ABS/G_LR_L=5um_mu=' +str(mu)+ '_Vz=' +str(Vz)+ '_T=' + str(T)+ '_ZM.jpg')
#plt.savefig('/Users/laiyihua/Google Drive/UMD/Research/Majorana/Two ABS/G_LR_Eb=1.0_noQD.jpg')
plt.show()