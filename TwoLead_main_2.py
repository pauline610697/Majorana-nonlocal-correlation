
# coding: utf-8

# In[4]:

from mpi4py import MPI

import numpy as np
import kwant
import scipy.sparse.linalg as sla
import TwoLead_module_2 as Maj
import matplotlib.pyplot as plt


# In[3]:

comm = MPI.COMM_WORLD;
rank = comm.Get_rank();

NS_dict = {'alpha':2.5, 'Delta_0':0.9,'wireLength':500,'t':25.0, 'mu_lead':25.0, 'Nbarrier':2, 'Ebarrier':1.0, 'QD':'no', 'QD2':'no', 'VD':4, 'dotLength':30, 'SE':'no', 'VZC':'yes','Vzc':6.7,'Vz':3.3, 'voltage':0.0, 'varymu':'no','mu':3.0,'lamd':1.5,'gamma':0.01};

# ======== G as a function of Vz ==============

voltageMin = -0.5; voltageMax = 0.5; voltageNumber = 1001;
voltageRange = np.linspace(voltageMin, voltageMax, voltageNumber);

VzMin = 2.5; VzMax = 3.5; VzNumber = 201;
VzRange = np.linspace(VzMin, VzMax, VzNumber);
#VzStep = 0.0075; NS_dict['Vz'] = rank*VzStep;
if NS_dict['VZC']=='yes':
	Delta0=NS_dict['Delta_0']
	NS_dict['Delta_0']=Delta0*np.sqrt(1-(NS_dict['Vz']/NS_dict['Vzc'])**2)
	print(NS_dict['Delta_0'])

#gFile1 = open('T_LR_ee_3.0_ZM.txt','w');
#gFile2 = open('T_LR_he_3.0_ZM.txt','w');
#gFile3 = open('T_RL_ee_3.0_ZM.txt','w');
#gFile4 = open('T_RL_he_3.0_ZM.txt','w');
#gFile5 = open('Gr_LL_3.0_ZM.txt','w');
#gFile6 = open('Gr_RR_3.0_ZM.txt','w');
gFile7 = open('GT_LR_3.0_Vz='+ str(NS_dict['Vz'])+'_no_ZM.txt','w');
gFile8 = open('GT_RL_3.0_Vz='+ str(NS_dict['Vz'])+'_no_ZM.txt','w');
for voltage in voltageRange:
    NS_dict['voltage']=voltage;
    T1,T2,T3,T4,Gr_LL,Gr_RR,GT_LR,GT_RL = Maj.conductance(NS_dict);
    #gFile1.write( str(T1) + ',' );
    #gFile2.write( str(T2) + ',' );
    #gFile3.write( str(T3) + ',' );
    #gFile4.write( str(T4) + ',' );
    #gFile5.write( str(Gr_LL) + ',' );
    #gFile6.write( str(Gr_RR) + ',' );
    gFile7.write( str(GT_LR) + ',' );
    gFile8.write( str(GT_RL) + ',' );
#gFile1.write('\n');
#gFile1.close();
#gFile2.write('\n');
#gFile2.close();
#gFile3.write('\n');
#gFile3.close();
#gFile4.write('\n');
#gFile4.close();
#gFile5.write('\n');
#gFile5.close();
#gFile6.write('\n');
#gFile6.close();
gFile7.write('\n');
gFile7.close();
gFile8.write('\n');
gFile8.close();

# ======== TV as a function of Vz ==============
#VzMin = -1.5; VzMax = 1.5; VzNumber = 3001;
#VzRange = np.linspace(VzMin, VzMax, VzNumber);

#tvFile = open('TV_rank'+ str(rank)+'.txt','w');
#tvFile = open('TV_Fig4(c).txt','w');
#for voltage in VzRange:
#    NS_dict['Vz']=voltage;
#    temp = Maj.TV(NS_dict);
#    if abs(temp.imag) < 10**(-5):
#            tvFile.write( str(temp.real) + ',');
#tvFile.write('\n');
#tvFile.close();

# In[ ]:

# ======== G as a function of mu ==============

#voltageMin = -1.5; voltageMax = 1.5; voltageNumber = 3001;
#voltageRange = np.linspace(voltageMin, voltageMax, voltageNumber);
#
#mu0=2.0; muStep = 0.005; NS_dict['mu'] = mu0 + rank*muStep;
#
#gFile = open('G_Vz'+str(NS_dict['Vz'])+'_L'+str(NS_dict['wireLength'])+'_rank'+ str(rank)+'.txt','w');
#for voltage in voltageRange:
#    NS_dict['voltage']=voltage;
#    gFile.write( str(Maj.conductance(NS_dict)) + ',' );
#gFile.write('\n');
#gFile.close();






