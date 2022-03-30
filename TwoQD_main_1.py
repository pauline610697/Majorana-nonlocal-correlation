from mpi4py import MPI

import numpy as np
import kwant
import scipy.sparse.linalg as sla
import TwoQD_module_2 as Maj
import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD;
rank = comm.Get_rank();

NS_dict = {'alpha':2.5, 'Delta_0':0.9,'wireLength':500,'t':25.0, 'mu_lead':25.0, 'Nbarrier':2, 'Ebarrier':10.0, 'QD':'yes', 'QD2':'no', 'VD':4, 'dotLength':30, 'SE':'no', 'VZC':'yes','Vzc':20,'Vz':3.0, 'voltage':0.0, 'varymu':'no','mu':1.0,'lamd':1.5,'gamma':0.01};

# ======== G as a function of Vz ==============
voltageMin = -1.5; voltageMax = 1.5; voltageNumber = 3001;
voltageRange = np.linspace(voltageMin, voltageMax, voltageNumber);

#VzStep = 0.0075; NS_dict['Vz'] = rank*VzStep;
if NS_dict['VZC']=='yes':
	Delta0=NS_dict['Delta_0']
	NS_dict['Delta_0']=Delta0*np.sqrt(1-(NS_dict['Vz']/NS_dict['Vzc'])**2)
	print(NS_dict['Delta_0'])

gFile = open('GL_rank'+ str(rank)+'.txt','w');
for voltage in voltageRange:
    NS_dict['voltage']=voltage;
    gFile.write( str(Maj.conductance(NS_dict)) + ',' );
gFile.write('\n');
gFile.close();

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
