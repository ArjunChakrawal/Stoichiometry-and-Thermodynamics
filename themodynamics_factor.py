# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 14:12:08 2022

@author: arch9809
"""



import matplotlib.pyplot as plt
import numpy as np

LC = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]
nosc = np.arange(-3, 3.1, 0.1)
g= 4-nosc
dGox = 60.3-28.5*nosc # kJ/ C mol

z = [4, 2,1,8] 
m = [1,0.33,0.33,0.167]
X=[2,1,1,1]
Ered = [0.82,0.43,0.014,-0.22]
F = 96.485 # kJ/volt 
R = 0.008314
T = 273.15

EA =['O2','N','Fe','SO4'] 
# plt.figure()
# i=3

# dGred = -z[i]*F*Ered[i] 
# dGrxn = dGox + dGred
   
# FT = 1- np.exp((dGrxn + m[i]*50)/(X[i]*R*T))
# plt.plot(nosc, FT)
# plt.xlabel('NOSC')
# plt.ylabel('FT')
# plt.ylim([0,1.2])
 

plt.figure()
for i in range(0,len(z)):    
    
    dGred = -z[i]*F*Ered[i] 
    dGrxn = dGox + dGred       
    FT = 1- np.exp((dGrxn + m[i]*50)/(X[i]*R*T))
    FT2= 1/(1 + np.exp( (dGrxn + F*80*0.001)/(R*T)))
    plt.plot(g, FT,'-', label = EA[i],color=LC[i])
    plt.plot(g, FT2,'--', label = EA[i],color=LC[i])
    plt.xlabel('NOSC')
    plt.ylabel('FT')
    plt.legend()
    plt.ylim([0,1.2])
    
    
    
    
    
    
    
    