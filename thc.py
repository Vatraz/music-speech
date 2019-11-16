# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 20:27:29 2018

@author: user
"""

from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.pyplot as pltT
import numpy as np
from scipy.io import wavfile
import matplotlib.colors as colors
import math as math


import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters



def extremes(t,f,Sx, hashesPerSecond, neighborhoodSize):
    
    data_max = filters.maximum_filter(Sx, neighborhoodSize)
    maxima = (Sxx == data_max)

    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)

    numHashes = int(round(hashesPerSecond*( max(t)-min(t))))
    
    hashes = []
    x, y = [], []
    for dy,dx in slices:
        x_center = (dx.start + dx.stop - 1)/2
        y_center = (dy.start + dy.stop - 1)/2  
        
        value = Sx[dy.start,dx.start]
        x_center=x_center*(max(t)-min(t))/len(t)+min(t)
        x.append(x_center)
        
        y_center = y_center*(max(f)-min(f))/len(f)
        y.append(y_center)
        hashes.append([x_center,y_center,value])
    
        hashes = sorted(hashes, key=lambda x: -x[2])
        hashes = list(filter(lambda x: x[1]>=250 and
               x[1]<=15000,
               hashes))
    return hashes[:numHashes]


fs, data = wavfile.read('yazoo.wav')

size = len(data)/fs

#for step in range(0,round(size)+1,5):

    
 
t_start = 45
t_long = 5
                 
x = data[t_start*fs:(t_start*fs+t_long*fs),1]
                        
f, t, Sxx = signal.spectrogram(x, fs)

t = t+t_start

"""pltT.plot(x,'r-')
pltT.ylabel('Wartość próbki')
pltT.xlabel('Number próbki')
pltT.title('"Shake It Off" - Taylor Swift\nPrzedział t[s] = 45 - 50')
pltT.Show()"""

plt.figure(num=None, figsize=(8, 6), dpi=80)
plt.pcolormesh(t, f, Sxx,
               norm=colors.LogNorm(vmin=Sxx.min(), vmax=Sxx.max()))
plt.ylabel('Częstotliwość [Hz]')
plt.xlabel('Czas [s]')


hashes = extremes(t,f,Sxx,75,8)
matches = []

for item in hashes:
    matchedItems = list(filter(lambda x: (x[0]-item[0] <= 0.15)and
                               (x[0]>item[0]) and
                               abs(math.log10(x[1]/item[1]))<=0.05946,hashes))
    for match in matchedItems:
        ar1 = np.asarray(item)
        ar2 = np.asarray(match)
        sumArr = np.concatenate((ar1,ar2),axis=0)
        matches.append(sumArr)
        

    
lines = np.asarray(matches)
hashesArray = np.asarray(hashes)

plt.plot(hashesArray[:,0],hashesArray[:,1], 'r+')


plt.plot([lines[:,0],lines[:,3]],[lines[:,1],lines[:,4]], 'k-')

plt.show()


"""hashesDT = np.round(lines[:,3]-lines[:,0],2)
hashesDF = np.round(np.log10(lines[:,4]/lines[:,1])/0.05946,2)
hashesTstart = lines[:,0]
hashesF = np.round(np.log10(lines[:,1]),1)
with open('D:/Machine Learning/WinPython/F - YouTube Cover.txt', 'a') as the_file:
    for i in range(0,len(hashesDT)-1):
         the_file.write(str(hashesTstart[i])+'\t'+str(hashesDT[i])+'\t'+str(hashesF[i])+'\t'+str(hashesDF[i])+'\n')"""
