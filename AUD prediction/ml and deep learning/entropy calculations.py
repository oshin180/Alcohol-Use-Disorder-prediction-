import numpy as np
import pandas as pd

targets = pd.read_excel('F:/AUD files for LSTM/trying entropy.xlsx')
x=targets.iloc[:,0].values

L=np.array(x)
m=2
r=(0.2 * L.std(ddof=0))

N = len(L)
B = 0.0
A = 0.0

# Split time series and save all templates of length m
xmi = np.array([L[i : i + m] for i in range(N - m)])
xmj = np.array([L[i : i + m] for i in range(N - m + 1)])

# Save all matches minus the self-match, compute B
B = np.sum([np.sum(np.abs(xmii - xmj).max(axis=1) <= r) - 1 for xmii in xmi])

# Similar for computing A
m += 1
xm = np.array([L[i : i + m] for i in range(N - m + 1)])

A = np.sum([np.sum(np.abs(xmi - xm).max(axis=1) <= r) - 1 for xmi in xm])
sample_entropy=-np.log(A/B)


import matplotlib.pyplot as plt
targets.hist()
    






























    
       
       
