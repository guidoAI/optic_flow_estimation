# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 18:46:15 2022

1D Lucas Kanade

@author: guido
"""

import numpy as np
from matplotlib import pyplot as plt

def fun(x):
    
    return 10 - 0.5 * x + np.sin(x)

# We take a simple 1D function and shift it in the x-axis:
x = np.arange(0, 10, 0.01)
f = fun(x)
dx = 0.5
g = fun(x-dx)

plt.ion()
plt.figure()
plt.plot(x, f, x, g)
plt.show(block=False)
plt.draw()
plt.pause(1)

x_query = float(input('Where do you want to determine optic flow? x = '))

small = 0.01
df_dx = (fun(x_query) - fun(x_query+small)) / small
h = (fun(x_query-dx) - fun(x_query)) / df_dx;

print(f'Lucas and Kanade say that the shift = {h}')