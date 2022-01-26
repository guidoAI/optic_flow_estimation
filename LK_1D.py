# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 18:46:15 2022

1D Lucas Kanade

@author: guido
"""

import numpy as np
from matplotlib import pyplot as plt

def fun(x, offset=10, slope = -0.5, a = 1, fun_type = 'sine_slope'):    
    
    if(fun_type == 'sine_slope'):
        y = offset + slope * x + np.sin(a * x)
    elif(fun_type == 'quadratic'):
        y = offset + slope * x + a * x**2
    elif(fun_type == 'linear'):
        y = offset + slope * x
        
    return y
        
def fun_dx(x, offset=10, slope = -0.5, a = 1, fun_type = 'sine_slope'):    
    
    if(fun_type == 'sine_slope'):
        dy_dx = slope + a * np.cos(a * x)
    elif(fun_type == 'quadratic'):
        dy_dx = slope + 2* a * x
    elif(fun_type == 'linear'):
        dy_dx = slope
    
    return dy_dx
    
def fun_dx2(x, offset=10, slope = -0.5, a = 1, fun_type = 'sine_slope'):    
    
    if(fun_type == 'sine_slope'):
         dy_dxdx = -a*a * np.sin(a * x)
    elif(fun_type == 'quadratic'):
         dy_dxdx = 2 * a
    elif(fun_type == 'linear'):
        dy_dxdx = 0
        
    return dy_dxdx

# ground-truth shift:
dx = 0.5

# properties of the function

#fun_type = 'quadratic'
#offset= 10
#slope = 0
#a = 2
#dx = 0.5

#fun_type = 'linear'
#offset= 0
#slope = 0
#a = 0
#dx = 0.5

fun_type = 'sine_slope'
offset= 5
slope = 0 #0.5
a = 5
# ground-truth shift:
dx = 2*np.pi/(a*2) # + 0.1

# range in which we show the function:
max_x = 3
step = 0.01 

# We take a simple 1D function and shift it in the x-axis:
x = np.arange(-max_x, max_x, step)
f = fun(x, offset, slope, a, fun_type)
g = fun(x+dx, offset, slope, a, fun_type)

# Plot the functions
plt.figure()
plt.plot(x, f, x, g)
plt.xticks(np.arange(-max_x, max_x,1))
plt.grid()
plt.legend(['f', 'g'])
plt.show()

plt.pause(1)

x_query = float(input('Where do you want to determine optic flow? x = '))

# Determine f'(x) and apply Eq. 4:
small = 0.01
df_dx = (fun(x_query+small, offset, slope, a, fun_type) - fun(x_query, offset, slope, a, fun_type)) / small
h = (fun(x_query+dx, offset, slope, a, fun_type) - fun(x_query, offset, slope, a, fun_type)) / df_dx

# Output result:
print(f'Lucas and Kanade say that the shift at {x_query} = {h}, while the ground truth is {dx}')
print(f'Approximation df/dx = {df_dx}, real value = {fun_dx(x_query, offset, slope, a, fun_type)}')
print(f'df/dx({x_query}) = {fun_dx(x_query, offset, slope, a, fun_type)}, df/dx2({x_query}) = {fun_dx2(x_query, offset, slope, a, fun_type)}')