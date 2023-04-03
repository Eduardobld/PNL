#!/usr/bin/env python
# coding: utf-8

# # Método Cuasi-Newton

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize


# In[3]:


#Primero definimos nuestra funcion f.


# In[4]:


def f(xy):
    x, y = xy
    return np.exp(x**2)*y**2 + 2*np.exp(y**2)*x**2 + 4*x*y + 2*x**2 + 4*x - 2*y


# In[5]:


#Calculamos las derivadas de nuestra función.


# In[6]:


def df(xy):
    x, y = xy
    df_dx = 2*x*np.exp(x**2)*y**2 + 4*np.exp(y**2)*x + 4*y + 4*x + 4
    df_dy = 2*y*np.exp(x**2)*x**2 + 4*np.exp(y**2)*y + 4*x - 2
    return np.array([df_dx, df_dy])


# In[7]:


#Definimos nuestro punto semilla.


# In[8]:


x0 = np.array([25, -30])


# In[9]:


#Aplicamos nuestro método de CUASI-NEWTON.


# In[10]:


res = minimize(f, x0, method='BFGS', jac=df, tol=0.00001)
print(res)


# In[11]:


#Ahora para graficar.


# In[12]:


x_vals = np.linspace(-5, 5, 100)
y_vals = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = f([X, Y])

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.scatter(res.x[0], res.x[1], f(res.x), c='r', s=100, marker='o')

plt.show()


# In[ ]:




