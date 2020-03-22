#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os

root = "/Users/bourymbodj/Projects/nmp_qc/nmp_qc/data//qm9/dsgdb9nsd/"
file_list = os.listdir(root)

num_mols = len(file_list)
print(num_mols)


# In[6]:



def read_xyz(file_name):
    with open(file_name, 'rb') as file:
        num_atoms= (int)(file.readline()) 
       # n=";".join(num_atoms)
        #num_atoms=(n.split(";"))
        properties = file.readline().split()[5:17] # only take the properties used in the experiments 
        properties = [num.replace(b'*^', b'e') for num in properties] 
        properties = [float(prop) for prop in properties]
        atom_types = [0]*num_atoms
        coords = np.array(np.zeros([num_atoms,3]))
        for na in range(num_atoms):
            coord_line = file.readline().split()
            atom_types[na] = coord_line[0]
            xyz_coords = coord_line[1:4]
            xyz_coords = [num.replace(b'*^', b'e') for num in xyz_coords] 
            coords[na,:] = 0#[float(num) for num in xyz_coords]  
        vib_freqs = file.readline()
        smiles = file.readline().split()[0]
        inchis = file.readline()
        
    return smiles, properties, atom_types, coords


# In[7]:


import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed


# In[8]:


import multiprocessing as mp
import numpy as np
from multiprocessing.pool import ThreadPool
import time
num=0

start = time.time()
N= mp.cpu_count()
files = os.scandir(root)
print (N)
with mp.pool.ThreadPool(processes = 8) as p:
        results= p.map(read_xyz, [root+file.name for file in files])
      
end = time.time()
print(end - start)


# In[9]:


def column(matrix, i):
    return [row[i] for row in matrix]

smiles =column(results,0)
properties =column(results,1)


# In[10]:


print (smiles)


# In[11]:


import pandas as pd


# In[31]:


df1 = pd.DataFrame(properties)


# In[32]:


df1.head()


# In[33]:


s= [smile.decode() for smile in smiles]


# In[34]:


df = pd.DataFrame(s)


# In[35]:


df.head()


# In[42]:


result=pd.concat([df,df1],axis=1)
result.head()


# In[43]:


result.to_csv('QM9_Smiles_.csv', index=False)


# In[ ]:




