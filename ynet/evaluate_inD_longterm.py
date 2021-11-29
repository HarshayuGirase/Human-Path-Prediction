#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import yaml
import argparse
import torch
from model import YNet


# In[3]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# #### Some hyperparameters and settings

# In[4]:


CONFIG_FILE_PATH = 'config/inD_longterm.yaml'  # yaml config file containing all the hyperparameters
DATASET_NAME = 'ind'

TEST_DATA_PATH = 'data/inD/test.pkl'
TEST_IMAGE_PATH = 'data/inD/test'
OBS_LEN = 5  # in timesteps
PRED_LEN = 30  # in timesteps
NUM_GOALS = 20  # K_e
NUM_TRAJ = 1  # K_a

ROUNDS = 3  # Y-net is stochastic. How often to evaluate the whole dataset
BATCH_SIZE = 8


# #### Load config file and print hyperparameters

# In[5]:


with open(CONFIG_FILE_PATH) as file:
    params = yaml.load(file, Loader=yaml.FullLoader)
experiment_name = CONFIG_FILE_PATH.split('.yaml')[0].split('config/')[1]
params


# #### Load preprocessed Data

# In[6]:


df_test = pd.read_pickle(TEST_DATA_PATH)


# In[7]:


df_test.head()


# #### Initiate model and load pretrained weights

# In[8]:


model = YNet(obs_len=OBS_LEN, pred_len=PRED_LEN, params=params)


# In[9]:


model.load(f'pretrained_models/{experiment_name}_weights.pt')


# #### Evaluate model

# In[10]:


model.evaluate(df_test, params, image_path=TEST_IMAGE_PATH,
               batch_size=BATCH_SIZE, rounds=ROUNDS, 
               num_goals=NUM_GOALS, num_traj=NUM_TRAJ, device=None, dataset_name=DATASET_NAME)


# In[1]:


get_ipython().system('nvidia-smi')


# In[ ]:





# In[ ]:





# In[ ]:




