from transformers import set_seed as set_transformers_seed
import tensorflow as tf
import numpy as np
import os
import random
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.absolute())+'/../../layers/')
#############################################

def printnsay(thefile,text):
  print(text)
  with open(thefile,'a') as f:
    f.write(text+'\n')
def justsay(thefile,text):
  with open(thefile,'a') as f:
    f.write(text+'\n')
def set_seeds(theseed):
  np.random.seed(theseed)
  tf.random.set_seed(theseed)  
  random.seed(theseed)
  os.environ['PYTHONHASHSEED']=str(theseed)
  set_transformers_seed(theseed)

#######################################
def update_ldict(ldict,ddict):
  for k in ddict:
    ldict[k].append(ddict[k])
  return(ldict)
