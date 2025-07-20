
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"
from transformers import BertTokenizer, TFBertModel

import pickle
import time
import json
import random

import numpy as np
import json
import argparse
from collections import defaultdict

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
print('physicial devices', physical_devices)

tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

import sys
sys.path.append('../utils/for_train/')
from hugrat_utils import set_seeds, get_dataset_rec, printnsay, update_ldict
sys.path.append('../layers/')
from hirat_model import *
##########################################################################

@tf.function
def wrap_pred(model,bx,do_bx,training,bsize):
  pred,z = model(bx,do_bx,training=training,bsize=bsize)#tf.shape(bx)[0])
  return(pred,z)
##########################################################################

##########################################################################
def dev_epoch(data_gen):  
  devdict=defaultdict(list)  
  ## evaluation
  bsizes=[]
  egs=[];ees=[];elosses=[];belosses=[];eobjs=[];epks=[];ezsum=[];ezdiff=[];
  for by,bx,do_bx in data_gen.batch(32).take(10000000):    
    bsize = tf.shape(bx)[0]
    #print('dev shapes', np.shape(bx), np.shape(by))
    dev_ddict=cag_wrap(args=args,mdict=mdict,
                         cag=jpraw,
                         x=bx,do_x=do_bx,y=by,train=False,
                         bsize=bsize)  
    #if not args['enc_act']=='no_out':
    #  bsizes.append(np.shape(bx)[0])     
    #else:
    mask = tf.cast(tf.not_equal(by,args['label_pad']),dtype=tf.int32)
    num = tf.reduce_sum(mask)
    bsizes.append(num.numpy())    
                                
    devdict = update_ldict(devdict,dev_ddict)
  return(bsizes,devdict)
##########################################################################
if __name__=='__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('mdir')
  #parser.add_argument('dset')
  targs = parser.parse_args()  
  mdir=targs.mdir

  args = json.load(open(mdir+'config.json'))
  args['load_chkpt_dir']= mdir
  args['logfile']='log.log'

  thekey = args['traindevpair']['aspect']

  args,mdict = load_classmodel(args,chkptdir=mdir)

  ## stats on train/dev
  res={}



  # ## stats on test
  if args['aspect']=='service':
    args['traindevpair']['test_file']='../data/hotel/service/service.test.sents'
  if args['aspect']=='clean':
    args['traindevpair']['test_file']='..//data/hotel/clean/clean.test.sents'    

  if os.path.exists(args['traindevpair']['test_file']):
    testfile = args['traindevpair']['test_file']
  else:
    testfile = args['traindevpair']['test_file'].replace('test.SENTS3','test.sents2')


  data_test = get_test_data(testfile,
                        mdict['tokenizer'],
                      maxlen=args['max_len'],bert_len=args['bert_len'],
                      classpick=-1,rando=0,evenclass=0,
                      pady=args['pady'],
                      label_pad=args['label_pad'])  
  true_z = [];pred_z=[];rand_z=[];
  for by,bx,do_bx in data_test.batch(1).take(args['TOTAKE']):      
      pred,z = wrap_pred(mdict['model'],bx,do_bx,training=False,bsize=tf.shape(bx)[0])
      true_z.append(by.numpy()[0])
      pred_z.append(tf.math.argmax(z[0]).numpy())
      num_sents = tf.reduce_sum(do_bx[0],axis=-1).numpy()
      rand_z.append(random.choice(range(num_sents)))
  gottem = [1 if p in t else 0 for p,t in zip(pred_z,true_z)]
  res['test_prec']=np.mean(gottem)
  
  print(res)
  ## save results
  json.dump(res,open(targs.mdir+'results.json','w'),indent=2)

