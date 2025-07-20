
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
#if len(physical_devices)>0:
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
#else:
#  print('missing gpus i guess!')
import sys
sys.path.append('../utils/forr_train/')
from hugrat_utils import set_seeds, get_dataset_rec, printnsay, update_ldict
sys.path.append('../layers/')

##########################################################################

@tf.function
def wrap_pred(model,bx,do_bx,training,bsize):
  pred,z,pred_comp = model(bx,do_bx,training=training,bsize=bsize)#tf.shape(bx)[0])
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

    mask = tf.cast(tf.not_equal(by,args['label_pad']),dtype=tf.int32)
    num = tf.reduce_sum(mask)
    bsizes.append(num.numpy())    
                                
    devdict = update_ldict(devdict,dev_ddict)
  return(bsizes,devdict)
##########################################################################
def multi_check(astr,strlist):
    for strl in strlist:
        if strl in astr:
            return(True)
    return(False)



if __name__=='__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('mdir')
  targs = parser.parse_args()  
  mdir=targs.mdir

  args = json.load(open(mdir+'config.json'))
  args['load_chkpt_dir']= mdir
  args['logfile']='log.log'



  if 'model_type' not in args or args['model_type']=='hirat':
    print('HIRAT MODEL')
    from hirat_model import *    
  elif args['model_type']=='hiratN':
    print('N MODEL')
    from hiratN_model import *
  elif args['model_type']=='comp':
    print('COMP MODEL')
    from hiratcomp_model import *



  thekey = args['traindevpair']['aspect']#get_key(args)

  args,mdict = load_classmodel(args,chkptdir=mdir)

  args['selected_train']='/'.join(args['train_file'].split('/')[:-1])+'/'+'og.train.selected5'
  args['selected_dev']='/'.join(args['train_file'].split('/')[:-1])+'/'+'og.dev.selected5'
  args['counter_train']='/'.join(args['train_file'].split('/')[:-1])+'/'+'og.train.counter5'
  args['counter_dev']='/'.join(args['train_file'].split('/')[:-1])+'/'+'og.dev.counter5'


  # ## stats on train/dev
  res={}

  # ## stats on test
  if args['aspect']=='service':
    args['traindevpair']['test_file']='../data/hotel/service/service.test.sents'
  if args['aspect']=='clean':
    args['traindevpair']['test_file']='../data/hotel/clean/clean.test.sents'    

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

  if args['aspect']=='connect':
    kwords = ['connect','cuts out', 'dropout']
  elif args['aspect']=='vague':
     kwords=['food','dish','flavor','delicious','bland','cook','ingredients']
    
  gottem=[]
  for by,bx,do_bx in data_test.batch(1).take(args['TOTAKE']):      
      pred,z = wrap_pred(mdict['model'],bx,do_bx,training=False,bsize=tf.shape(bx)[0])      
      pred_z = tf.math.argmax(z[0]).numpy()
      xn = bx[0].numpy() ## [0] drops the batch

      pred_zx = xn[pred_z] ## [0] cuz hardcode 1 rationae
      pred_str = mdict['tokenizer'].decode(pred_zx)
      gottem.append(multi_check(pred_str,kwords))



  #print('gottem', gottem)
  res['test_prec']=np.mean(gottem)
  
  print(res)
  ## save results
  json.dump(res,open(targs.mdir+'results_synth_FINISHED4.json','w'),indent=2)

