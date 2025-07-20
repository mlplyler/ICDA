
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
sys.path.append('./for_train/')
sys.path.append('../util/')
from parse_utils import *
sys.path.append('../layers/')
##########################################################################
def make_counterdata(sentss,zss,yss,softpredss,rats0s,rats1s):
  len2r0=defaultdict(list)
  len2r1=defaultdict(list)
  for i,r in enumerate(rats0s):
      len2r0[len(r)].append(r)
  for i,r in enumerate(rats1s):
      len2r1[len(r)].append(r)       
  dumps=[]    
  for s,z,y,p in zip(sentss,zss,yss,softpredss):
      if y==1:
          if len(rats0s)>0:
            newrat=random.choice(len2r0[len(z)])

          else:
              raise NotImplementedError('no rats0')
      else:
          if len(rats1s)>0:
            newrat=random.choice(len2r1[len(z)])
          else:
              raise NotImplementedError('no rats1')
      s2 = list(s)      
      for ii,zi in enumerate(z):
          s2[zi]=newrat[ii]      
      dumps.append(str(int(y))+'\t'+'[SEP]'.join(s)+'\t'+p+'\t'+' '.join([str(zi) for zi in z]))
      dumps.append(str(int(1-y))+'\t'+'[SEP]'.join(s2)+'\t'+p+'\t'+' '.join([str(zi) for zi in z]))
  return(dumps)

def reorderby(alist,inds):
   return([alist[i] for i in inds])

##########
if __name__=='__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('mdir')               ## directory of the model
  parser.add_argument('-e',default=.5) ## minimum error on og for a counterfactual  
  parser.add_argument('-sms',default=.5) ## percentile of errors to get in rat sets
  parser.add_argument('-savetag',default='') ## string added to save file
  parser.add_argument('-capfactor',default=4) ## whats the max size of the augmented dataset
  targs = parser.parse_args()  
  mdir=targs.mdir

  minerror = float(targs.e) ## minerror for taking a counterfactual
  print('MIN ERROR',)



  args = json.load(open(mdir+'config.json'))
  args['load_chkpt_dir']= mdir
  args['logfile']='log.log'

  ###from hiratcomp_model import *
  if 'model_type' not in args or args['model_type']=='hirat':
    from hirat_model import *

  args,mdict = load_classmodel(args,chkptdir=mdir)



  data_train = get_data_gen(args['traindevpair']['train_file'],
                        mdict['tokenizer'],
                      maxlen=args['max_len'],bert_len=args['bert_len'],
                      classpick=-1,rando=0,evenclass=0,
                      pady=args['pady'],
                      label_pad=args['label_pad'],
                      group_lens=0)  
  

  data_dev = get_data_gen(args['traindevpair']['dev_file'],
                        mdict['tokenizer'],
                      maxlen=args['max_len'],bert_len=args['bert_len'],
                      classpick=-1,rando=0,evenclass=0,
                      pady=args['pady'],
                      label_pad=args['label_pad'],
                      group_lens=0)  
  
  @tf.function
  def wrap_pred(model,bx,do_bx,training,bsize):
    pred,z = model(bx,do_bx,training=training,bsize=bsize)
    return(pred,z)
  ###############################
  datasets = [data_dev,data_train]
  savenames = ["og.dev.counter.onlyright.SEP5"+targs.savetag,
               "og.train.counter.onlyright.SEP5"+targs.savetag]
  ogdata = [args['traindevpair']['dev_file'],args['traindevpair']['train_file']]
  for the_dataset,savename,ogd in zip(datasets,savenames,ogdata):
    sents=[];z_inds=[];ys=[]
    k=0
    flines = get_lines(ogd)
    sents=[l.split('\t')[1].split('[SEP]') for l in flines]
    allsents=[]
    softpreds=[]
    errors=[]
    for by,bx,do_bx in the_dataset.batch(1).take(args['TOTAKE']):            
        pred,z = wrap_pred(mdict['model'],bx,do_bx,
                           training=False,bsize=tf.shape(bx)[0])
        pred = pred.numpy()[:,1]
        truth = by.numpy()[:,0]           
        for j in range(pred.shape[0]):                         
          error = abs(pred[j]-truth[j])
          if 1:
              z_inds.append(np.where(z[j].numpy()==1)[0])
              allsents.append(sents[k])
              ys.append(truth[j])
              softpreds.append(str(float(pred[j])))
              errors.append(error)
          k+=1
        #break
    #################################################
    ## dump everything
    rats0=[[s[zi] for zi in z] 
          for s,z,y,e in zip(allsents,z_inds,ys,errors) 
          if y==0.0]
    rats1=[[s[zi] for zi in z] 
          for s,z,y,e in zip(allsents,z_inds,ys,errors) 
          if y==1.0]
    dumps=make_counterdata(sentss=allsents,zss=z_inds,yss=ys,
                           softpredss=softpreds,rats0s=rats0,rats1s=rats1)
    with open(mdir+savename+'.ALL','w') as f:
      f.write('\n'.join(dumps))
      
    #################################################            
    ## filter data
    sents0=[];z0=[];y0=[];softpreds0=[];errors0=[] 
    sents1=[];z1=[];y1=[];softpreds1=[];errors1=[]
    for s,z,y,p in zip(allsents,z_inds,ys,softpreds):
      pred = float(p) 
      truth=y
      error  =abs(pred-truth)
      if error<=minerror:
        if y==0:
          sents0.append(s)
          z0.append(z)
          y0.append(y)
          softpreds0.append(str(pred))
          errors0.append(error)
        elif y==1:
           sents1.append(s)
           z1.append(z)
           y1.append(y)
           softpreds1.append(str(pred))
           errors1.append(error)
        else:
           raise NotImplementedError
    ## how many to keep per class
        ## makes balanced, never more data than original data
        ## assume that the original data is balanced
    minnum = min([int(len(allsents)/float(targs.capfactor)),len(z0),len(z1)]) 
    print('MINNUM', minnum)    
    ## sort everything by error so we get the 'best' ones
    print('LEN ERRORS', len(errors0), len(errors1))
    inds0 = np.argsort(errors0)
    inds1 = np.argsort(errors1)
    ## do the rorder
    sents0=reorderby(sents0,inds0)
    z0 = reorderby(z0,inds0)
    y0=reorderby(y0,inds0)
    softpreds0=reorderby(softpreds0,inds0)
    errors0=reorderby(errors0,inds0)
    #
    sents1=reorderby(sents1,inds1)
    z1 = reorderby(z1,inds1)
    y1=reorderby(y1,inds1)
    softpreds1=reorderby(softpreds1,inds1)
    errors1=reorderby(errors1,inds1)
    ## overwrite
    allsents = sents0[:minnum]+sents1[:minnum]
    z_inds = z0[:minnum]+z1[:minnum]
    ys = y0[:minnum] + y1[:minnum]
    softpreds = softpreds0[:minnum] + softpreds1[:minnum]    
    errors = errors0[:minnum] + errors1[:minnum]

    errors0=[e for e,y in zip(errors,ys) if y==0.0]  
    errors1=[e for e,y in zip(errors,ys) if y==1.0] 

    print('num errors', len(errors0),len(errors1))
    ## do sms filtering
    p0 = sorted(errors0)[int(np.round(len(errors0)*float(targs.sms)))]
    p1 = sorted(errors1)[int(np.round(len(errors1)*float(targs.sms)))]
    print('p0', p0,'p1',p1)
    ## get rationale set
    rats0=[[s[zi] for zi in z] 
          for s,z,y,e in zip(allsents,z_inds,ys,errors) 
          if y==0.0 and e<=p0]
    rats1=[[s[zi] for zi in z] 
          for s,z,y,e in zip(allsents,z_inds,ys,errors) 
          if y==1.0 and e<=p1]
    ## how many did we keep
    print('rat sets 0 and 1',len(rats0),len(rats1))
    dropped=k-len(rats0)-len(rats1)
    print('dropped', dropped, 'out of', k)
    ## dump stuff filtered data
    dumps=make_counterdata(sentss=allsents,zss=z_inds,
                           yss=ys,softpredss=softpreds,
                           rats0s=rats0,rats1s=rats1)
    with open(mdir+savename,'w') as f:
      f.write('\n'.join(dumps))
    #####################################################
    rats0str = ['[SEP]'.join(r) for r in rats0]
    rats1str = ['[SEP]'.join(r) for r in rats1]
    with open(mdir+savename+'.rats0.txt','w') as f:
       f.write('\n'.join(rats0str))
    with open(mdir+savename+'.rats1.txt','w') as f:
       f.write('\n'.join(rats1str))