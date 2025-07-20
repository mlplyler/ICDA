

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"
from transformers import BertTokenizer, TFBertModel ## this import is mysteriously important

import pickle
import time
import json

import numpy as np
import json
import argparse
from collections import defaultdict, Counter

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
print('physicial devices', physical_devices)
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

import sys
sys.path.append('../utils/for_train/')
from hugrat_utils import set_seeds, get_dataset_rec, printnsay, update_ldict
sys.path.append('../layers/')

#############################################
def dev_epoch_wz(a_data_dev):
  print('EVAL WZ', epoch)
  bins=np.arange(0,1.1,.1)
  bincounts=[Counter(),Counter()]
  devdict=defaultdict(list)  
  ## evaluation
  bsizes=[]
  egs=[];ees=[];elosses=[];belosses=[];eobjs=[];epks=[];ezsum=[];ezdiff=[];
  for by,bx,do_bx in a_data_dev.batch(args['eval_batch']).take(TOTAKE):    
    bsize = tf.shape(bx)[0]
    dev_ddict=cag_wrap(args=args,mdict=mdict,
                         cag=jpraw,
                         x=bx,do_x=do_bx,y=by,train=False,
                         bsize=bsize)  
    loss,acc,z,pred_hard=jpraw_wz(args=args,model=mdict['model'],opt=None,
                        x=bx,do_x=do_bx,y=by,train=False,
                        bsize=tf.shape(bx)[0])
    pred_hard=pred_hard.numpy()    
    for j in range(bsize):      
      zj = np.where(z[j].numpy()==1)[0]
      numdo=tf.reduce_sum(do_bx[j],axis=-1).numpy()
      ratbins=np.digitize(zj/numdo,bins)
      ys=by[j,0].numpy()
      if ys==0 and pred_hard[j]==0:
        bincounts[0].update(ratbins)
      elif ys==1 and pred_hard[j]==1:
        bincounts[1].update(ratbins)
    dev_ddict = {'loss':loss.numpy(),
                 'acc':acc.numpy(),
                 }
    mask = tf.cast(tf.not_equal(by,args['label_pad']),dtype=tf.int32)
    num = tf.reduce_sum(mask)
    bsizes.append(num.numpy())    
                                
    devdict = update_ldict(devdict,dev_ddict)
  for i,bincount in enumerate(bincounts):
    skeys=sorted(bincount.keys())
    vals=[bincount[k] for k in range(1,12)]
    vals=np.array(vals)/np.sum(vals)
    devdict['ratbins'+str(i)]='//'.join(['{:.5f}'.format(k) for k in vals])
    print('devdict ratbins', devdict['ratbins'+str(i)])
  return(bsizes,devdict)  
#############################################
def checkpoint_logic(besdev_epoch,thebesdev,devbsizes,devdicts,degen_margin=.2):  
  dev_obj = np.mean([np.mean([np.dot(bsizes,devdict[x])/np.sum(bsizes)
                      for x in devdict if 'loss' in x]) ## gen cost 
              for bsizes,devdict in zip(devbsizes,devdicts)]) ## average over files
  
  bins1 = [[float(t) for t in devdict['ratbins1'].split('//')] for devdict in devdicts]
  bins0 = [[float(t) for t in devdict['ratbins0'].split('//')] for devdict in devdicts]

  first_degen =[
            abs(abs(b1[0]-b1[1]) - abs(b0[0]-b0[1]))<=degen_margin
                 for b1,b0 in zip(bins1,bins0)]
  class_degen = [max([abs(b0-b1) for b0,b1 in zip(bins0_,bins1_)])<=degen_margin
                 for bins0_,bins1_ in zip(bins1,bins0)]
  if len(devdicts)==2:
    each_other0 = max([bins0[0][i]-bins0[1][i] for i in range(len(bins0[0]))]) 
    each_other1 = max([bins1[0][i]-bins1[1][i] for i in range(len(bins1[0]))]) 
    each_other = 1 if max([each_other0,each_other1])<=.3 else 0
    print('each others', each_other0,each_other1,each_other)
  else:
    each_other=1

  print('first_degen', first_degen)
  print('class_degen', class_degen)

  first_degen_bool = sum(first_degen)==len(args['dev_files'])
  class_degen_bool = sum(class_degen)==len(args['dev_files'])

  gotchkpt=0    
  if ((dev_obj<thebesdev) ## beter pred      
      and first_degen_bool
      and class_degen_bool      
      ) or epoch==-1:
    besdev_epoch = epoch
    thebesdev = dev_obj
    print('NEW BEST!!', thebesdev)
    if args['dosave']: 
      gotchkpt=1     
      save_path = mdict['chkptman'].save()
      print('saved bes to', save_path)  # put logfile in right place
  return(besdev_epoch,thebesdev,gotchkpt,first_degen_bool,class_degen_bool)
#############################################
if __name__=='__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('configs')
  targs = parser.parse_args()  
  ## args
  with open(targs.configs,'r') as f:
      cstr = f.read()
  args = json.loads(cstr)
  
  if 'model_type' not in args or args['model_type']=='hirat':
    print('HIRAT MODEL')
    from hirat_model import *      
  elif args['model_type']=='comp':
    print('COMP MODEL')
    from hiratcomp_model import *

  print('ARGS\n\n',args,'\n\n')
  #######
  # make the logpath
  if not os.path.exists(args['log_path']):
      os.makedirs(args['log_path'])
  
  print('ARGS\n\n',args,'\n\n')
  
  ## set random seed
  set_seeds(args['rand_seed'])  
  ######set_seeds(69)  
  ## load model
  args,mdict = load_classmodel(args)
  ## get data huggingface
  if args['shuffrats']:
    data_train = get_data_gen_shuffrats(args['train_file'],
                        mdict['tokenizer'],
                      maxlen=args['max_len'],bert_len=args['bert_len'],
                      classpick=-1,rando=1,evenclass=0,
                      pady=args['pady'],
                      label_pad=args['label_pad'],
                      shuffle_sents=args['shuffle_sents'],
                      group_lens=args['group_lens'])  
  else:
    data_train = get_data_gen(args['train_file'],
                        mdict['tokenizer'],
                      maxlen=args['max_len'],bert_len=args['bert_len'],
                      classpick=-1,rando=1,evenclass=0,
                      pady=args['pady'],
                      label_pad=args['label_pad'],
                      shuffle_sents=args['shuffle_sents'],
                      group_lens=args['group_lens'])  

  data_devs = [get_data_gen(devfile,
                        mdict['tokenizer'],
                      maxlen=args['max_len'],bert_len=args['bert_len'],
                      classpick=-1,rando=0,evenclass=0,
                      pady=args['pady'],
                      label_pad=args['label_pad'],
                      shuffle_sents=0,
                      group_lens=args['group_lens'])  
                for devfile in args['dev_files']]
    ## training loop stuff
  besdev_epoch=np.inf#0
  ti_track=0
  thebesdev = np.inf
  gotsparse=False;firstsparse=False;
  
  if 'TOTAKE' in args:
    TOTAKE=args['TOTAKE']
    print('TOTAKE', TOTAKE)
  else:
    TOTAKE=400000000
  print('ARGS\n\n',args,'\n\n')
  with open(args['log_path']+'config.json','w') as f:
      json.dump(args,f,indent=2)
  

  
  for epoch in range(-1,args['abs_max_epoch']):
    print('EPOCH', epoch)
    ## training
    if epoch>-1:      
      accum_training_epoch(data_train=data_train,
                           args=args,mdict=mdict,train=True)
      traindict = {'acc':[np.nan],'loss':[np.nan]}
      trainbsizes = [1]
    else:
      traindict = {'acc':[np.nan],'loss':[np.nan]}
      trainbsizes = [1]
    ## dev check
    devbsizes=[];devdicts=[]
    for data_dev_gen in data_devs:
      bsizes,devdict = dev_epoch_wz(data_dev_gen)    
      devbsizes.append(bsizes);devdicts.append(devdict)
    ## checkpoint logic    
    print('CHECKPOINT LOGIC')                                
    besdev_epoch,thebesdev,gotchkpt,first_degen_bool,class_degen_bool = checkpoint_logic(
                            besdev_epoch,thebesdev,
                            devbsizes,devdicts,
                            degen_margin=args['degen_margin'])
    
    ## log a mean devdict as the base, the individuals will have _i attached
    skeys = [x for x in devdict.keys() if not type(devdict[x])==str]
    devdict_m = {x:np.mean([np.dot(bsizes,devdict[x])/np.sum(bsizes)
                            for bsizes,devdict in zip(devbsizes,devdicts)])
                            for x in skeys}
    printnsay(thefile=args['logfile'],
        text = 'epoch:{:.0f}'.format(epoch) + ','        
        +','.join(['train_{}:{:.5f}'.format(x,
        np.dot(trainbsizes,traindict[x])/np.sum(trainbsizes))
         for x in traindict])
        +','+','.join(['dev_{}:{:.5f}'.format(x,devdict_m[x]) for x in devdict_m])+','
        +','.join([','.join(['dev_{}:{:.5f}'.format(x+'_'+str(devi),
        np.dot(bsizes,devdict[x])/np.sum(bsizes))   
        if not type(devdict[x])==str else 
          'dev_{}:{}'.format(x+'_'+str(devi),devdict[x])  ## if its a string, just keep it    
        for x in devdict]) for devi,(bsizes,devdict) in enumerate(zip(devbsizes,devdicts))])
        +',not_first_degen:'+str(first_degen_bool)
        +',not_class_degen:'+str(class_degen_bool)
        +',chkpt:'+str(gotchkpt)
        )      


    ## max epoch check
    if epoch>=args['abs_max_epoch']-1: ## -1 becuase first epoch is zero!
        printnsay(thefile = args['logfile'],
                text = 'BROKE!! epoch '+str(epoch))
        print('DONE EPOCHS', epoch)
        #break
        exit()
        
    ## track the epoch
    mdict['theckpt'].step.assign_add(1)

    ## this captures case of -1 epoch not being good...
    if besdev_epoch==np.inf and epoch>=0:
      besdev_epoch=epoch

    ## patience check
    if epoch-besdev_epoch >= args['patience']:
      printnsay(thefile =args['logfile'],
      text='early exit '+str(besdev_epoch))
      exit()
