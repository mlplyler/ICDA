import tensorflow as tf
import numpy as np
import random
import os
import pickle
from transformers import BertTokenizer, TFBertModel
from tdecoder_model import Encoder as TranEncoder
import sys
import pathlib
from collections import defaultdict
sys.path.append(str(pathlib.Path(__file__).parent.absolute())+'/../utils/for_train/')
from cf_utils import CustomSchedule, CustomScheduleBert, LR_up_down
import sys
from hugrat_utils import  update_ldict, printnsay
from hirat_model import *
###https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/models/bert/modeling_tf_bert.py#L1690

class myModel(tf.keras.Model):
  def __init__(self,args,padid,bert):    
    super().__init__()
    self.zdrop=tf.keras.layers.Dropout(rate=args['zdrop'])
    self.padding_id = padid

    ## bert
    self.bert = bert
    self.bert.trainable=args['train_bert']
    self.bert_out = self.cls_out

    ### enocder
    self.encoders=[]
    for li in range(args['encoder_num']):
      encoderlayer=TranEncoder(**args['tran_args'])
      self.encoders.append(encoderlayer)
    
    ### outlayer
    ## this is a dummy!!!
    self.linear_swap = tf.keras.layers.Dense(2,activation='softmax')      
    self.ratlayer = tf.keras.layers.Dense(2)    
    if args['num_classes'] is not None:
      if args['num_classes']==-1 and args['num_classes'] is not None:##regression
        self.outlayer = tf.keras.layers.Dense(1)  
        self.outlayer_comp = tf.keras.layers.Dense(1,name='outlayer_comp')  
      else: 
        self.outlayer = tf.keras.layers.Dense(args['num_classes'],
                                activation='softmax')  
        self.outlayer_comp = tf.keras.layers.Dense(args['num_classes'],
                                activation='softmax',name='outlayer_comp')  
    
    
    self.mini_embed = tf.keras.layers.Embedding(input_dim=1,
                        output_dim = args['tran_args']['d_model'])
    ## save the args
    self.args=args    
  def max_out(self,enc_out,masks):
    rout = enc_out * masks + (1. - masks) * (-1e6)
    rout = tf.reduce_max(rout, axis=1) 
    return(rout)
  def mean_out(self,enc_out,masks):
    print('enc shape', np.shape(enc_out))
    print('mask shape', np.shape(masks))
    rout = enc_out * masks
    print('top', np.shape(tf.reduce_sum(rout,axis=1)))
    print('bottom', np.shape(tf.reduce_sum(masks,axis=1)))
    rout = tf.reduce_sum(rout,axis=1)/(tf.reduce_sum(masks,axis=1)+1e-10) ##     
    return(rout)
  def cls_out(self,enc_out,masks=None):    
    rout = enc_out[:,0] ## take first time step
    return(rout)     
  def no_out(self,enc_out,masks=None):    
    return(enc_out)     
  def get_padmask(self,x):
    return(tf.cast(tf.not_equal(x, self.padding_id), tf.float32) )
  def emb_toks(self,x,do_x,training=True,bsize=None):  
    print('x1',np.shape(x))
    xshape=tf.shape(x)
    xflat = tf.reshape(x,[xshape[0]*xshape[1],xshape[2]])
    print('xflat', np.shape(xflat))
    attention_mask = tf.cast(tf.not_equal(xflat,self.padding_id),dtype=tf.float32)
    out=self.bert(xflat,
                  attention_mask=attention_mask,
                  training=training,output_hidden_states=True)
    print('last hiddent_state', np.shape(out.last_hidden_state))
    xbout = self.bert_out(out.last_hidden_state)
    print('first xbout', np.shape(xbout))        
    xboutshape=tf.shape(xbout)
    x=tf.reshape(xbout,[xshape[0],xshape[1],xboutshape[-1]])
    print('x after reshape', np.shape(x))          
    print('x after bert',np.shape(x))     
    return(x)       
  def emb_sents(self,x,do_x,training=True,bsize=None):
    ################################        
    dobool=do_x ## alternative to above line!
    print('do bool2 ', np.shape(dobool))
    ################################    
    for encoder in self.encoders:
      x = encoder(x,training=training,attention_mask=dobool)   
      print('encoder out', np.shape(x))  
    print('x after reclayers',np.shape(x)) 
    return(x,dobool)
  def apply_ratmask(self,x,z):
    z = tf.expand_dims(z,axis=-1)
    print('apply_mask', np.shape(x),np.shape(z))
    x = x*z + (1-z)*(-1e-10)    
    x = tf.reduce_max(x,axis=1) ## only not weird for K=1, 
    return(x)
  def z_logit_avg(self,zlogit,pred_logit):
    print('pred_logit', pred_logit)
    print('zlogit1', np.shape(zlogit))
    zlogit=tf.math.softmax(zlogit,axis=-1)
    zlogit = zlogit[:,:,1] ## just keep the do_z logit
    print('zlogit2', np.shape(zlogit))
    zlogit=tf.reduce_mean(zlogit,axis=0)## over batch
    zprod1 = zlogit*pred_logit
    zprod0 = zlogit*(1-pred_logit)    
    print('zprods', np.shape(zprod0),np.shape(zprod1))
    return([zprod0,zprod1])    
  def call(self,x,do_x,training=True,bsize=None):    
    x_toks = self.emb_toks(x,do_x,training,bsize=bsize)
    print('x_toks', np.shape(x_toks))
    x_sents,mask2 = self.emb_sents(x_toks,do_x,training,bsize=bsize)
    print('x_sents', np.shape(x_sents))
    z_logit = self.ratlayer(x_sents)

    z_logit = self.zdrop(z_logit,training=training)
    print('z_logit shape', np.shape(z_logit))
    print('z_logits', np.shape(z_logit))

    z_mask = self.zstuff_full(z_logit,
                              masks=tf.cast(mask2,dtype=tf.float32),
                              training=training,
                              bsize=bsize,maxlen=self.args['max_len'],
                              K=self.args['rat_K'])

    print('z_mask', np.shape(z_mask))
    x_rat = self.apply_ratmask(x_toks,z_mask)    
    x_comp = self.apply_ratmask(x_toks,1-z_mask)
    print('x_rat', np.shape(x_rat))    
    x_out = self.outlayer(x_rat)     
    x_out_comp = self.outlayer_comp(x_comp)     
    print('x_out', np.shape(x_out)) 
    return(x_out,z_mask,x_out_comp)
  
  @tf.function
  def zstuff_full(self,z,masks,training=False,bsize=None,maxlen=None,
                  K=1                   
                   ):
      z =  tf.math.softmax(z,axis=-1) 
      zpass = z[:,:,1]+ (1. - masks) * (-1e6)            
      z_hard = tf.one_hot(tf.math.argmax(zpass,axis=-1),depth=maxlen)
      print('SHAPES', np.shape(z_hard),np.shape(zpass), np.shape(z))
      z = tf.stop_gradient(z_hard - z[:,:,1]) + z[:,:,1]          
      return(z)
  def get_loss_g(self,loss,loss_c,lambda_g,numclass):      
      ## in "rethinking...", they say h is "some constant", 
      ## at first i thought it was the entropy of the labels, but their code makes me think 0
      #h=0#tf.math.log(numclass)
      loss = lambda_g*tf.reduce_max([loss-loss_c,0])
      return(loss)
  def get_loss(self,x,labels,makehot=True,bsize=None): 

    if self.args['num_classes']==-1:
      loss=tf.reduce_mean(
        tf.keras.losses.mean_squared_error(y_pred=x,y_true=labels),
        axis=-1)      
    else:  
      if makehot:
        print('one_hotted!!!!!!1')
        print('num classes', self.args['num_classes'])
        print('labels', np.shape(labels))
        ytrue = tf.cast(tf.one_hot(
                      tf.cast(tf.math.round(labels[:,0]),dtype=tf.int32),
                  depth=self.args['num_classes']),dtype=tf.float32)
        print('sketchy ytrue irst', np.shape(ytrue))        
      else:
        ytrue=labels      
      print('loss you think', np.shape(ytrue), np.shape(x))  
      loss = tf.keras.losses.categorical_crossentropy(
          y_true=ytrue, 
                y_pred=x, from_logits=False, label_smoothing=0.0, 
                )
      loss = tf.reduce_mean(loss)
    return(loss)
###############################################  
#@tf.function
def get_top_k_mask(arr,K,bsize=69,maxlen=69):
  '''
  magic from
  https://stackoverflow.com/questions/43294421/
  
  returns a binary array of shape array 
  where the 1s are at the topK values along axis -1
  '''
  values, indices = tf.nn.top_k(arr, k=K, sorted=False)
  temp_indices = tf.meshgrid(*[tf.range(d) for d in (tf.unstack(
        tf.shape(arr)[:(arr.get_shape().ndims - 1)]) + [K])], indexing='ij')
  temp_indices = tf.stack(temp_indices[:-1] + [indices], axis=-1)
  full_indices = tf.reshape(temp_indices, [-1, arr.get_shape().ndims])
  values = tf.reshape(values, [-1])


  mask_st = tf.SparseTensor(indices=tf.cast(
        full_indices, dtype=tf.int64), 
        values=tf.ones_like(values), dense_shape=[bsize,maxlen])
  mask = tf.sparse.to_dense(tf.sparse.reorder(mask_st),default_value=0)  
  return(mask)   
#################################################################
@tf.function
def a_step(hot,model,x,y,do_x,bsize,
           train,lambda_g,numclass,
           ):
  vars = [v for v in model.trainable_variables
          if '_comp' not in v.name]
  vars_comp = [v for v in model.trainable_variables
          if '_comp'  in v.name]
  with tf.GradientTape() as gradtape, tf.GradientTape() as comptape:
    out,z,out_comp = model(x,do_x,training=train,bsize=bsize)    
    loss_og = model.get_loss(out,y,hot,bsize=bsize)
    loss_comp = model.get_loss(out_comp,y,hot,bsize=bsize)
    loss =loss_og + model.get_loss_g(loss=loss_og,loss_c=loss_comp,
                                     lambda_g=lambda_g,numclass=numclass)    
  print('vars', len(vars))
  print('vars_comp', len(vars_comp))
  grads = gradtape.gradient(loss,vars)
  grads_comp = comptape.gradient(loss_comp,vars_comp)
  return(grads,grads_comp)
#@tf.function
def apply_grads(optimizer,accum_gradient,vars):
  optimizer.apply_gradients(zip(accum_gradient,vars))
#@tf.function
def accum_training_epoch(data_train,args,mdict,train):
  vars = [v for v in mdict['model'].trainable_variables
          if '_comp' not in v.name]
  vars_comp = [v for v in mdict['model'].trainable_variables
          if '_comp'  in v.name]
  train=tf.constant(train,dtype=tf.bool)
  numsamp=tf.constant(args['num_samples'],dtype=tf.float32)
  accum_gradient = [tf.zeros_like(this_var) for this_var in vars]
  accum_gradient_comp = [tf.zeros_like(this_var) for this_var in vars_comp]
  islive=tf.constant(0,dtype=tf.float32);
  ahot = tf.constant(bool(1-args['hot_y']),dtype=tf.bool)
  #k=0
  for by,bx,do_bx in data_train.batch(
        args['train_batch'],num_parallel_calls=-1).take(
          args['TOTAKE']):             
    bsize=tf.shape(bx)[0]
    grads,grads_comp = a_step(hot=ahot,model=mdict['model'],x=bx,
                   y=by,do_x=do_bx,bsize=bsize,
                   train=train,
                   lambda_g=args['lambda_g'],
                   numclass = args['traindevpair']['num_classes'],
                   )    
    accum_gradient = [(acum_grad+grad) 
                      if grad is not None else acum_grad
                       for acum_grad, grad 
                        in zip(accum_gradient, grads)]
    accum_gradient_comp = [(acum_grad_comp+grad_comp) 
                      if grad_comp is not None else acum_grad_comp
                       for acum_grad_comp, grad_comp 
                        in zip(accum_gradient_comp, grads_comp)]    
    islive+=tf.cast(bsize,dtype=tf.float32)

    if islive%numsamp==0:
      ## update
      accum_gradient = [this_grad/numsamp
                        for this_grad in accum_gradient]
      accum_gradient_comp = [this_grad/numsamp
                        for this_grad in accum_gradient_comp]
      apply_grads(mdict['optimizer'],accum_gradient,vars)
      apply_grads(mdict['optimizer_comp'],accum_gradient_comp,vars_comp)
      
      ## reset
      accum_gradient = [tf.zeros_like(this_var) 
                        for this_var in vars]
      accum_gradient_comp = [tf.zeros_like(this_var) 
                        for this_var in vars_comp]
      islive=tf.constant(0,dtype=tf.float32);

  if islive>0: ## do the left over    
    accum_gradient = [this_grad/islive
                      for this_grad in accum_gradient]
    accum_gradient_comp = [this_grad/islive
                      for this_grad in accum_gradient_comp]
    apply_grads(mdict['optimizer'],accum_gradient,vars)
    apply_grads(mdict['optimizer_comp'],accum_gradient_comp,vars_comp)    
##################################################



@tf.function
def jpraw(args,model,opt,x,do_x,y,train,bsize=None):
  out,z,out_comp = model(x,do_x,training=train,bsize=bsize)    
  loss_og = model.get_loss(out,y,1-args['hot_y'],bsize=bsize)
  loss_comp = model.get_loss(out_comp,y,1-args['hot_y'],bsize=bsize)
  loss =loss_og + model.get_loss_g(loss=loss_og,loss_c=loss_comp,
                                     lambda_g=args['lambda_g'],
                            numclass=args['traindevpair']['num_classes'])    

  acc = pred_acc(y,out,args['label_pad'],
                 False)
  return(loss,acc)

@tf.function
def jpraw_wz(args,model,opt,x,do_x,y,train,bsize=None):
  out,z,out_comp = model(x,do_x,training=train,bsize=bsize)    
  loss_og = model.get_loss(out,y,1-args['hot_y'],bsize=bsize)
  loss_comp = model.get_loss(out_comp,y,1-args['hot_y'],bsize=bsize)
  loss =loss_og + model.get_loss_g(loss=loss_og,loss_c=loss_comp,
                                     lambda_g=args['lambda_g'],
                            numclass=args['traindevpair']['num_classes'])    
  acc = pred_acc(y,out,args['label_pad'],
                 False)
  if args['hot_y']:
      y = tf.argmax(y,axis=-1)
  pred_hard = tf.cast(tf.equal(x=out, 
                      y=tf.reduce_max(out, -1, keepdims=True)),
                            y.dtype)[:,1] 
  return(loss,acc,z,pred_hard)

@tf.function
def jppreds(args,model,opt,x,do_x,y,train,bsize=None):
  out,z,out_comp = model(x,do_x,training=train,bsize=bsize)    
  loss_og = model.get_loss(out,y,1-args['hot_y'],bsize=bsize)
  loss_comp = model.get_loss(out_comp,y,1-args['hot_y'],bsize=bsize)
  loss =loss_og + model.get_loss_g(loss=loss_og,loss_c=loss_comp,
                                     lambda_g=args['lambda_g'],
                            numclass=args['traindevpair']['num_classes'])    
  pred_hard = tf.cast(tf.math.round(tf.argmax(out,axis=-1)),y.dtype)  
  return(out,pred_hard,loss)


def cag_wrap(args,mdict,cag,x,do_x,y,train,bsize=None):
  loss,acc=cag(args,mdict['model'],mdict['optimizer'],x,
               do_x,y,train,bsize=bsize)
  loss = loss.numpy()
  acc=  acc.numpy()
  ddict={}
  ddict['loss']=loss
  ddict['acc']=acc
  return(ddict)

def cag_wrap_wdat(args,mdict,cag,x,do_x,y,train,bsize=None):
  loss,acc,pred_hard,pred_soft=cag(args,mdict['model'],mdict['optimizer'],
                                   x,do_x,y,train,bsize=bsize)
  loss = loss.numpy()
  acc=  acc.numpy()
  ddict={}
  ddict['loss']=loss
  ddict['acc']=acc
  return(ddict,pred_hard.numpy(),pred_soft.numpy())


@tf.function
def jpraw_wdat(args,model,opt,x,do_x,y,train):
  out = model(x,do_x,training=train)    
  loss = model.get_loss(out,y,1-args['hot_y'])
  if args['num_classes']!=-1:
    if args['hot_y']:
      y = tf.argmax(y,axis=-1)
      acc = pred_acc(y,out)
    else:
      acc = pred_acc(y[:,0],out)
    pred_hard = tf.cast(tf.equal(x=out, 
                      y=tf.reduce_max(out, -1, keepdims=True)),
                            y.dtype)     
  else:
    acc=-1
    pred_hard=out
  return(loss,acc,pred_hard,out)



def load_classmodel(args,chkptdir=None,chkpt_text=''):
  args = add_default_args(args,rollrandom=False)
  if chkptdir is None and args['load_chkpt_dir'] is not None and len(args['load_chkpt_dir'])>0:
    chkptdir=args['load_chkpt_dir']
  mdict={}
  mdict['args']=args  
  #### model 
  try:
    btokenizer = BertTokenizer.from_pretrained(args['hug_chkpt'])    
    print('got tokenizer from hug_chkp', args['hug_chkpt'])
  except:
    if 'tokendir' not in args:      
      tokendir = args['hug_chkpt']+'/tokenizer/'
    else:
      tokendir = args['tokendir']
    ###
    print("tokendir", tokendir)
    btokenizer = BertTokenizer.from_pretrained(tokendir)    
  
  mdict['tokenizer']=btokenizer

  try:
    bmodel = TFBertModel.from_pretrained(args['hug_chkpt'],from_pt=True,
      hidden_dropout_prob=args['dropout_bert'],
      attention_probs_dropout_prob=args['dropout_bert'],
          )
  except:
    bmodel = TFBertModel.from_pretrained(args['hug_chkpt']+'/hugchkpt/',                    
                      from_pt=False,
                      hidden_dropout_prob=args['dropout_bert'],
                      attention_probs_dropout_prob=args['dropout_bert'],
                      )

  mdict['model']=myModel(args,padid=mdict['tokenizer'].pad_token_id,
                            bert=bmodel)  
  #### checkpoint stuff
  mdict['theckpt'] = tf.train.Checkpoint(step=tf.Variable(1),
                              net=mdict['model'])
  if chkpt_text=='':
    mdict['chkptman'] = tf.train.CheckpointManager(
                              mdict['theckpt'],
                              args['log_path'],                              
                              max_to_keep=args['max_chkpts'])
  else:
    print('ELSE ON CHECKPT TEXT', chkpt_text)
    mdict['chkptman'] = tf.train.CheckpointManager(
                            mdict['theckpt'],
                            args['log_path']+chkpt_text+'/', 
                            max_to_keep=args['max_chkpts'])
                                     
                                     

  if chkptdir is not None and chkptdir!='':
    if 'ckpt' not in chkptdir and args['ckpt']=='':
      chkptdir = tf.train.latest_checkpoint(chkptdir)
      print('GETTING ALST CHECKPOINT',chkptdir)
    else:
      ## only get here if load_chkpt_dirORchkptdir and ckpt
      print('USING ARGS[ckpt]')
      chkptdir = args['ckpt']
      

    print('LOADING loading classifier', chkptdir)
    mdict['restore_status']=mdict['theckpt'].restore(
                chkptdir
                            ).assert_nontrivial_match()#.assert_consumed()#
  else:
    print('NO LOAD CLASS MODEL', chkptdir)  
  #### optimizer  
  if args['lr'] == -1:
      cflr =  CustomSchedule(float(bmodel.config.hidden_size))
  elif args['lr']==-2:
    cflr = CustomScheduleBert(warmup_steps=args['warmup'],
                              peak_lr=args['peak_lr'])
  elif args['lr']==-3:
    train_lines = sum(1 for _ in open(args['train_file']))
    train_batch_counts = args['abs_max_epoch']*int(np.round(
          train_lines/(args['train_batch']*args['num_gpu']*args['num_samples'])))
    print('train_lines', train_lines)
    print('train_batch_counts')
    if 'lr_b1' not in args:
      b1 =int(np.round(train_batch_counts*.1))
    else:
      b1=args['lr_b1']
    if 'lr_b2' not in args:
      b2=int(np.round(train_batch_counts*.9))
    else:
      b2 = args['lr_b2']
    boundaries = [b1,b2]
    print('boundaries', boundaries)
    cflr = LR_up_down(peak_lr=args['peak_lr'],
                          warmup_end=boundaries[0],
                          decline_start=boundaries[1])
  else:  
    cflr = tf.Variable(args['lr'],dtype=tf.float32)
  if args['weight_decay']==0:
    mdict['optimizer'] =  tf.keras.optimizers.Adam(learning_rate=cflr)   
    mdict['optimizer_comp'] =  tf.keras.optimizers.Adam(learning_rate=cflr)   
  else:
    mdict['optimizer'] =  tf.keras.optimizers.AdamW(learning_rate=cflr,
                                                    weight_decay=args['weight_decay'])   
    mdict['optimizer_comp'] =  tf.keras.optimizers.AdamW(learning_rate=cflr,
                                                    weight_decay=args['weight_decay'])   
  return(args,mdict)

def add_default_args(args,rollrandom=True):
  defaultargs = {    
    "train_file":"",
    "dev_file":"",
    "test_file":"",
    "train_batch":16,
    "eval_batch":16,
    "vocab_size":5,
    "embedding_dim":128,
    "recurrent_dim":128,    
    "recurrent_type":"gru",
    "recurrent_num":1,
    "dropout":.1,
    "dropout1":0.0,
    "dropout2":0.0,
    "weight_decay":0,
    "recurrent_dropout":0.0,
    "dropout_bert":0.10,
    "log_path":"",
    "load_chkpt_dir":"", ## where you want to load a chkpt from
    "ckpt":"",           ## which checkpoint you want, "" for last checkpoint
    "data_type":"ratform",
    "aspects":[0],
    "num_classes":1,
    "lr":1e-5,
    "TOTAKE":1000000,
    "max_len":64,
    "bert_len":512,
    "abs_max_epoch":20,
    "dosave":1,
    "train_bert":1,
    "num_gpu":1,
    "num_samples":1,
    "shuffle_sents":0,
    'check_num':8000,
    "sms":0.10,
    "min_error":0.50,
    "max_chkpts":1,
    "lambda_g":1,
    "shuffrats":0,
    "logfile" : "logfile.log"}
  newargs = dict(defaultargs)
  theseed = random.randint(1,10000000)
  newargs['rand_seed']=theseed ## this will be overwritten if in args
  try:
    newargs['HOSTNAME']=os.environ['HOSTNAME']
  except:
    newargs['HOSTNAME']=None 
  for k in args:
    newargs[k]=args[k]
  newargs['logfile']=newargs['log_path']+newargs['logfile']
  newargs['resfile']=newargs['log_path']+'eval_results.txt'

  
  if 'train_max_len' not in newargs:
    newargs['train_max_len']=newargs['max_len']
  if 'train_bert_len' not in newargs:
    newargs['train_bert_len']=newargs['bert_len']
  if 'dev_max_len' not in newargs:
    newargs['dev_max_len']=newargs['max_len']
  if 'dev_bert_len' not in newargs:
    newargs['dev_bert_len']=newargs['bert_len']    
  return(newargs) 

