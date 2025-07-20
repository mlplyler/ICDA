import argparse
import json
import os
import numpy as np
############################################
def get_lines(fname):
    fstr = open(fname,'r').read()
    flines = fstr.split('\n')
    flines= [ l for l in flines if len(l)>0]
    return(flines)  
############################################
def set_perc_change(set0,set1):
    return((len(set0)-len(set0.intersection(set1)))/len(set1))
############################################
def get_full_rats(apath):
    ## get rationales on full train dataset
    alines = get_lines(apath+'og.train.counter.onlyright.SEP5.ALL')
    oglines = [l for i,l in enumerate(alines) if i%2==0]
    rats  = [l.split('\t')[1].split('[SEP]')[int(l.split('\t')[-1])] for l in oglines]
    rats0=set([r for r,l in zip(rats,oglines) if l.startswith('0')])
    rats1=set([r for r,l in zip(rats,oglines) if l.startswith('1')])
    return(rats0,rats1)
############################################
def get_loss(logfile):
  if not os.path.exists(logfile):
    return(1e10)
  fstr = open(logfile).read()
  flines=[l for l in fstr.split('\n')
          if len(l)>0]
  devlines = [l for l in flines if 'dev' in l]
  not_first_degen = [eval(d.split('not_first_degen:')[1].split(',')[0])
                      for d in devlines]
  not_class_degen = [eval(d.split('not_class_degen:')[1].split(',')[0])
                      for d in devlines]
  lss = [float(d.split('dev_loss:')[1].split(',')[0]) for d in devlines]
  chks = [float(d.split('chkpt:')[1].split(',')[0]) for d in devlines]
  gotinds = [i for i,g in enumerate(chks) if g]
  minind = max(gotinds) if len(gotinds)>0 else 0
  ## check to see if this model is degenerate
  ## it is, multiply the loss by big number so its not picked
  if not not_first_degen[minind] or not not_class_degen[minind]:
    mult = 1e10
  else:
    mult=1
  return(lss[minind]*mult)
############################################
def get_chkpt(logfile):
  fstr = open(logfile).read()
  flines=[l for l in fstr.split('\n')
          if len(l)>0]
  devlines = [l for l in flines if 'dev' in l]
  chks = [float(d.split('chkpt:')[1].split(',')[0]) for d in devlines]
  gotinds = [i for i,g in enumerate(chks) if g]
  minind = max(gotinds) if len(gotinds)>0 else 0
  return(minind)## return the index of the last checkpoint
############################################
############################################
############################################
if __name__=='__main__':
  ## parse args
  parser = argparse.ArgumentParser()
  parser.add_argument('configs')
  parser.add_argument('thedir')
  targs = parser.parse_args()  

  ## parse configs
  with open(targs.configs,'r') as f:
      cstr = f.read()
  args = json.loads(cstr)

  ## supername, usued to name future iters
  ##   drop the last /, now a training _, need to add a / later
  supername=args['log_path'][:-1]+'_'

  ## initialize next file paths
  next_args = dict(args) ## a dictionary
  next_args_path=targs.configs   ## path to a json
  next_log_path=next_args['log_path'] ## path to where we save the model and logs
  old_args = dict(next_args)  

  CAPFACTOR = 4#max([1,4-i])

  ## do iterative CDA  
  diff=1
  keepiter=True
  r0s=[];r1s=[]
  for i in range(args['max_cda_iters']):   
    BESJ=None
    isbetter_diffj=[]
    ## save original stuff
    ogargs=dict(next_args)            
    print('OGARGS rand seed', ogargs['rand_seed'])
    next_log_path=ogargs['log_path']
    if not os.path.exists(ogargs['log_path']):
      os.makedirs(ogargs['log_path'])
    ## do multiple runs, first is fresh, second is load previous
    J = len(args['rand_seeds']) if i==0 else len(args['rand_seeds'])+1  
    for j in range(J):
      ## modify log_path
      next_args_j = dict(ogargs)
      next_args_path_j = next_log_path+str(j)+'/config.json'
      if not os.path.exists(next_args_path_j): ############ ONLY IF WE DONT HAVE CONFIGS
        ## if its the last run, get the previous load_chkpt
        if j >= len(args['rand_seeds']):
           next_args_j['load_chkpt_dir'] = old_args['log_path']
        next_args_j['log_path']=ogargs['log_path']+str(j)+'/'
        ## update the seed
        next_args_j['rand_seed'] = args['rand_seeds'][j] if j<len(args['rand_seeds']) else ogargs['rand_seed']
        if next_args_j['rand_seed']==-1:
           next_args_j = args['rand_seeds'][0]
        ## save args      
        if not os.path.exists(next_args_j['log_path']):
          os.makedirs(next_args_j['log_path'])
        with open(next_args_path_j,'w') as f:       
          json.dump(next_args_j,f,indent=2)
        ## do the training
        print('ITERATION', i, 'RUN', j)
        ## train_selector
        os.system(
          'cd ../models/{}/sw/train/ && python train_hirats.py {}'.format(        
              targs.thedir,              
              next_args_path_j)) 
        ## infer_counterfacutals, concatentate
        print('mdir',next_args_j['log_path'])
        os.system(
          'cd ../models/{}/sw/utils/ && python dump_CDA.py {}  -e {} -sms {} -capfactor {}'.format(
          targs.thedir,          
          next_args_j['log_path'],##mdir  
          next_args_j['min_error'], #e
          next_args_j['sms'], #sms
          CAPFACTOR

          ))
        os.system(
          "cd ../models/{}/sw/utils/ && python extract_counter_data.py {}".format(
            targs.thedir,
            next_args_j['log_path']
          ))
        ## get the rationale sets
        try:
          r0,r1 = get_full_rats(next_args_j['log_path'])
        except:
           r0,r1 = set(),set() ## if we can't load them, assume empty
        ## diff from previously used
        if i>1:
          diffj = (set_perc_change(r0s[-1],r0)+set_perc_change(r1s[-1],r1))/2
          ## if the rate of change in rationales is decreasing, lets just call it good
          if diffj<diff:
             isbetter_diffj.append(1) ## its improving so multiplier is 1
          else:
             isbetter_diffj.append(1e10)                          
        else:
           isbetter_diffj.append(1)
      else: ## THIS ELSE ACTUALLY BREAKS THE INTENTION OF ISBETTER_DIFF IF YOU ARE LOADING MODELS PAST I>1 CUZ YOU HAVENT DONT THE SET_PERC_CHANGE THING
           isbetter_diffj.append(1)     
    ##################################
    ## only if we havent made the cfs
    if not os.path.exists(next_log_path+'og.dev.selected5'):       
      ## parse the logs from the j runs      
      losses = [get_loss(next_log_path+str(j)+'/logfile.log')*isbetter_diffj[j]
                  for j in range(J)]            
      print('LOSSes', losses)
      ## get the path to the best run  
      minind = np.argmin(losses)
      print('MININD', minind)      
      bespath = next_log_path+str(minind)+'/'
      print('bespath', bespath)
      print('next_log_path', next_args_path)
      ## copy the best stuff to the base next_path
      os.system('cp -r  {}/* {}'.format(bespath,next_log_path))
      ## update the log_path in that new config.json, this makes it so log_path doesnt have j
      with open(next_log_path+'config.json','r') as f:
          cstr = f.read()
      newargs = json.loads(cstr)    
      newargs['log_path']=next_log_path
      ## parse the log to get the checkpoint ind
      chkind = get_chkpt(next_log_path+'logfile.log')
      if chkind==0: ## if we didnt do any training cuz epoch==-1
        keepiter=False
      ## save that newarg
      with open(next_log_path+'config.json','w') as f:       
        json.dump(newargs,f,indent=2)
    #exit()
    ## load the previous args in case we don't have them cuz skipped
    with open(next_log_path+'config.json','r') as f:
          cstr = f.read()
    newargs = json.loads(cstr)        

    ## prepare arguments for next iteration  
    old_log_path=str(next_args['log_path'])  
    old_args = dict(next_args)
    next_args['train_file']=next_args['log_path']+'og.train.counter.onlyright.SEP5'
    next_args['dev_file']=next_args['log_path']+'og.dev.counter.onlyright.SEP5'
    next_args['dev_files']=[next_args['log_path']+'og.dev.selected5',
                            next_args['log_path']+'og.dev.counter5']
    next_args['log_path']  =supername+str(i)+'/'
    next_args['rand_seed']=newargs['rand_seed']## get that rand_seed from best previous run
    next_args['shuffrats']=1 ##!!!!!!!!!!!!!!!!!!!!!!!!

    ## track the train rationales used
    r0,r1 = get_full_rats(old_log_path)
    r0s.append(r0);r1s.append(r1)
    ## are we converging?!
    if i>0:
       diff = (set_perc_change(r0s[i-1],r0s[i])+set_perc_change(r1s[i-1],r1s[i]))/2

    if not os.path.exists(next_args['log_path']):
      ## save to current log_path
      next_args_path = old_log_path+'/next_configs.json'
      with open(next_args_path,'w') as f:       
        json.dump(next_args,f,indent=2)
    ## check if iterative has 'converged'
    if not keepiter:
      exit()



