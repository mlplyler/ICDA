
import os
import json
import argparse

##########################################################################
if __name__=='__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('mdir')
  targs = parser.parse_args()  
  mdir=targs.mdir

  for k in ['train','dev']:
    fstr = open(mdir+'og.{}.counter.onlyright.SEP5'.format(k),'r').read()
    flines = fstr.split('\n')
    
    oglines = [flines[i] for i in range(0,len(flines),2)]
    colines = [flines[i] for i in range(1,len(flines),2)]

    print('oglines', len(oglines))
    print('flines', len(flines))
    print('colines', len(colines))

    with open(mdir+'og.{}.selected5'.format(k),'w') as f:
      f.write('\n'.join(oglines))
    with open(mdir+'og.{}.counter5'.format(k),'w') as f:
      f.write('\n'.join(colines))