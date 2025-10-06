import tensorflow as tf
import numpy as np
import json
from collections import defaultdict,Counter

#####################
class LR_up_down(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, peak_lr, warmup_end=1000,decline_start=10000):
    super(LR_up_down, self).__init__()
    self.thetiny = tf.constant(1e-20,dtype=tf.float32)
    self.delta=(peak_lr-self.thetiny)/warmup_end  
    self.warmup_end=tf.constant(warmup_end,dtype=tf.float32)
    self.decline_start=tf.constant(decline_start,dtype=tf.float32)
    self.peak=tf.constant(peak_lr,dtype=tf.float32)
  def __call__(self, step):
    ## autograph doesnt like elif or skipping else, so this
    step=tf.cast(step,tf.float32)       
    lr = self.peak 
    lr = tf.cond(step<self.warmup_end,
                 true_fn=lambda:self.thetiny+step*self.delta,
                 false_fn=lambda:lr)
    lr = tf.cond(step>=self.decline_start,
                 true_fn=
              lambda:self.peak-self.delta*(step-self.decline_start),
                 false_fn=lambda:lr)
    return  tf.math.maximum(self.thetiny, lr)    
###############################

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)
    
    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)                        
     
class CustomScheduleBert(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, warmup_steps=10000,down_steps=200000,peak_lr=1e-4):
    super(CustomScheduleBert, self).__init__()

    self.start_lr = 1e-12
    
    self.warmup_steps = warmup_steps
    self.down_steps = down_steps
    self.peak_lr = peak_lr
    
    self.wstep = tf.cast((peak_lr-self.start_lr)/warmup_steps,dtype=tf.float32)
    self.dstep = tf.cast((peak_lr-self.start_lr)/(down_steps-warmup_steps),dtype=tf.float32)
    self.tracker=0

  @tf.function
  def __call__(self, step):   
    step=tf.cast(step,tf.float32)
    if step<=self.warmup_steps:
      lr = self.start_lr+self.wstep*step
    else:
      lr = self.peak_lr-self.dstep*(step-self.warmup_steps)
    return(lr)                     
 