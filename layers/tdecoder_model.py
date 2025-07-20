## from https://www.tensorflow.org/text/tutorials/transformer
import numpy as np
import tensorflow as tf


#################################################################

class Encoder(tf.keras.layers.Layer):
  def __init__(self, *, num_layers, d_model, num_heads,
               dff, dropout_rate=0.1,do_pos=True):
    super().__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.pos_embedding = PositionalEmbedding(
         d_model=d_model)

    self.enc_layers = [
        EncoderLayer(d_model=d_model,
                     num_heads=num_heads,
                     dff=dff,
                     dropout_rate=dropout_rate)
        for _ in range(num_layers)]
    self.layer_norm = tf.keras.layers.LayerNormalization()
    self.dropout = tf.keras.layers.Dropout(dropout_rate)
    

    self.do_pos=do_pos
  def call(self, x,attention_mask=None):
    # `x` is token-IDs shape: (batch, seq_len)
    if self.do_pos:
      x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.

    # Add dropout.
    x = self.dropout(x)

    for i in range(self.num_layers):
      x = self.enc_layers[i](x,attn_mask=attention_mask)

    return x  # Shape `(batch_size, seq_len, d_model)`.  



######################################################################
class Decoder(tf.keras.layers.Layer):
  def __init__(self, *, num_layers, d_model, num_heads, dff, 
               dropout_rate=0.1):
    super(Decoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.pos_embedding = PositionalEmbedding(
                                             d_model=d_model)
    self.dropout = tf.keras.layers.Dropout(dropout_rate)
    self.dec_layers = [
        DecoderLayer(d_model=d_model, num_heads=num_heads,
                     dff=dff, dropout_rate=dropout_rate)
        for _ in range(num_layers)]


  def call(self, x):
    '''
    from tutorial
    # `x` is token-IDs shape (batch, target_seq_len)

    for us, x is already embedded
    '''
    print('decoder x', np.shape(x))
    x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)

    x = self.dropout(x)

    for i in range(self.num_layers):
      x  = self.dec_layers[i](x)
      print('dec_layer', np.shape(x))



    # The shape of x is (batch_size, target_seq_len, d_model).
    return x
######################################################################
class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self,
               *,
               d_model,
               num_heads,
               dff,
               dropout_rate=0.1):
    super(DecoderLayer, self).__init__()

    self.causal_self_attention = CausalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.ffn = FeedForward(d_model, dff)

  def call(self, x):
    x = self.causal_self_attention(x=x)
    x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
    return x
######################################################################
class FeedForward(tf.keras.layers.Layer):
  def __init__(self, d_model, dff, dropout_rate=0.1):
    super().__init__()
    self.seq = tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),
      tf.keras.layers.Dense(d_model),
      tf.keras.layers.Dropout(dropout_rate)
    ])
    self.add = tf.keras.layers.Add()
    self.layer_norm = tf.keras.layers.LayerNormalization()

  def call(self, x):
    x = self.add([x, self.seq(x)])
    x = self.layer_norm(x) 
    return x
######################################################################
class BaseAttention(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
    self.layernorm = tf.keras.layers.LayerNormalization()
    self.add = tf.keras.layers.Add()
######################################################################
class CausalSelfAttention(BaseAttention):
  def call(self, x):
    attn_output = self.mha(
        query=x,
        value=x,
        key=x,
        use_causal_mask = True)
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x
######################################################################
class PositionalEmbedding(tf.keras.layers.Layer):
  '''
  tutorial implementation does embedding AND position embedding here
  we will do embedding before, so just position embedding here
  '''
  def __init__(self,  d_model):
    super().__init__()
    self.d_model = d_model    
    self.pos_encoding = positional_encoding(length=2048, 
                                            depth=d_model)
  
  def call(self, x):
    length = tf.shape(x)[1]
    # This factor sets the relative scale of the embedding and positonal_encoding.
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x = x + self.pos_encoding[tf.newaxis, :length, :]
    return x
######################################################################  
def positional_encoding(length, depth):
  depth = depth/2

  positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
  depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

  angle_rates = 1 / (10000**depths)         # (1, depth)
  angle_rads = positions * angle_rates      # (pos, depth)

  pos_encoding = np.concatenate(
      [np.sin(angle_rads), np.cos(angle_rads)],
      axis=-1) 

  return tf.cast(pos_encoding, dtype=tf.float32)

###################################
class GlobalSelfAttention(tf.keras.layers.Layer):
  def __init__(self,num_heads,key_dim,dropout):
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads,
                                                  key_dim=key_dim,
                                                  dropout=dropout)
    self.layernorm = tf.keras.layers.LayerNormalization()
    self.add = tf.keras.layers.Add()
  def call(self, x,attn_mask=None):    
    attn_mask = self.mycompute_mask(attn_mask)
    attn_output = self.mha(
        query=x,
        value=x,
        key=x,
        attention_mask=attn_mask)
    #print('type on mha', type(self.mha))
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x
  def mycompute_mask(self,tmask):
    ## this is a rip from how MultiHeadAttention calcs based on keras masks
    ## https://github.com/keras-team/keras/blob/v2.14.0/keras/layers/attention/multi_head_attention.py#L130-L731
    query_mask = tmask#getattr(query, "_keras_mask", None)
    value_mask = tmask#getattr(value, "_keras_mask", None)
    key_mask = tmask#None#getattr(key, "_keras_mask", None)
    auto_mask = None
    attention_mask=None
    if query_mask is not None:
        query_mask = tf.cast(query_mask, tf.bool)  # defensive casting
        # B = batch size, T = max query length
        auto_mask = query_mask[:, :, tf.newaxis]  # shape is [B, T, 1]
    if value_mask is not None:
        value_mask = tf.cast(value_mask, tf.bool)  # defensive casting
        # B = batch size, S == max value length
        mask = value_mask[:, tf.newaxis, :]  # shape is [B, 1, S]
        auto_mask = mask if auto_mask is None else auto_mask & mask
    if key_mask is not None:
        key_mask = tf.cast(key_mask, tf.bool)  # defensive casting
        # B == batch size, S == max key length == max value length
        mask = key_mask[:, tf.newaxis, :]  # shape is [B, 1, S]
        auto_mask = mask if auto_mask is None else auto_mask & mask
    if auto_mask is not None:
        # merge attention_mask & automatic mask, to shape [B, T, S]
        attention_mask = (
            auto_mask
            if attention_mask is None
            else tf.cast(attention_mask, bool) & auto_mask
        )
    return(attention_mask)





class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self,*, d_model, num_heads, dff, dropout_rate=0.1):
    super().__init__()

    self.self_attention = GlobalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.ffn = FeedForward(d_model, dff)



  def call(self, x,attn_mask):
    x = self.self_attention(x,attn_mask=attn_mask)
    x = self.ffn(x)
    return x
  
