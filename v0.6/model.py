from Config import Config
from helper import Vocab

import TfUtils
import helper
import myRNN

import tensorflow as tf
import numpy as np
import sys
import os
import argparse
import logging
import cPickle as pkl

args=None
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="training options")
    
    parser.add_argument('--load-config', action='store_true', dest='load_config', default=False)
    parser.add_argument('--weight-path', action='store', dest='weight_path', required=True)

    parser.add_argument('--train-test', action='store', dest='train_test', default='train', choices=['train', 'test'])
    
    args = parser.parse_args()

class PtrNet(object):
  
    def __init__(self, args=args, test=False):
        self.vocab = Vocab()
        self.config=Config()
        
        self.weight_path = args.weight_path
            
        if args.load_config == False:
            self.config.saveConfig(self.weight_path+'/config')
            print 'default configuration generated, please specify --load-config and run again.'
            sys.exit()
        else:
            self.config.loadConfig(self.weight_path+'/config')
        
        self.step_p_epoch = self.load_data(test)
        
        self.add_placeholders()
        self.add_embedding()
        self.fetch_input()

        train_loss, valid_loss, self.prediction = self.add_model()

        self.train_op = self.add_train_op(train_loss)
        self.loss = valid_loss
        self.reg_loss = train_loss - valid_loss
        
        MyVars = [v for v in tf.trainable_variables()]
        MyVars_name = [v.name for v in MyVars]
        self.MyVars = MyVars
        print MyVars_name

    def train_module(self, cell_dec, encoder_inputs,
                     enc_lengths, dec_lengths, order_index, scope=None):
        '''
        Args:
            cell_dec : lstm cell object, a configuration
            encoder_inputs : shape(b_sz, tstp_enc, s_emb_sz)
            enc_lengths : shape(b_sz,), encoder input lengths
            dec_lengths : shape(b_sz), decoder input lengths
            order_index : shape(b_sz, tstp_dec), decoder label

        '''
        small_num = -np.Inf
        input_shape = tf.shape(encoder_inputs)
        b_sz = input_shape[0]
        tstp_enc = input_shape[1]
        tstp_dec = tstp_enc  # since no noise, time step of decoder should be the same as encoder
        h_enc_sz = self.config.h_enc_sz
        h_dec_sz = self.config.h_dec_sz
        s_emb_sz = np.int(encoder_inputs.get_shape()[2])  # should be a python-determined number

        cell_enc = tf.nn.rnn_cell.BasicLSTMCell(self.config.h_enc_sz)

        def enc(dec_h, in_x, lengths, fake_call=False):
            '''
            Args:
                dec_h: shape(b_sz, tstp_dec, h_dec_sz)
                in_x: shape(b_sz, tstp_enc, s_emb_sz)
                lengths: shape(b_sz)
            Returns:
                res: shape(b_sz, tstp_dec, tstp_enc, Ptr_sz)
            '''

            def func_f(in_x, enc_h, in_h_hat, fake_call=False):
                '''
                Args:
                    in_x: shape(b_sz, tstp_dec, tstp_enc, enc_emb_sz)
                    in_h: shape(b_sz, tstp_dec, tstp_enc, h_enc_sz*2)
                Returns:
                    res: shape(b_sz, tstp_dec, tstp_enc, enc_emb_sz+h_enc_sz*2)

                '''
                if fake_call:
                    return s_emb_sz + h_enc_sz*4

                in_x_sz = int(in_x.get_shape()[-1])
                in_h_sz = int(enc_h.get_shape()[-1])
                if not in_x_sz:
                    assert ValueError('last dimension of the first' +
                                      ' arg should be known, while got %s'
                                      % (str(type(in_x_sz))))
                if not in_h_sz:
                    assert ValueError('last dimension of the second' +
                                      ' arg should be known, while got %s'
                                      % (str(type(in_h_sz))))
                enc_in_ex = tf.expand_dims(in_x, 1)  # shape(b_sz, 1, tstp_enc, s_emb_sz)
                enc_in = tf.tile(enc_in_ex,  # shape(b_sz, tstp_dec, tstp_enc, s_emb_sz)
                                 [1, tstp_dec, 1, 1])
                res = tf.concat(3, [enc_in, enc_h, in_h_hat])
                return res      # shape(b_sz, tstp_dec, tstp_enc, enc_emb_sz+h_enc_sz*4)

            def attend(enc_h, enc_len):
                '''
                Args:
                    enc_h: shape(b_sz, tstp_dec, tstp_enc, h_enc_sz*2)
                    enc_len: shape(b_sz)
                '''
                enc_len = tf.expand_dims(enc_len, 1)    # shape(b_sz, 1)
                attn_enc_len = tf.tile(enc_len, [1, tstp_dec])
                attn_enc_len = tf.reshape(attn_enc_len, [b_sz*tstp_dec])
                attn_enc_h = tf.reshape(enc_h,          # shape(b_sz*tstp_dec, tstp_enc, h_enc_sz*2)
                                        [b_sz*tstp_dec, tstp_enc,
                                         np.int(enc_h.get_shape()[-1])])
                attn_out = TfUtils.self_attn(  # shape(b_sz*tstp_dec, tstp_enc, h_enc_sz*2)
                    attn_enc_h, attn_enc_len)
                h_hat = tf.reshape(attn_out,   # shape(b_sz, tstp_dec, tstp_enc, h_enc_sz*2)
                                   [b_sz, tstp_dec,
                                    tstp_enc,
                                    np.int(attn_out.get_shape()[-1])])
                return h_hat

            if fake_call:
                return func_f(None, None, None, fake_call=True)
            def get_lstm_in_len():
                inputs = func_enc_input(dec_h, in_x)    # shape(b_sz, tstp_dec, tstp_enc, enc_emb_sz)
                enc_emb_sz = np.int(inputs.get_shape()[-1])
                enc_in = tf.reshape(inputs,
                                    shape=[b_sz*tstp_dec, tstp_enc, enc_emb_sz])
                enc_len = tf.expand_dims(lengths, 1)        # shape(b_sz, 1)
                enc_len = tf.tile(enc_len, [1, tstp_dec])   # shape(b_sz, tstp_dec)
                enc_len = tf.reshape(enc_len, [b_sz*tstp_dec])    # shape(b_sz*tstp_dec,)
                return enc_in, enc_len

            '''shape(b_sz*tstp_dec, tstp_enc, enc_emb_sz), shape(b_sz*tstp_dec)'''
            enc_in, enc_len = get_lstm_in_len()

            '''tup(shpae(b_sz*tstp_dec, tstp_enc, h_enc_sz))'''
            lstm_out, _ = tf.nn.bidirectional_dynamic_rnn(cell_enc,
                                                          cell_enc,
                                                          enc_in,
                                                          enc_len,
                                                          swap_memory=True,
                                                          dtype=tf.float32,
                                                          scope='sent_encoder')
            enc_out = tf.concat(2, lstm_out)    # shape(b_sz*tstp_dec, tstp_enc, h_enc_sz*2)
            enc_out = tf.reshape(enc_out,       # shape(b_sz, tstp_dec, tstp_enc, h_enc_sz*2)
                                 shape=[b_sz, tstp_dec,
                                        tstp_enc, h_enc_sz*2])

            enc_out_hat = attend(enc_out, lengths)
            res = func_f(in_x, enc_out, enc_out_hat)
            return res  # shape(b_sz, tstp_dec, tstp_enc, Ptr_sz)

        def func_enc_input(dec_h, enc_input, fake_call=False):
            '''
            Args:
                enc_input: encoder input, shape(b_sz, tstp_enc, s_emb_sz)
                dec_h: decoder hidden state, shape(b_sz, tstp_dec, h_dec_sz)
            Returns:
                output: shape(b_sz, tstp_dec, tstp_enc, s_emb_sz+h_dec_sz)
            '''
            enc_emb_sz = s_emb_sz + h_dec_sz
            if fake_call:
                return enc_emb_sz

            dec_h_ex = tf.expand_dims(dec_h, 2)  # shape(b_sz, tstp_dec, 1, h_dec_sz)
            dec_h_tile = tf.tile(dec_h_ex,       # shape(b_sz, tstp_dec, tstp_enc, h_dec_sz)
                                 [1, 1, tstp_enc, 1])
            enc_in_ex = tf.expand_dims(enc_input, 1)    # shape(b_sz, 1, tstp_enc, s_emb_sz)
            enc_in_tile = tf.tile(enc_in_ex,            # shape(b_sz, tstp_dec, tstp_enc, s_emb_sz)
                                  [1, tstp_dec, 1, 1])
            output = tf.concat(3,                       # shape(b_sz, tstp_dec, tstp_enc, s_emb_sz+h_dec_sz)
                               [enc_in_tile, dec_h_tile])

            output = tf.reshape(output, shape=[b_sz, tstp_dec, tstp_enc, s_emb_sz + h_dec_sz])
            return output  # shape(b_sz, tstp_dec, tstp_enc, s_emb_sz+h_dec_sz)

        def func_point_logits(dec_h, enc_ptr, enc_len):
            '''
            Args:
                dec_h : shape(b_sz, tstp_dec, h_dec_sz)
                enc_ptr : shape(b_sz, tstp_dec, tstp_enc, Ptr_sz)
                enc_len : shape(b_sz,)
            '''
            dec_h_ex = tf.expand_dims(dec_h, dim=2)     # shape(b_sz, tstp_dec, 1, h_dec_sz)
            dec_h_ex = tf.tile(dec_h_ex, [1, 1, tstp_enc, 1])  # shape(b_sz, tstp_dec, tstp_enc, h_dec_sz)
            linear_concat = tf.concat(3, [dec_h_ex, enc_ptr])  # shape(b_sz, tstp_dec, tstp_enc, h_dec_sz+ Ptr_sz)
            point_linear = TfUtils.last_dim_linear(  # shape(b_sz, tstp_dec, tstp_enc, h_dec_sz)
                linear_concat, output_size=h_dec_sz,
                bias=False, scope='Ptr_W')
            point_v = TfUtils.last_dim_linear(  # shape(b_sz, tstp_dec, tstp_enc, 1)
                tf.tanh(point_linear), output_size=1,
                bias=False, scope='Ptr_V')

            point_logits = tf.squeeze(point_v, squeeze_dims=[3])  # shape(b_sz, tstp_dec, tstp_enc)

            enc_len = tf.expand_dims(enc_len, 1)                # shape(b_sz, 1)
            enc_len = tf.tile(enc_len, [1, tstp_dec])           # shape(b_sz, tstp_dec)
            mask = TfUtils.mkMask(enc_len, maxLen=tstp_enc)     # shape(b_sz, tstp_dec, tstp_enc)
            point_logits = tf.select(mask, point_logits,        # shape(b_sz, tstp_dec, tstp_enc)
                                     tf.ones_like(point_logits) * small_num)


            return point_logits

        def get_initial_state(hidden_sz):
            '''
            Args:
                hidden_sz: must be a python determined number
            '''
            avg_in_x = TfUtils.reduce_avg(encoder_inputs,           # shape(b_sz, s_emb_sz)
                                          enc_lengths, dim=1)
            state = tf.nn.rnn_cell._linear(avg_in_x, hidden_sz,     # shape(b_sz, hidden_sz)
                                           bias=False,
                                           scope='initial_transformation')
            state = tf.nn.rnn_cell.LSTMStateTuple(state, tf.zeros_like(state))
            return state

        def get_bos(emb_sz):
            with tf.variable_scope('bos_scope') as vscope:
                try:
                    ret = tf.get_variable(name='bos', shape=[1, emb_sz], dtype=tf.float32)
                except:
                    vscope.reuse_variables()
                    ret = tf.get_variable(name='bos', shape=[1, emb_sz], dtype=tf.float32)
            ret_bos = tf.tile(ret, [b_sz, 1])
            return ret_bos

        def decoder():
            def get_dec_in():
                dec_in = TfUtils.batch_embed_lookup(encoder_inputs, order_index)    # shape(b_sz, tstp_dec, s_emb_sz)
                bos = get_bos(s_emb_sz)                                             # shape(b_sz, s_emb_sz)
                bos = tf.expand_dims(bos, 1)                                        # shape(b_sz, 1, s_smb_sz)
                dec_in = tf.concat(1, [bos, dec_in])                                # shape(b_sz, tstp_dec+1, s_emb_sz)
                dec_in = dec_in[:, :-1, :]                                          # shape(b_sz, tstp_dec, s_emb_sz)
                return dec_in

            dec_in = get_dec_in()                               # shape(b_sz, tstp_dec, s_emb_sz)
            initial_state = get_initial_state(h_dec_sz)         # shape(b_sz, h_dec_sz)
            dec_out, _ = tf.nn.dynamic_rnn(cell_dec, dec_in,    # shape(b_sz, tstp_dec, h_dec_sz)
                                           dec_lengths,
                                           initial_state=initial_state,
                                           swap_memory=True,
                                           dtype=tf.float32,
                                           scope=scope)
            with tf.variable_scope(scope):
                enc_out = enc(dec_out,      # shape(b_sz, tstp_dec, tstp_enc, Ptr_sz)
                              encoder_inputs,
                              enc_lengths)
                point_logits = func_point_logits(dec_out, enc_out, enc_lengths)     # shape(b_sz, tstp_dec, tstp_enc)
            return point_logits

        point_logits = decoder()    # shape(b_sz, tstp_dec, tstp_enc)
        return point_logits

    def decoder_test(self, cell_dec, encoder_inputs,
                      enc_lengths, dec_lengths, scope=None):
        '''
        Args:
            cell_dec : lstm cell object, a configuration
            encoder_inputs : shape(b_sz, tstp_enc, s_emb_sz)
            enc_lengths : shape(b_sz,), encoder input lengths
            dec_lengths : shape(b_sz), decoder input lengths
            order_index : shape(b_sz, tstp_dec), decoder label

        '''

        small_num = -np.Inf
        input_shape = tf.shape(encoder_inputs)
        b_sz = input_shape[0]
        tstp_enc = input_shape[1]
        tstp_dec = tstp_enc  # since no noise, time step of decoder should be the same as encoder
        h_enc_sz = self.config.h_enc_sz
        h_dec_sz = self.config.h_dec_sz
        s_emb_sz = np.int(encoder_inputs.get_shape()[2])  # should be a python-determined number

        # dec_emb_sz not determined
        cell_enc = tf.nn.rnn_cell.BasicLSTMCell(self.config.h_enc_sz)

        def enc(dec_h, in_x, lengths, fake_call=False):
            '''
            Args:
                inputs: shape(b_sz, tstp_enc, enc_emb_sz)

            '''

            def func_f(in_x, in_h, in_h_hat, fake_call=False):
                if fake_call:
                    return s_emb_sz + h_enc_sz*4

                in_x_sz = int(in_x.get_shape()[-1])
                in_h_sz = int(in_h.get_shape()[-1])
                if not in_x_sz:
                    assert ValueError('last dimension of the first' +
                                      ' arg should be known, while got %s'
                                      % (str(type(in_x_sz))))
                if not in_h_sz:
                    assert ValueError('last dimension of the second' +
                                      ' arg should be known, while got %s'
                                      % (str(type(in_h_sz))))
                res = tf.concat(2, [in_x, in_h, in_h_hat])
                return res

            if fake_call:
                return func_f(None, None, None, fake_call=True)
            inputs = func_enc_input(dec_h, in_x)

            lstm_out, _ = tf.nn.bidirectional_dynamic_rnn(cell_enc,
                                                          cell_enc,
                                                          inputs,
                                                          lengths,
                                                          swap_memory=True,
                                                          dtype=tf.float32,
                                                          scope='sent_encoder')
            enc_out = tf.concat(2, lstm_out)  # shape(b_sz, tstp_enc, h_enc_sz*2)
            enc_out = tf.reshape(enc_out, [b_sz, tstp_enc, h_enc_sz*2])

            enc_out_hat = TfUtils.self_attn(enc_out, lengths)
            res = func_f(in_x, enc_out, enc_out_hat)
            return res  # shape(b_sz, tstp_enc, dec_emb_sz)

        def func_enc_input(dec_h, enc_input, fake_call=False):
            '''
            Args:
                enc_input: encoder input, shape(b_sz, tstp_enc, s_emb_sz)
                dec_h: decoder hidden state, shape(b_sz, h_dec_sz)
            '''
            enc_emb_sz = s_emb_sz + h_dec_sz
            if fake_call:
                return enc_emb_sz

            dec_h_ex = tf.expand_dims(dec_h, 1)  # shape(b_sz, 1, h_dec_sz)
            dec_h_tile = tf.tile(dec_h_ex, [1, tstp_enc, 1])

            output = tf.concat(2, [enc_input, dec_h_tile])  # shape(b_sz, tstp_enc, s_emb_sz + h_dec_sz)
            output = tf.reshape(output, shape=[b_sz, tstp_enc, s_emb_sz + h_dec_sz])
            return output  # shape(b_sz, tstp_enc, s_emb_sz + h_dec_sz)

        enc_emb_sz = func_enc_input(None, None, fake_call=True)
        dec_emb_sz = enc(None, None, None, fake_call=True)

        def func_point_logits(dec_h, enc_e, enc_len):
            '''
            Args:
                dec_h : shape(b_sz, h_dec_sz)
                enc_e : shape(b_sz, tstp_enc, dec_emb_sz)
                enc_len : shape(b_sz,)
            '''

            dec_h_ex = tf.expand_dims(dec_h, dim=1)  # shape(b_sz, 1, h_dec_sz)
            dec_h_ex = tf.tile(dec_h_ex, [1, tstp_enc, 1])  # shape(b_sz, tstp_enc, h_dec_sz)
            linear_concat = tf.concat(2, [dec_h_ex, enc_e])  # shape(b_sz, tstp_enc, h_dec_sz+ dec_emb_sz)
            point_linear = TfUtils.last_dim_linear(  # shape(b_sz, tstp_enc, h_dec_sz)
                linear_concat, output_size=h_dec_sz,
                bias=False, scope='Ptr_W')
            point_v = TfUtils.last_dim_linear(  # shape(b_sz, tstp_enc, 1)
                tf.tanh(point_linear), output_size=1,
                bias=False, scope='Ptr_V')
            point_logits = tf.squeeze(point_v, squeeze_dims=[2])  # shape(b_sz, tstp_enc)
            mask = TfUtils.mkMask(enc_len, maxLen=tstp_enc)  # shape(b_sz, tstp_enc)
            point_logits = tf.select(mask, point_logits,
                                     tf.ones_like(point_logits) * small_num)  # shape(b_sz, tstp_enc)

            return point_logits

        def func_point_idx(dec_h, enc_e, enc_len, hit_mask):
            '''
            Args:
                hit_mask: shape(b_sz, tstp_enc)
            '''
            logits = func_point_logits(dec_h, enc_e, enc_len)  # shape(b_sz, tstp_enc)
            prob = tf.nn.softmax(logits)
            prob = tf.select(hit_mask, tf.zeros_like(prob),
                             prob, name='mask_hit_pos')
            idx = tf.cast(tf.arg_max(prob, dimension=1), dtype=tf.int32)  # shape(b_sz,) type of int32
            return idx  # shape(b_sz,)

        def get_bos(emb_sz):
            with tf.variable_scope('bos_scope') as vscope:
                try:
                    ret = tf.get_variable(name='bos', shape=[1, emb_sz], dtype=tf.float32)
                except:
                    vscope.reuse_variables()
                    ret = tf.get_variable(name='bos', shape=[1, emb_sz], dtype=tf.float32)
            ret_bos = tf.tile(ret, [b_sz, 1])
            return ret_bos

        def get_initial_state(hidden_sz):
            '''
            Args:
                hidden_sz: must be a python determined number
            '''
            avg_in_x = TfUtils.reduce_avg(encoder_inputs,  # shape(b_sz, s_emb_sz)
                                          enc_lengths, dim=1)
            state = tf.nn.rnn_cell._linear(avg_in_x, hidden_sz,  # shape(b_sz, hidden_sz)
                                           bias=False,
                                           scope='initial_transformation')
            state = tf.nn.rnn_cell.LSTMStateTuple(state, tf.zeros_like(state))
            return state

        bos = get_bos(s_emb_sz)  # shape(b_sz, s_emb_sz)

        init_state = get_initial_state(h_dec_sz)

        def loop_fn(time, cell_output, cell_state, hit_mask):
            """
            Args:
                cell_output: shape(b_sz, h_dec_sz) ==> d
                cell_state: tup(shape(b_sz, h_dec_sz))
                pointer_logits_ta: pointer logits tensorArray
                hit_mask: shape(b_sz, tstp_enc)
            """

            if cell_output is None:  # time == 0
                next_cell_state = init_state
                next_input = bos  # shape(b_sz, dec_emb_sz)
                next_idx = tf.zeros(shape=[b_sz],
                                    dtype=tf.int32)  # shape(b_sz, tstp_enc)
                elements_finished = tf.zeros(
                    shape=[b_sz], dtype=tf.bool, name='elem_finished')
                next_hit_mask = tf.zeros(shape=[b_sz, tstp_enc],
                                    dtype=tf.bool, name='hit_mask')
            else:

                next_cell_state = cell_state

                encoder_e = enc(cell_output, encoder_inputs, enc_lengths)  # shape(b_sz, tstp_enc, dec_emb_sz)
                next_idx = func_point_idx(cell_output, encoder_e, enc_lengths, hit_mask)  # shape(b_sz,)

                cur_hit_mask = tf.one_hot(next_idx, on_value=True,      # shape(b_sz, tstp_enc)
                                      off_value=False, depth=tstp_enc,
                                      dtype=tf.bool)
                next_hit_mask = tf.logical_or(hit_mask, cur_hit_mask,   # shape(b_sz, tstp_enc)
                                              name='next_hit_mask')

                next_input = TfUtils.batch_embed_lookup(encoder_inputs, next_idx)  # shape(b_sz, s_emb_sz)

                elements_finished = (time >= dec_lengths)  # shape(b_sz,)

            return (elements_finished, next_input,
                    next_cell_state, next_hit_mask, next_idx)

        emit_idx_ta, _ = myRNN.train_rnn(cell_dec, loop_fn, scope=scope)
        output_idx = emit_idx_ta.pack()  # shape(tstp_dec, b_sz)
        output_idx = tf.transpose(output_idx, perm=[1, 0])  # shape(b_sz, tstp_dec)

        return output_idx  # shape(b_sz, tstp_dec)

    def add_placeholders(self):
        self.ph_encoder_input = tf.placeholder(tf.int32, (None, None, None), name='ph_encoder_input') #(batch_size, tstps_en, max_len_sentence)
        self.ph_decoder_label = tf.placeholder(tf.int32, (None, None), name='ph_decoder_label') #(b_sz, tstps_dec)
        self.ph_input_encoder_len = tf.placeholder(tf.int32, (None,), name='ph_input_encoder_len') #(batch_size)
        self.ph_input_decoder_len = tf.placeholder(tf.int32, (None,), name='ph_input_decoder_len') #(batch_size)
        self.ph_input_encoder_sentence_len = tf.placeholder(tf.int32, (None, None), name='ph_input_encoder_sentence_len') #(batch_size, tstps_en)
        
        self.ph_dropout = tf.placeholder(tf.float32, name='ph_dropout')
    
    def add_embedding(self):
        if self.config.pre_trained:
            embed_dic = helper.readEmbedding(self.config.embed_path+str(self.config.embed_size))  #embedding.50 for 50 dim embedding
            embed_matrix = helper.mkEmbedMatrix(embed_dic, self.vocab.word_to_index)
            self.embedding = tf.Variable(embed_matrix, name='Embedding')
        else:
            self.embedding = tf.get_variable('Embedding', [len(self.vocab), self.config.embed_size], trainable=True)
    
    def fetch_input(self):
        b_sz = tf.shape(self.ph_encoder_input)[0]
        tstps_en = tf.shape(self.ph_encoder_input)[1]
        tstps_de = tf.shape(self.ph_decoder_label)[1]
        emb_sz = self.config.embed_size
        h_sz = self.config.h_rep_sz
        
        def lstm_sentence_rep(input):
            with tf.variable_scope('lstm_sentence_rep_scope') as scope:
                input = tf.reshape(input, shape=[b_sz*tstps_en, -1, emb_sz]) #(b_sz*tstps_en, len_sen, emb_sz)
                length = tf.reshape(self.ph_input_encoder_sentence_len, shape=[-1]) #(b_sz*tstps_en)
                
                lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(h_sz)

                """tup(shape(b_sz*tstp_enc, len_sen, h_sz))"""
                rep_out, _ = tf.nn.bidirectional_dynamic_rnn(     # tup(shape(b_sz*tstp_enc, len_sen, h_sz))
                    lstm_cell, lstm_cell, input, length, dtype=tf.float32,
                    swap_memory=True, time_major=False, scope = 'sentence_encode')

                rep_out = tf.concat(2, rep_out) #(b_sz*tstps_en, len_sen, h_sz*2)
                rep_out = TfUtils.reduce_avg(rep_out, length, dim=1)    # shape(b_sz*tstps_en, h_sz*2)
                output = tf.reshape(rep_out, shape=[b_sz, tstps_en, 2*h_sz]) #(b_sz, tstps_en, h_sz*2)
                
            return output, None, None
            
        def cnn_sentence_rep(input):
            # input (batch_size, tstps_en, len_sentence, embed_size)
            input = tf.reshape(input, shape=[b_sz*tstps_en, -1, emb_sz]) #(b_sz*tstps_en, len_sen, emb_sz)
            length = tf.reshape(self.ph_input_encoder_sentence_len, shape=[-1]) #(b_sz*tstps_en)
            
            filter_sizes = self.config.filter_sizes
            
            in_channel = self.config.embed_size
            out_channel = self.config.num_filters
            
            def convolution(input, tstps_en, length):
                len_sen = tf.shape(input)[1]
                conv_outputs = []
                for i, filter_size in enumerate(filter_sizes):
                    with tf.variable_scope("conv-%s" % filter_size):
                        filter_shape = [filter_size, in_channel, out_channel]
                        W = tf.get_variable(name='W', shape=filter_shape)
                        b = tf.get_variable(name='b', shape=[out_channel])
                        conv = tf.nn.conv1d(                # size (b_sz* tstps_en, len_sen, out_channel)
                          input,
                          W,
                          stride=1,
                          padding="SAME",
                          name="conv")
                        # Apply nonlinearity
                        h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                        conv_outputs.append(h)
                input = tf.concat(2, conv_outputs) #(b_sz*tstps_en, len_sen, out_channel * len(filter_sizes))
                
                mask = tf.sequence_mask(length, len_sen, dtype=tf.float32) #(b_sz*tstps_en, len_sen)
                
                pooled = tf.reduce_max(input*tf.expand_dims(mask, 2), [1]) #(b_sz*tstps_en, out_channel*len(filter_sizes))
                
                #size (b_sz, tstps_en, out_channel*len(filter_sizes))
                pooled = tf.reshape(pooled, shape=[b_sz, tstps_en, out_channel*len(filter_sizes)])
    
                return pooled
            
            with tf.variable_scope('cnn_sentence_rep_scope') as scope:
                output = convolution(input, tstps_en, length) #size (b_sz, tstps_en, out_channel*len(filter_sizes))
                
                eos = tf.nn.embedding_lookup(self.embedding, 
                                             tf.ones(shape=(b_sz, 1), dtype=tf.int32)*self.vocab.encode(self.vocab.eos))   #(b_sz, 1, emb_sz)
                scope.reuse_variables()
                eos = convolution(eos, 1, tf.ones([b_sz], dtype=tf.int32)) #size (b_sz, 1, out_channel*len(filter_sizes))
                sos = tf.nn.embedding_lookup(self.embedding, 
                                             tf.ones(shape=(b_sz, 1), dtype=tf.int32)*self.vocab.encode(self.vocab.sos))   #(b_sz, 1, emb_sz)
                sos = convolution(sos, 1, tf.ones([b_sz], dtype=tf.int32)) #size (b_sz, 1, out_channel*len(filter_sizes))
                
            return output, eos, sos
            
        def cbow_sentence_rep(input):
            output = helper.average_sentence_as_vector(input, 
                                              self.ph_input_encoder_sentence_len) #(b_sz, tstp_en, emb_sz)
            eos = tf.nn.embedding_lookup(self.embedding, 
                                     tf.ones(shape=(b_sz, 1), dtype=tf.int32)*self.vocab.encode(self.vocab.eos))   #(b_sz, 1, emb_sz)
            sos = tf.nn.embedding_lookup(self.embedding, 
                                     tf.ones(shape=(b_sz, 1), dtype=tf.int32)*self.vocab.encode(self.vocab.sos))   #(b_sz, 1, emb_sz)
            return output, eos, sos
        
        encoder_input = tf.nn.embedding_lookup(self.embedding, self.ph_encoder_input) #(batch_size, tstps_en, len_sentence, embed_size)
        if self.config.sent_rep == 'lstm':
            encoder_input, eos, sos = lstm_sentence_rep(encoder_input)  # (b_sz, tstp_en, emb_sz)
        elif self.config.sent_rep == 'cnn':
            encoder_input, eos, sos = cnn_sentence_rep(encoder_input)  # (b_sz, tstp_en, emb_sz)
        elif self.config.sent_rep == 'cbow':
            encoder_input, eos, sos = cbow_sentence_rep(encoder_input)  # (b_sz, tstp_en, emb_sz)
        else:
            assert ValueError('sent_rep: ' + self.config.sent_rep)
            exit()
        self.encoder_input = encoder_input

    def add_model(self):
        """
            input_tensor #(batch_size, num_sentence, embed_size)
            input_len    #(batch_size)
        """

        b_sz = tf.shape(self.encoder_input)[0]
        tstp_enc = tf.shape(self.encoder_input)[1]
        tstp_dec = tf.shape(self.ph_decoder_label)[1]

        enc_in = self.encoder_input     # shape(b_sz, tstp_enc, s_emb_sz)
        enc_len = self.ph_input_encoder_len     # shape(b_sz,)
        dec_len = self.ph_input_encoder_len     # shape(b_sz,)
        order_idx = self.ph_decoder_label       # shape(b_sz, tstp_dec)

        cell_dec = tf.nn.rnn_cell.BasicLSTMCell(self.config.h_dec_sz)
        with tf.variable_scope('add_model') as vscope:
            out_logits = self.train_module(     # shape(b_sz, tstp_dec, tstp_enc)
                cell_dec, enc_in, enc_len,
                dec_len, order_idx,
                scope='decoder_train')
            vscope.reuse_variables()
            predict_idx = self.decoder_test(
                cell_dec, enc_in, enc_len,
                dec_len, scope='decoder_train')   # shape(b_sz, tstp_dec)

        train_loss, valid_loss = self.add_loss_op(out_logits, order_idx, dec_len)


        return train_loss, valid_loss, predict_idx
    
    def add_loss_op(self, logits, sparse_label, dec_lengths):
        reg_loss = tf.add_n([tf.nn.l2_loss(v) for v in
                             tf.trainable_variables()
                             if v != self.embedding]) * self.config.reg

        valid_loss = TfUtils.seq_loss(logits, sparse_label, dec_lengths)
        train_loss = reg_loss + valid_loss
        return train_loss, valid_loss
    
    def add_train_op(self, loss):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        self.learning_rate = tf.train.exponential_decay(self.config.lr, global_step,
                                                   int(self.config.decay_epoch * self.step_p_epoch), self.config.decay_rate, staircase=True)
        #optimizer = tf.train.AdamOptimizer(self.learning_rate)
        optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op
    
    """data related"""
    def load_data(self, test):
        self.vocab.load_vocab_from_file(self.config.vocab_path)
        
        self.train_data = helper.load_data(self.config.train_data)
        self.val_data = helper.load_data(self.config.val_data)

        self.train_data_flatten_list = [j for i in self.train_data for j in i]

        step_p_epoch = len(self.train_data) // self.config.batch_size
        if test:
            self.test_data = helper.load_data(self.config.test_data)
            step_p_epoch = 0
        return step_p_epoch

    def create_feed_dict(self, input_batch, sent_len, encoder_len, label_batch=None, decoder_len=None, mode='train'):
        """
        note that the order of value in input_batch tuple matters 
        Args
            input_batch, tuple (encoder_input, decoder_input, decoder_label)
            encoder_len, a length list shape of (batch_size)
            decoder_len, a length list shape of (batch_size+1) with one more word <sos> or <eos>
        Returns
            feed_dict: a dictionary that have elements
        """
        if mode == 'train':
            placeholders = (self.ph_encoder_input, self.ph_input_encoder_sentence_len, self.ph_decoder_label, 
                            self.ph_input_encoder_len, self.ph_input_decoder_len, self.ph_dropout)
            data_batch = (input_batch, sent_len, label_batch, encoder_len, decoder_len, self.config.dropout)
        elif mode == 'predict':
            placeholders = (self.ph_encoder_input, self.ph_input_encoder_sentence_len, self.ph_input_encoder_len, self.ph_dropout)
            data_batch = (input_batch, sent_len, encoder_len, self.config.dropout)
        
        feed_dict = dict(zip(placeholders, data_batch))
        
        return feed_dict

    def run_epoch(self, sess, input_data, verbose=None):
        """
        Runs an epoch of training.

        Trains the model for one-epoch.

        Args:
            sess: tf.Session() object
            input_data: tuple of (encode_input, decode_input, decode_label)
        Returns:
            avg_loss: scalar. Average minibatch loss of model on epoch.
        """
        data_len = len(input_data)
        total_steps =data_len // self.config.batch_size
        total_loss = []
        for step, (b_data, b_order) in enumerate(
                                    helper.data_iter(input_data, self.config.batch_size)):
            order_indices = b_order
            losses = []
            for i in range(self.config.processing_step):
                (ret_batch, ret_label, sent_num_enc, sent_num_dec, sent_len
                 ) = helper.shuffleData(b_data, order_indices, self.vocab)
                feed_dict = self.create_feed_dict(ret_batch, sent_len, sent_num_enc, ret_label, sent_num_dec)
                _, loss, lr, pred = sess.run([self.train_op, self.loss, self.learning_rate, self.prediction], feed_dict=feed_dict)
                pred = pred.tolist()
                order_indices = helper.reorder(order_indices, pred, sent_num_dec)
                losses.append(loss)
            total_loss.append(np.mean(losses))
            if verbose and step % verbose == 0:
                sys.stdout.write('\r{} / {} : loss = {}, lr = {}'.format(
                    step, total_steps, np.mean(total_loss[-verbose:]), lr))
                sys.stdout.flush()
        sys.stdout.write('\n')
        avg_loss = np.mean(total_loss)
        return avg_loss
    
    def fit(self, sess, input_data, verbose=None):
        """
        Runs an epoch of validation or test. return test error

        Args:
            sess: tf.Session() object
            input_data: tuple of (encode_input, decode_input, decode_label)
        Returns:
            avg_loss: scalar. Average minibatch loss of model on epoch.
        """
        total_loss = []
        for step, (b_data, b_order) in enumerate(
                helper.data_iter(input_data, self.config.batch_size)):
            order_indices = b_order
            losses = []
            for i in range(self.config.processing_step):
                (ret_batch, ret_label, sent_num_enc, sent_num_dec, sent_len
                 ) = helper.shuffleData(b_data, order_indices, self.vocab)
                feed_dict = self.create_feed_dict(ret_batch, sent_len, sent_num_enc, ret_label, sent_num_dec)
                loss, pred = sess.run([self.loss, self.prediction], feed_dict=feed_dict)
                pred = pred.tolist()
                order_indices = helper.reorder(order_indices, pred, sent_num_dec)
                losses.append(loss)
            total_loss.append(np.mean(losses))
        avg_loss = np.mean(total_loss)
        return avg_loss
    
    def predict(self, sess, input_data, verbose=None):
        preds = []
        true_label = []
        lengths = []
        for _, (b_data, b_order) in enumerate(helper.data_iter(input_data, self.config.batch_size)):
            order_indices = b_order
            pred = None
            ret_label = None
            sent_num_dec = None
            for i in range(self.config.processing_step):
                (ret_batch, ret_label, sent_num_enc, sent_num_dec, sent_len
                ) = helper.shuffleData(b_data, order_indices, self.vocab)
                feed_dict = self.create_feed_dict(ret_batch, sent_len, sent_num_enc, ret_label, sent_num_dec)
                pred = sess.run(self.prediction, feed_dict=feed_dict)
                pred = pred.tolist()
                order_indices = helper.reorder(order_indices, pred, sent_num_dec)

            preds += pred
            true_label += ret_label.tolist()
            lengths += sent_num_dec

        return preds, true_label, lengths

def test_case(sess, model, data, onset='VALIDATION'):
    """pred must be list"""
    def pad_list(lst, pad=-1):
        inner_max_len = max(map(len, lst))
        map(lambda x: x.extend([pad]*(inner_max_len-len(x))), lst)
        return np.array(lst)

    print '#'*20, 'ON '+onset+' SET START ', '#'*20
    loss = model.fit(sess, data)
    pred, true_label, lengths = model.predict(sess, data)

    true_label = pad_list(true_label)
    true_label = np.array(true_label)
    pred = pad_list(pred, pad=0)
    pred = np.array(pred)

    true_label = true_label.tolist()
    pred = pred.tolist()
    accuracy = helper.calculate_accuracy_seq(pred, true_label, lengths)
    helper.print_pred_seq(pred[:10], true_label[:10])
    
    print 'Overall '+onset+' loss is: {}'.format(loss)
    print 'Overall '+onset+' accuracy is: {}'.format(accuracy)
    logging.info('Overall '+onset+' loss is: {}'.format(loss))
    logging.info('Overall '+onset+' accuracy is: {}'.format(accuracy))
    print '#'*20, 'ON '+onset+' SET END ', '#'*20
    
    # return loss, pred, true_label, accuracy
    return loss, pred, true_label, lengths

def train_run():
    logging.info('Training start')
    model = PtrNet()
    saver = tf.train.Saver()

    config=tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    with tf.Session(config=config) as sess:

        best_loss = np.Inf
        best_val_epoch = 0
        sess.run(tf.initialize_all_variables())

        if os.path.exists(model.weight_path+'/checkpoint'):
            saver.restore(sess, model.weight_path+'/parameter.weight')

        for epoch in range(model.config.max_epochs):
            print "="*20+"Epoch ", epoch, "="*20
            loss = model.run_epoch(sess, model.train_data, verbose=1)

            print "Mean loss in this epoch is: ", loss
            logging.info('%s %d%s' % ('='*20+'Epoch', epoch, '='*20))
            logging.info("Mean loss in {}th epoch is: {}".format(epoch, loss))

            val_loss, _, _, accu = test_case(sess, model, model.val_data, onset='VALIDATION')

            if best_loss > val_loss:
                best_loss = val_loss
                best_val_epoch = epoch
                if not os.path.exists(model.weight_path):
                    os.makedirs(model.weight_path)

                saver.save(sess, model.weight_path+'/parameter.weight')
                print 'saved!'+'=='*20
            if epoch - best_val_epoch > model.config.early_stopping:
                logging.info("Normal Early stop")
                break

    logging.info("Training complete")

def test_run():
    model = PtrNet(test='test')
    saver = tf.train.Saver()

    config=tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    with tf.Session(config=config) as sess:
        sess.run(tf.initialize_all_variables())
        saver.restore(sess, model.weight_path+'/parameter.weight')
        _, pred, true_label, lengths = test_case(sess, model, model.test_data, onset='TEST')
    with open(model.weight_path+'/pred_true_label', 'wb') as fd:
        pkl.dump(pred, fd)
        pkl.dump(true_label, fd)
        pkl.dump(lengths, fd)

def main(_):
    np.random.seed(1234)
    tf.set_random_seed(5678)
    if not os.path.exists(args.weight_path):
        os.makedirs(args.weight_path)
    logFile = args.weight_path+'/run.log'

    if args.train_test == "train":
        
        try:
            os.remove(logFile)
        except OSError:
            pass
        logging.basicConfig(filename=logFile, format='%(levelname)s %(asctime)s %(message)s', level=logging.INFO)
        train_run()
    else:
        logging.basicConfig(filename=logFile, format='%(levelname)s %(asctime)s %(message)s', level=logging.INFO)
        test_run()

if __name__ == '__main__':
    tf.app.run()
