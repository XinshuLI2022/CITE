
#coding=utf-8
import tensorflow as tf
import numpy as np

from .util import *

class cite_net(object):


    def __init__(self, x, t, y_ , p_t, FLAGS, p_alpha, p_lambda, do_in, do_out, dims,x_pos,x_neg,temp,batch_size):
        self.variables = {}
        self.wd_loss = 0

        if FLAGS.nonlin.lower() == 'elu':
            self.nonlin = tf.nn.elu
        else:
            self.nonlin = tf.nn.relu

        self._build_graph(x, t, y_ , p_t, FLAGS, p_alpha, p_lambda, do_in, do_out, dims,x_pos,x_neg,temp,batch_size)

    def _add_variable(self, var, name):
     
        basename = name
        i = 0
        while name in self.variables:
            name = '%s_%d' % (basename, i) 
            i += 1

        self.variables[name] = var

    def _create_variable(self, var, name):


        var = tf.Variable(var, name=name)
        self._add_variable(var, name)
        return var

    def _create_variable_with_weight_decay(self, initializer, name, wd):
 
        var = self._create_variable(initializer, name)
        self.wd_loss += wd*tf.nn.l2_loss(var)
        return var

    def _build_graph(self, x, t, y_ , p_t, FLAGS, p_alpha, p_lambda, do_in, do_out, dims,x_pos,x_neg,temp,batch_size):
     

        self.x = x
        self.t = t
        self.y_ = y_
        self.p_t = p_t
        self.p_alpha = p_alpha
        self.p_lambda = p_lambda
        self.do_in = do_in
        self.do_out = do_out
        self.x_pos = x_pos
        self.x_neg = x_neg
        self.temp = temp


        dim_input = dims[0]
        dim_in = dims[1]
        dim_out = dims[2]
        dim_cl_1 = dims[3]
        dim_cl_2 = dims[4]
        batch_size = batch_size

        weights_in = []; biases_in = []


        if FLAGS.n_in == 0:
            dim_in = dim_input
        if FLAGS.n_out == 0:
            if FLAGS.split_output == False:
                dim_out = dim_in+1
            else:
                dim_out = dim_in

        if FLAGS.batch_norm:
            bn_biases = []
            bn_scales = []
            bn_biases_pos = []
            bn_scales_pos = []
            bn_biases_neg = []
            bn_scales_neg = []


        ''' Construct input/representation layers '''
        h_in = [x]
        h_in_pos = [x_pos]
        h_in_neg = [x_neg]
      
        
        # representation layer
        for i in range(0, FLAGS.n_in):
            if i==0:
                weights_in.append(tf.Variable(tf.random_normal([dim_input, dim_in], stddev=FLAGS.weight_init/np.sqrt(dim_input))))
            else:
                weights_in.append(tf.Variable(tf.random_normal([dim_in,dim_in], stddev=FLAGS.weight_init/np.sqrt(dim_in))))
            
          

            biases_in.append(tf.Variable(tf.zeros([1,dim_in])))
            z = tf.matmul(h_in[i], weights_in[i]) + biases_in[i]
            z_pos = tf.matmul(h_in_pos[i], weights_in[i]) + biases_in[i]
            z_neg = tf.matmul(h_in_neg[i], weights_in[i]) + biases_in[i]


            if FLAGS.batch_norm:
                batch_mean, batch_var = tf.nn.moments(z, [0])
                batch_mean_pos, batch_var_pos = tf.nn.moments(z_pos, [0])
                batch_mean_neg, batch_var_neg = tf.nn.moments(z_neg, [0])

                if FLAGS.normalization == 'bn_fixed':
                    z = tf.nn.batch_normalization(z, batch_mean, batch_var, 0, 1, 1e-3)
                    z_pos = tf.nn.batch_normalization(z_pos, batch_mean_pos, batch_var_pos, 0, 1, 1e-3)
                    z_neg = tf.nn.batch_normalization(z_neg, batch_mean_neg, batch_var_neg, 0, 1, 1e-3)

                else:
                    bn_biases.append(tf.Variable(tf.zeros([dim_in])))
                    bn_scales.append(tf.Variable(tf.ones([dim_in])))
                    bn_biases_pos.append(tf.Variable(tf.zeros([dim_in])))
                    bn_scales_pos.append(tf.Variable(tf.ones([dim_in])))
                    bn_biases_neg.append(tf.Variable(tf.zeros([dim_in])))
                    bn_scales_neg.append(tf.Variable(tf.ones([dim_in])))
                    z = tf.nn.batch_normalization(z, batch_mean, batch_var, bn_biases[-1], bn_scales[-1], 1e-3)
                    z_pos = tf.nn.batch_normalization(z_pos, batch_mean_pos, batch_var_pos, bn_biases_pos[-1], bn_scales_pos[-1], 1e-3) 
                    z_neg = tf.nn.batch_normalization(z_neg, batch_mean_neg, batch_var_neg, bn_biases_neg[-1], bn_scales_neg[-1], 1e-3)    
            h_in.append(self.nonlin(z))
            h_in_pos.append(self.nonlin(z_pos))
            h_in_neg.append(self.nonlin(z_neg))
            h_in[i+1] = tf.nn.dropout(h_in[i+1], do_in)
            h_in_pos[i+1] = tf.nn.dropout(h_in_pos[i+1], do_in)
            h_in_neg[i+1] = tf.nn.dropout(h_in_neg[i+1], do_in)
            
        h_rep = h_in[len(h_in)-1]
        h_rep_pos = h_in_pos[len(h_in_pos)-1]
        h_rep_neg = h_in_neg[len(h_in_neg)-1]
      

        if FLAGS.normalization == 'divide':
            h_rep_norm = h_rep / safe_sqrt(tf.reduce_sum(tf.square(h_rep), axis=1, keep_dims=True))
            h_rep_norm_pos = h_rep_pos / safe_sqrt(tf.reduce_sum(tf.square(h_rep_pos), axis=1, keep_dims=True))
            h_rep_norm_neg = h_rep_neg / safe_sqrt(tf.reduce_sum(tf.square(h_rep_neg), axis=1, keep_dims=True))
        elif FLAGS.normalization == 'l2':
            h_rep_norm = tf.nn.l2_normalize(h_rep,-1)
            h_rep_norm_pos = tf.nn.l2_normalize(h_rep_pos,-1)
            h_rep_norm_neg = tf.nn.l2_normalize(h_rep_neg,-1)
        else:
            h_rep_norm = 1.0*h_rep
            h_rep_norm_pos = 1.0*h_rep_pos
            h_rep_norm_neg = 1.0*h_rep_neg

        '''Contrastive Head'''
    
        weights_cl = []
        bias_cl = []
        w_cl_1 = self._create_variable_with_weight_decay(
            tf.random_normal([dim_in, dim_cl_1],
                            stddev=FLAGS.weight_init / np.sqrt(dim_in)),
                            'w_cl_1', 1.0)
        weights_cl.append(w_cl_1)
        w_cl_2 = self._create_variable_with_weight_decay(
            tf.random_normal([dim_cl_1, dim_cl_2],
                            stddev=FLAGS.weight_init / np.sqrt(dim_cl_1)),
                            'w_cl_2', 1.0)
        weights_cl.append(w_cl_2)
        bias_cl.append(self._create_variable(tf.zeros([1,dim_cl_1]), 'b_cl_1'))
        bias_cl.append(self._create_variable(tf.zeros([1,dim_cl_2]), 'b_cl_2'))
        ##fully-connected layer 1
        p_1 = tf.matmul(h_rep_norm,w_cl_1) 
        p_1_pos =tf.matmul(h_rep_norm_pos,w_cl_1) 
        p_1_neg =tf.matmul(h_rep_norm_neg,w_cl_1) 
        ##Batch Normalization
        p_1_mean, p_1_var = tf.nn.moments(p_1, [0])
        p_1_pos_mean, p_1_pos_var = tf.nn.moments(p_1_pos, [0])
        p_1_neg_mean, p_1_neg_var = tf.nn.moments(p_1_neg, [0])
        p_1 = tf.nn.batch_normalization(p_1, p_1_mean, p_1_var, 0, 1, 1e-3)
        p_1_pos = tf.nn.batch_normalization(p_1_pos, p_1_pos_mean, p_1_pos_var, 0, 1, 1e-3)
        p_1_neg = tf.nn.batch_normalization(p_1_neg, p_1_neg_mean, p_1_neg_var, 0, 1, 1e-3)
        ##Relu
        p_1 = tf.nn.relu(p_1)
        p_1_pos = tf.nn.relu(p_1_pos)
        p_1_neg = tf.nn.relu(p_1_neg)
  
        ##fully-connected layer 2
        p_2 = tf.matmul(p_1,w_cl_2) 
        p_2_pos =tf.matmul(p_1_pos,w_cl_2) 
        p_2_neg =tf.matmul(p_1_neg,w_cl_2) 
        
        ##Batch Normalization
        p_2_mean, p_2_var = tf.nn.moments(p_2, [0])
        p_2_pos_mean, p_2_pos_var = tf.nn.moments(p_2_pos, [0])
        p_2_neg_mean, p_2_neg_var = tf.nn.moments(p_2_neg, [0])
        p_2 = tf.nn.batch_normalization(p_2, p_2_mean, p_2_var, 0, 1, 1e-3)
        p_2_pos = tf.nn.batch_normalization(p_2_pos, p_2_pos_mean, p_2_pos_var, 0, 1, 1e-3)
        p_2_neg = tf.nn.batch_normalization(p_2_neg, p_2_neg_mean, p_2_neg_var, 0, 1, 1e-3)

        ##Contrastive loss
        labels = tf.tile(tf.reshape(tf.one_hot(0, tf.shape(h_rep_norm_neg)[0]+1),(-1,tf.shape(h_rep_norm_neg)[0]+1)),[tf.shape(h_rep_norm)[0],1])
        logits_pos = tf.expand_dims(tf.reduce_sum(tf.multiply(p_2,p_2_pos),-1),-1)/temp
        logits_neg = tf.matmul(p_2,p_2_neg,transpose_b=True)/temp
        logits = tf.concat(1,[logits_pos,logits_neg])
        cl_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels))
     



        ##Output

        y, weights_out, weights_pred = self._build_output_graph(h_rep_norm, t, dim_in, dim_out, do_out, FLAGS)

        ''' Compute sample reweighting '''
        if FLAGS.reweight_sample:
            w_t = t/(2*p_t)
            w_c = (1-t)/(2*1-p_t)
            sample_weight = w_t + w_c
        else:
            sample_weight = 1.0

        self.sample_weight = sample_weight

        ''' Construct factual loss function '''
        if FLAGS.loss == 'l1':
            risk = tf.reduce_mean(sample_weight*tf.abs(y_-y))
            pred_error = -tf.reduce_mean(res)
        elif FLAGS.loss == 'log':
            y = 0.995/(1.0+tf.exp(-y)) + 0.0025
            res = y_*tf.log(y) + (1.0-y_)*tf.log(1.0-y)

            risk = -tf.reduce_mean(sample_weight*res)
            pred_error = -tf.reduce_mean(res)
        else:
            risk = tf.reduce_mean(sample_weight*tf.square(y_ - y))
            pred_error = tf.sqrt(tf.reduce_mean(tf.square(y_ - y)))

     

        ''' Total error '''
        tot_error = risk

        if FLAGS.p_alpha>0:
            tot_error = tot_error + p_alpha*cl_loss

        if FLAGS.p_lambda>0:
            tot_error = tot_error + p_lambda*self.wd_loss




        self.output = y
        self.tot_loss = tot_error
        self.pred_loss = pred_error
        self.weights_in = weights_in
        self.weights_out = weights_out
        self.weights_pred = weights_pred
        self.h_rep = h_rep
        self.h_rep_norm = h_rep_norm
        self.cl_loss = cl_loss
        self.weights_cl = weights_cl
        self.bias_cl = bias_cl
    


    def _build_output(self, h_input, dim_in, dim_out, do_out, FLAGS):
        h_out = [h_input]
        dims = [dim_in] + ([dim_out]*FLAGS.n_out)

        weights_out = []; biases_out = []

        for i in range(0, FLAGS.n_out):
            wo = self._create_variable_with_weight_decay(
                    tf.random_normal([dims[i], dims[i+1]],
                        stddev=FLAGS.weight_init/np.sqrt(dims[i])),
                    'w_out_%d' % i, 1.0)
            weights_out.append(wo)

            biases_out.append(tf.Variable(tf.zeros([1,dim_out])))
            z = tf.matmul(h_out[i], weights_out[i]) + biases_out[i]


            h_out.append(self.nonlin(z))
            h_out[i+1] = tf.nn.dropout(h_out[i+1], do_out)

        weights_pred = self._create_variable(tf.random_normal([dim_out,1],
            stddev=FLAGS.weight_init/np.sqrt(dim_out)), 'w_pred')
        bias_pred = self._create_variable(tf.zeros([1]), 'b_pred')

        if FLAGS.n_out == 0:
            self.wd_loss += tf.nn.l2_loss(tf.slice(weights_pred,[0,0],[dim_out-1,1])) #don't penalize treatment coefficient
        else:
            self.wd_loss += tf.nn.l2_loss(weights_pred)

        ''' Construct linear classifier '''
        h_pred = h_out[-1]
        y = tf.matmul(h_pred, weights_pred)+bias_pred

        return y, weights_out, weights_pred

    
    ''' Construct Prediction Head '''
    def _build_output_graph(self, rep, t, dim_in, dim_out, do_out, FLAGS):
    
        if FLAGS.split_output:

            i0 = tf.to_int32(tf.where(t < 1)[:,0])
            i1 = tf.to_int32(tf.where(t > 0)[:,0])
     
            rep0 = tf.gather(rep, i0)
            rep1 = tf.gather(rep, i1)

            y0, weights_out0, weights_pred0 = self._build_output(rep0, dim_in, dim_out, do_out, FLAGS)
            y1, weights_out1, weights_pred1 = self._build_output(rep1, dim_in, dim_out, do_out, FLAGS)
          
            y = tf.dynamic_stitch([i0, i1], [y0, y1])
            weights_out = weights_out0 + weights_out1
            weights_pred = weights_pred0 + weights_pred1
        else:
            h_input = tf.concat(1,[rep, t])
            y, weights_out, weights_pred = self._build_output(h_input, dim_in+1, dim_out, do_out, FLAGS)

        return y, weights_out, weights_pred
