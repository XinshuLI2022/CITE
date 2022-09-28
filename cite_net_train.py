#coding=utf-8
import tensorflow as tf
import numpy as np
import sys, os
import random
import datetime
import traceback

import cite.cite_net as cite
from cite.util import *
from cite.propensity import *
from cite.contrastive import *
''' Define parameter flags '''
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('loss', 'l2', """Which loss function to use (l1/l2/log)""")
tf.app.flags.DEFINE_integer('n_in', 2, """Number of representation layers. """)
tf.app.flags.DEFINE_integer('n_out', 2, """Number of regression layers. """)
tf.app.flags.DEFINE_float('p_alpha', 1e-4, """Contrastive loss param. """)
tf.app.flags.DEFINE_float('p_lambda', 0.0, """Weight decay regularization parameter. """)
tf.app.flags.DEFINE_integer('rep_weight_decay', 1, """Whether to penalize representation layers with weight decay""")
tf.app.flags.DEFINE_float('dropout_in', 0.9, """Input layers dropout keep rate. """)
tf.app.flags.DEFINE_float('dropout_out', 0.9, """Output layers dropout keep rate. """)
tf.app.flags.DEFINE_string('nonlin', 'relu', """Kind of non-linearity. Default relu. """)
tf.app.flags.DEFINE_float('lrate', 0.05, """Learning rate. """)
tf.app.flags.DEFINE_float('decay', 0.5, """RMSProp decay. """)
tf.app.flags.DEFINE_integer('batch_size', 100, """Batch size. """)
tf.app.flags.DEFINE_integer('dim_in', 100, """Pre-representation layer dimensions. """)
tf.app.flags.DEFINE_integer('dim_out', 100, """Post-representation layer dimensions. """)
tf.app.flags.DEFINE_integer('batch_norm', 0, """Whether to use batch normalization. """)
tf.app.flags.DEFINE_string('normalization', 'none', """How to normalize representation (after batch norm). none/bn_fixed/divide/project """)
tf.app.flags.DEFINE_integer('experiments', 1, """Number of experiments. """)
tf.app.flags.DEFINE_integer('iterations', 2000, """Number of iterations. """)
tf.app.flags.DEFINE_float('weight_init', 0.01, """Weight initialization scale. """)
tf.app.flags.DEFINE_float('lrate_decay', 0.95, """Decay of learning rate every 100 iterations """)
tf.app.flags.DEFINE_string('outdir', '../results/', """Output directory. """)
tf.app.flags.DEFINE_string('datadir', '../data/', """Data directory. """)
tf.app.flags.DEFINE_string('dataform', 'ihdp_npci_1-100.train.npz', """Training data filename form. """)
tf.app.flags.DEFINE_string('data_test', 'ihdp_npci_1-100.test.npz', """Test data filename form. """)
tf.app.flags.DEFINE_integer('sparse', 0, """Whether data is stored in sparse format (.x, .y). """)
tf.app.flags.DEFINE_integer('seed', 1, """Seed. """)
tf.app.flags.DEFINE_integer('repetitions', 1, """Repetitions with different seed.""")
tf.app.flags.DEFINE_string('optimizer', 'RMSProp', """Which optimizer to use. (RMSProp/Adagrad/GradientDescent/Adam)""")
tf.app.flags.DEFINE_integer('output_delay', 100, """Number of iterations between log/loss outputs. """)
tf.app.flags.DEFINE_integer('pred_output_delay', -1, """Number of iterations between prediction outputs. (-1 gives no intermediate output). """)
tf.app.flags.DEFINE_integer('save_rep', 1, """Save representations after training. """)
tf.app.flags.DEFINE_float('val_part', 0, """Validation part. """)
tf.app.flags.DEFINE_boolean('split_output', 0, """Whether to split output layers between treated and control. """)
tf.app.flags.DEFINE_boolean('reweight_sample', 1, """Whether to reweight sample for prediction loss with average treatment probability. """)
tf.app.flags.DEFINE_string('propensity_dir', 'propensity', """Propensity score directory. """)
tf.app.flags.DEFINE_float('temperature', 0.1, """Contrastive learning temperature. """)
tf.app.flags.DEFINE_integer('dim_cl_in', 200, """Contrastive Learning layer_1 dimensions. """)
tf.app.flags.DEFINE_integer('dim_cl_out', 100, """Contrastive Learning layer_2 dimensions. """)
tf.app.flags.DEFINE_string('p_train', 'Logistic-regression', """Propensity Score Train Mode. """)


if FLAGS.sparse:
    import scipy.sparse as sparse

NUM_ITERATIONS_PER_DECAY = 100




def train(CITE, sess, train_step, D, I_valid, D_test, logfile, i_exp,score_file):
    """ Trains a CITE model on supplied data """


    ''' Train/validation split '''
  
    n = D['x'].shape[0]
    I = range(n); I_train = list(set(I)-set(I_valid))
    n_train = len(I_train)
    
    '''Generate positive and negative samples'''
    
    p_score_train = load_propensity_score(score_file,D['x'][I_train,:])
    pos_train = x_positive(D['x'][I_train,:],p_score_train,FLAGS.batch_size)   
    neg_train = x_negative(D['x'][I_train,:],p_score_train,FLAGS.batch_size) 

    p_score_valid = load_propensity_score(score_file,D['x'][I_valid,:])
    pos_valid = x_positive(D['x'][I_valid,:],p_score_valid,FLAGS.batch_size//4)   
    neg_valid = x_negative(D['x'][I_valid,:],p_score_valid,FLAGS.batch_size//4)

    ''' Compute treatment probability'''
    p_treated = np.mean(D['t'][I_train,:])

    ''' Set up loss feed_dicts'''
    dict_factual = {CITE.x: D['x'][I_train,:], CITE.x_pos: pos_train,CITE.x_neg: neg_train, CITE.t: D['t'][I_train,:], CITE.y_: D['yf'][I_train,:], \
      CITE.do_in: 1.0, CITE.do_out: 1.0, CITE.p_alpha: FLAGS.p_alpha, \
      CITE.p_lambda: FLAGS.p_lambda, CITE.p_t: p_treated, CITE.temp: FLAGS.temperature}

    if FLAGS.val_part > 0:
        dict_valid = {CITE.x: D['x'][I_valid,:], CITE.x_pos: pos_valid,CITE.x_neg: neg_valid, CITE.t: D['t'][I_valid,:], CITE.y_: D['yf'][I_valid,:], \
          CITE.do_in: 1.0, CITE.do_out: 1.0, CITE.p_alpha: FLAGS.p_alpha, \
          CITE.p_lambda: FLAGS.p_lambda, CITE.p_t: p_treated, CITE.temp: FLAGS.temperature}

    if D['HAVE_TRUTH']:
        dict_cfactual = {CITE.x: D['x'][I_train,:], CITE.x_pos: pos_train,CITE.x_neg:neg_train, CITE.t: 1-D['t'][I_train,:], CITE.y_: D['ycf'][I_train,:], \
          CITE.do_in: 1.0, CITE.do_out: 1.0, CITE.temp: FLAGS.temperature}

    ''' Initialize TensorFlow variables '''
    sess.run(tf.global_variables_initializer())

    ''' Set up for storing predictions '''
    preds_train = []
    preds_test = []

    ''' Compute losses '''
    losses = []
    obj_loss, f_error, cl_err = sess.run([CITE.tot_loss, CITE.pred_loss, CITE.cl_loss],\
      feed_dict=dict_factual)

    cf_error = np.nan
    if D['HAVE_TRUTH']:
        cf_error = sess.run(CITE.pred_loss, feed_dict=dict_cfactual)

    valid_obj = np.nan
    valid_cl = np.nan
    valid_f_error = np.nan
    if FLAGS.val_part > 0:
        valid_obj, valid_pred, valid_cl = sess.run([CITE.tot_loss, CITE.pred_loss, CITE.cl_loss],\
          feed_dict=dict_valid)

    losses.append([obj_loss, f_error, cf_error, cl_err,valid_obj,valid_pred, valid_cl])

    objnan = False

    reps = []
    reps_test = []

    ''' Train for multiple iterations '''
    for i in range(FLAGS.iterations):

        ''' Fetch sample '''
        I = random.sample(range(0, n_train), FLAGS.batch_size)
    
        x_batch = D['x'][I_train,:][I,:]
        t_batch = D['t'][I_train,:][I]
        y_batch = D['yf'][I_train,:][I]
        p_score_batch = load_propensity_score(score_file,x_batch)
        x_pos_batch = x_positive(x_batch,p_score_batch,FLAGS.batch_size//4)   
        x_neg_batch = x_negative(D['x'][I_train,:],p_score_train,FLAGS.batch_size)
        
   

        ''' Do one step of gradient descent '''
        if not objnan:
            sess.run(train_step, feed_dict={CITE.x: x_batch, CITE.x_pos: x_pos_batch, CITE.x_neg: x_neg_batch, \
                CITE.t: t_batch, CITE.y_: y_batch, CITE.do_in: FLAGS.dropout_in, CITE.do_out: FLAGS.dropout_out, \
                CITE.p_alpha: FLAGS.p_alpha, CITE.p_lambda: FLAGS.p_lambda, CITE.p_t: p_treated, CITE.temp: FLAGS.temperature})


        ''' Compute loss every N iterations '''
        ##print loss every output_delay iteration
        if i % FLAGS.output_delay == 0 or i==FLAGS.iterations-1:
            obj_loss,f_error,cl_err = sess.run([CITE.tot_loss, CITE.pred_loss, CITE.cl_loss],
                feed_dict=dict_factual)
        

            cf_error = np.nan
            if D['HAVE_TRUTH']:
                cf_error = sess.run(CITE.pred_loss, feed_dict=dict_cfactual)

            valid_obj = np.nan; valid_cl = np.nan; valid_f_error = np.nan;
            if FLAGS.val_part > 0:
                valid_obj, valid_f_error, valid_cl = sess.run([CITE.tot_loss, CITE.pred_loss, CITE.cl_loss], feed_dict=dict_valid)

            losses.append([obj_loss, f_error, cf_error, cl_err, valid_obj,valid_f_error, valid_cl])
            loss_str = str(i) + '\tObj: %.3f,\tF: %.3f,\tCf: %.3f,\tCl: %.2g,\tVal_total: %.3f,\tVal_pred: %.2g,\tVal_cl: %.2f' \
                        % (obj_loss, f_error, cf_error, cl_err, valid_obj,valid_f_error, valid_cl)

            if FLAGS.loss == 'log':
                y_pred = sess.run(CITE.output, feed_dict={CITE.x: x_batch, \
                    CITE.t: t_batch, CITE.do_in: 1.0, CITE.do_out: 1.0})
                y_pred = 1.0*(y_pred > 0.5)
                acc = 100*(1 - np.mean(np.abs(y_batch - y_pred)))
                loss_str += ',\tAcc: %.2f%%' % acc

            log(logfile, loss_str)

            if np.isnan(obj_loss):
                log(logfile,'Experiment %d: Objective is NaN. Skipping.' % i_exp)
                objnan = True

        ''' Compute predictions every M iterations '''
        ##predict every pred_output_delay iterations
        if (FLAGS.pred_output_delay > 0 and i % FLAGS.pred_output_delay == 0) or i==FLAGS.iterations-1:

            y_pred_f = sess.run(CITE.output, feed_dict={CITE.x: D['x'], \
                CITE.t: D['t'], CITE.do_in: 1.0, CITE.do_out: 1.0})
            y_pred_cf = sess.run(CITE.output, feed_dict={CITE.x: D['x'], \
                CITE.t: 1-D['t'], CITE.do_in: 1.0, CITE.do_out: 1.0})
            preds_train.append(np.concatenate((y_pred_f, y_pred_cf),axis=1))

            if D_test is not None:
                y_pred_f_test = sess.run(CITE.output, feed_dict={CITE.x: D_test['x'], \
                    CITE.t: D_test['t'], CITE.do_in: 1.0, CITE.do_out: 1.0})
                y_pred_cf_test = sess.run(CITE.output, feed_dict={CITE.x: D_test['x'], \
                    CITE.t: 1-D_test['t'], CITE.do_in: 1.0, CITE.do_out: 1.0})
                preds_test.append(np.concatenate((y_pred_f_test, y_pred_cf_test),axis=1))

            if FLAGS.save_rep and i_exp == 1:
                reps_i = sess.run([CITE.h_rep], feed_dict={CITE.x: D['x'], \
                    CITE.do_in: 1.0, CITE.do_out: 0.0})
                reps.append(reps_i)

                if D_test is not None:
                    reps_test_i = sess.run([CITE.h_rep], feed_dict={CITE.x: D_test['x'], \
                        CITE.do_in: 1.0, CITE.do_out: 0.0})
                    reps_test.append(reps_test_i)

    return losses, preds_train, preds_test, reps, reps_test

def run(outdir):
    """ Runs an experiment and stores result in outdir """

    ''' Set up paths and start log '''
    npzfile = outdir+'result'
    npzfile_test = outdir+'result.test'
    repfile = outdir+'reps'
    repfile_test = outdir+'reps.test'

    logfile = outdir+'log.txt'
    f = open(logfile,'w')
    f.close()
    dataform = FLAGS.datadir + FLAGS.dataform
   
    has_test = False
    if not FLAGS.data_test == '': 
        has_test = True
        dataform_test = FLAGS.datadir + FLAGS.data_test

    ''' Set random seeds '''
    random.seed(FLAGS.seed)
    tf.set_random_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    ''' Save parameters '''
    save_config(outdir+'config.txt')

    log(logfile, 'Training with hyperparameters: alpha=%.2g, lambda=%.2g' % (FLAGS.p_alpha,FLAGS.p_lambda))

    ''' Load Data '''
    npz_input = False
    if dataform[-3:] == 'npz':
        npz_input = True
    if npz_input:
        datapath = dataform
        if has_test:
            datapath_test = dataform_test
    else:
        datapath = dataform % 1
        if has_test:
            datapath_test = dataform_test % 1

    log(logfile,     'Training data: ' + datapath)
    if has_test:
        log(logfile, 'Test data:     ' + datapath_test)
    D = load_data(datapath)
    D_test = None
    if has_test:
        D_test = load_data(datapath_test)

    log(logfile, 'Loaded data with shape [%d,%d]' % (D['n'], D['dim']))

    ''' Start Session '''
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True

    sess = tf.Session(config=config)

    ''' Initialize input placeholders '''
    x  = tf.placeholder("float", shape=[None, D['dim']], name='x') # Features
    t  = tf.placeholder("float", shape=[None, 1], name='t')   # Treatent
    y_ = tf.placeholder("float", shape=[None, 1], name='y_')  # Outcome
    x_pos = tf.placeholder("float", shape=[None,D['dim']], name='x_pos') #positive sample matrix
    x_neg = tf.placeholder("float", shape=[None,D['dim']], name='x_neg') #negative sample matrix

    ''' Parameter placeholders '''
    r_alpha = tf.placeholder("float", name='r_alpha')
    r_lambda = tf.placeholder("float", name='r_lambda')
    do_in = tf.placeholder("float", name='dropout_in')
    do_out = tf.placeholder("float", name='dropout_out')
    p = tf.placeholder("float", name='p_treated')
    temp = tf.placeholder("float", name='temp')

    ''' Define model graph '''
    log(logfile, 'Defining graph...\n')
    dims = [D['dim'], FLAGS.dim_in, FLAGS.dim_out, FLAGS.dim_cl_in,FLAGS.dim_cl_out]
    batch_size = FLAGS.batch_size
 
    CITE = cite.cite_net(x, t, y_, p, FLAGS, r_alpha, r_lambda, do_in, do_out, dims,x_pos,x_neg,temp,batch_size)

    ''' Set up optimizer '''
    global_step = tf.Variable(0, trainable=False)

    lr = tf.train.exponential_decay(FLAGS.lrate, global_step, \
        NUM_ITERATIONS_PER_DECAY, FLAGS.lrate_decay, staircase=True)

    opt = None
    if FLAGS.optimizer == 'Adagrad':
        opt = tf.train.AdagradOptimizer(lr)
    elif FLAGS.optimizer == 'GradientDescent':
        opt = tf.train.GradientDescentOptimizer(lr)
    elif FLAGS.optimizer == 'Adam':
        opt = tf.train.AdamOptimizer(lr)
    else:
        opt = tf.train.RMSPropOptimizer(lr, FLAGS.decay)

    ''' Unused gradient clipping '''

    train_step = opt.minimize(CITE.tot_loss,global_step=global_step)

    ''' Set up for saving variables '''
    all_losses = []
    all_preds_train = []
    all_preds_test = []
    all_valid = []
    all_preds_test = []

    ''' Handle repetitions '''
    ##repeat 100 times
    n_experiments = FLAGS.experiments
    if FLAGS.repetitions>1:
        if FLAGS.experiments>1:
            log(logfile, 'ERROR: Use of both repetitions and multiple experiments is currently not supported.')
            sys.exit(1)
        n_experiments = FLAGS.repetitions

    ''' Run for all repeated experiments '''
    for i_exp in range(1,n_experiments+1):

        if FLAGS.repetitions>1:
            log(logfile, 'Training on repeated initialization %d/%d...' % (i_exp, FLAGS.repetitions))
        else:
            log(logfile, 'Training on experiment %d/%d...' % (i_exp, n_experiments))

        ''' Load Data (if multiple repetitions, reuse first set)'''

        if i_exp==1 or FLAGS.experiments>1:
            D_exp_test = None
            if npz_input:
                D_exp = {}
                D_exp['x']  = D['x'][:,:,i_exp-1]
                D_exp['t']  = D['t'][:,i_exp-1:i_exp]
                D_exp['yf'] = D['yf'][:,i_exp-1:i_exp]
                if D['HAVE_TRUTH']:
                    D_exp['ycf'] = D['ycf'][:,i_exp-1:i_exp]
                else:
                    D_exp['ycf'] = None

                if has_test:
                    D_exp_test = {}
                    D_exp_test['x']  = D_test['x'][:,:,i_exp-1]
                    D_exp_test['t']  = D_test['t'][:,i_exp-1:i_exp]
                    D_exp_test['yf'] = D_test['yf'][:,i_exp-1:i_exp]
                    if D_test['HAVE_TRUTH']:
                        D_exp_test['ycf'] = D_test['ycf'][:,i_exp-1:i_exp]
                    else:
                        D_exp_test['ycf'] = None
            else:
                datapath = dataform % i_exp
                D_exp = load_data(datapath)
                if has_test:
                    datapath_test = dataform_test % i_exp
                    D_exp_test = load_data(datapath_test)

            D_exp['HAVE_TRUTH'] = D['HAVE_TRUTH']
            if has_test:
                D_exp_test['HAVE_TRUTH'] = D_test['HAVE_TRUTH']

       
        '''propensity score training'''

        propensity_dir = FLAGS.propensity_dir 
        _, clf = propensity_score_training(D_exp['x'],D_exp['t'],FLAGS.p_train)
        score_file = propensity_dir + "/propensity_model_"+str(i_exp) +".sav"
        pickle.dump(clf, open(score_file, 'wb'))
        print("Propensity score training finished")
        

        ''' Split into training and validation sets '''
        I_train, I_valid = validation_split(D_exp, FLAGS.val_part)


        ''' Run training loop '''
       
        losses, preds_train, preds_test, reps, reps_test = \
            train(CITE, sess, train_step, D_exp, I_valid, \
                D_exp_test, logfile, i_exp,score_file)

        ''' Collect all reps '''
        all_preds_train.append(preds_train)
        all_preds_test.append(preds_test)
        all_losses.append(losses)

        ''' Fix shape for output (n_units, dim, n_reps, n_outputs) '''
     
        out_preds_train = np.swapaxes(np.swapaxes(all_preds_train,1,3),0,2)
        if  has_test:
            out_preds_test = np.swapaxes(np.swapaxes(all_preds_test,1,3),0,2)
        out_losses = np.swapaxes(np.swapaxes(all_losses,0,2),0,1)

        ''' Store predictions '''
        log(logfile, 'Saving result to %s...\n' % outdir)

        ''' Save results and predictions '''
        all_valid.append(I_valid)
     
        np.savez(npzfile, pred=out_preds_train, loss=out_losses, val=np.array(all_valid))

        if has_test:
            np.savez(npzfile_test, pred=out_preds_test)

        ''' Save representations '''
        if FLAGS.save_rep and i_exp == 1:
            np.savez(repfile, rep=reps)

            if has_test:
                np.savez(repfile_test, rep=reps_test)

def main(argv=None):  # pylint: disable=unused-argument
    """ Main entry point """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S-%f")
    outdir = FLAGS.outdir+'/results_'+timestamp+'/'
    os.mkdir(outdir)

    try:
        run(outdir)
    except Exception as e:
        with open(outdir+'error.txt','w') as errfile:
            errfile.write(''.join(traceback.format_exception(*sys.exc_info())))
        raise

if __name__ == '__main__':
    tf.app.run()
