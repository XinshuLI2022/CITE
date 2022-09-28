import tensorflow as tf
import numpy as np

SQRT_CONST = 1e-10

FLAGS = tf.app.flags.FLAGS

def validation_split(D_exp, val_fraction):
    """ Construct a train/validation split """
    n = D_exp['x'].shape[0]

    if val_fraction > 0:
        n_valid = int(val_fraction*n)
        n_train = n-n_valid
        I = np.random.permutation(range(0,n))
        I_train = I[:n_train]
        I_valid = I[n_train:]
    else:
        I_train = range(n)
        I_valid = []


    return I_train, I_valid

def log(logfile,str):
    """ Log a string in a file """
    with open(logfile,'a') as f:
        f.write(str+'\n')
    print (str)

def save_config(fname):
    """ Save configuration """
    flagdict =  FLAGS.__dict__['__flags']
    s = '\n'.join(['%s: %s' % (k,str(flagdict[k])) for k in sorted(flagdict.keys())])
    f = open(fname,'w')
    f.write(s)
    f.close()

def load_data(fname):
    """ Load data set """
    if fname[-3:] == 'npz':
        data_in = np.load(fname)
        data = {'x': data_in['x'], 't': data_in['t'], 'yf': data_in['yf']}
        try:
            data['ycf'] = data_in['ycf']
        except:
            data['ycf'] = None
    else:
        if FLAGS.sparse>0:
            data_in = np.loadtxt(open(fname+'.y',"rb"),delimiter=",")
            x = load_sparse(fname+'.x')
        else:
            data_in = np.loadtxt(open(fname,"rb"),delimiter=",")
            x = data_in[:,5:]

        data['x'] = x
        data['t'] = data_in[:,0:1]
        data['yf'] = data_in[:,1:2]
        data['ycf'] = data_in[:,2:3]

    data['HAVE_TRUTH'] = not data['ycf'] is None

    data['dim'] = data['x'].shape[1]
    data['n'] = data['x'].shape[0]

    return data

def load_sparse(fname):
    """ Load sparse data set """
    E = np.loadtxt(open(fname,"rb"),delimiter=",")
    H = E[0,:]
    n = int(H[0])
    d = int(H[1])
    E = E[1:,:]
    S = sparse.coo_matrix((E[:,2],(E[:,0]-1,E[:,1]-1)),shape=(n,d))
    S = S.todense()

    return S

def safe_sqrt(x, lbound=SQRT_CONST):
    ''' Numerically safe version of TensorFlow sqrt '''
    return tf.sqrt(tf.clip_by_value(x, lbound, np.inf))








