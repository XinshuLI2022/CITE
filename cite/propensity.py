# import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle


def propensity_score_training(data, label, mode):

    '''
    :param data: pre-treatment covariates
    :param label: treatment that the units accually took
    :param mode: the method to to get the propsensity score
    :return: the propensity socre (the probability that a unit is in the treated group); the trainied propensity calcualtion model
    '''
    # split the data into trainset and testset randomly, return train_data,test_data,train_label,test_label
    train_x, eva_x, train_t, eva_t = train_test_split(data, label, test_size=0.3, random_state=42)

    if mode == 'Logistic-regression':
        train_t = train_t.flatten()
        clf = LogisticRegression('l2', class_weight='balanced',C=3.0 )
        clf.fit(train_x, train_t.flatten())
        prob_all = clf.predict_proba(data)
        return prob_all, clf


def onehot_trans(t, catog):
    trans = np.zeros([t.shape[0], catog.size])
    for i in range(t.shape[0]):
        if t[i,0] == 0:
            trans[i,0] = 1
        else:
            trans[i,1] = 1
    return trans

def load_propensity_score(model_file_name,x):
    loaded_model = pickle.load(open(model_file_name, 'rb'))
    result = loaded_model.predict_proba(x)
    propensity_score = result[:,1]
    propensity_score = propensity_score.flatten()
    return propensity_score







