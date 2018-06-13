import numpy as np
import scipy as sp
from sklearn.linear_model import Ridge, RidgeClassifier, LogisticRegression
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor 
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import operator
import copy

import bayeso.bayeso.bo

class MyAutoML:
    def __init__(self, info, verbose=True, debug_mode=False, run_on_gpu=False, num_train=None, num_valid=None, num_test=None, dim_data=None):
        self.label_num = info['label_num']
        self.target_num = info['target_num']
        self.task = info['task']
        self.metric = info['metric']
        self.postprocessor = MultiLabelEnsemble(LogisticRegression(), balance=False)
        if debug_mode >= 2:
            self.name = "RandomPredictor"
            self.model = RandomPredictor(self.target_num)
            self.predict_method = self.model.predict_proba
            return
        if info['task'] == 'regression':
            self.name = "RandomForestRegressor"
            self.model = RandomForestRegressor(n_estimators=10, random_state=42, warm_start=True)

            self.predict_method = self.model.predict
        else:
            self.name = "VotingClassifier"
            if num_train is not None and num_valid is not None and num_test is not None and dim_data is not None:
                num_all = num_train + num_valid + num_test
                if num_all > 10000 and dim_data > 1000:
                    n_estimators = 200
                else:
                    n_estimators = 500
            else:
                n_estimators = 200
            min_samples_split = 2

            list_estimators = []
            list_estimators.append(('gbc', GradientBoostingClassifier(n_estimators=n_estimators, verbose=verbose, random_state=42, min_samples_split=min_samples_split, warm_start=True)))
            list_estimators.append(('rfc', RandomForestClassifier(n_estimators=n_estimators, verbose=verbose, random_state=42, min_samples_split=min_samples_split, warm_start=True, n_jobs=-1)))
            list_estimators.append(('etc', ExtraTreesClassifier(n_estimators=n_estimators, verbose=verbose, random_state=42, min_samples_split=min_samples_split, warm_start=True, n_jobs=-1)))
#            list_estimators.append(('abc', AdaBoostClassifier(n_estimators=n_estimators)))

            self.model = VotingClassifier(estimators=list_estimators, voting='soft', weights=[1., 1., 1.])
            if info['task'] == 'multilabel.classification':
                self.model = MultiLabelEnsemble(self.model)

            self.predict_method = self.model.predict_proba

    def __repr__(self):
        return "MyAutoML : " + self.name

    def __str__(self):
        return "MyAutoML : \n" + str(self.model) 

    def fit(self, X, Y):
        self.model.fit(X, Y)
        if self.task != 'regression' and self.postprocessor != None:
            Yhat = self.predict_method(X)
            if len(Yhat.shape) == 1:
                Yhat = np.reshape(Yhat, (-1, 1))           
            self.postprocessor.fit(Yhat, Y)
        return self
        
    def predict(self, X):
        prediction = self.predict_method(X)
        if self.task != 'regression' and self.postprocessor!=None:          
            prediction = self.postprocessor.predict_proba(prediction)
        if self.target_num == 1 and len(prediction.shape) > 1 and prediction.shape[1] > 1:
            prediction = prediction[:, 1]
        if self.task == 'multiclass.classification':
            eps = 1e-15
            norma = np.sum(prediction, axis=1)
            for k in range(prediction.shape[0]):
                prediction[k, :] /= sp.maximum(norma[k], eps)  
        return prediction

class MultiLabelEnsemble:
    ''' MultiLabelEnsemble(predictorInstance, balance=False)
        Like OneVsRestClassifier: Wrapping class to train multiple models when 
        several objectives are given as target values. Its predictor may be an ensemble.
        This class can be used to create a one-vs-rest classifier from multiple 0/1 labels
        to treat a multi-label problem or to create a one-vs-rest classifier from
        a categorical target variable.
        Arguments:
            predictorInstance -- A predictor instance is passed as argument (be careful, you must instantiate
        the predictor class before passing the argument, i.e. end with (), 
        e.g. LogisticRegression().
            balance -- True/False. If True, attempts to re-balance classes in training data
            by including a random sample (without replacement) s.t. the largest class has at most 2 times
        the number of elements of the smallest one.
        Example Usage: mymodel =  MultiLabelEnsemble (GradientBoostingClassifier(), True)'''
	
    def __init__(self, predictorInstance, balance=False):
        self.predictors = [predictorInstance]
        self.n_label = 1
        self.n_target = 1
        self.n_estimators =  1 # for predictors that are ensembles of estimators
        self.balance=balance
        
    def __repr__(self):
        return "MultiLabelEnsemble"

    def __str__(self):
        return "MultiLabelEnsemble : \n" + "\tn_label={}\n".format(self.n_label) + "\tn_target={}\n".format(self.n_target) + "\tn_estimators={}\n".format(self.n_estimators) + str(self.predictors[0])
	
    def fit(self, X, Y):
        if len(Y.shape)==1: 
            Y = np.array([Y]).transpose() # Transform vector into column matrix
            # This is NOT what we want: Y = Y.reshape( -1, 1 ), because Y.shape[1] out of range
        self.n_target = Y.shape[1]                 # Num target values = num col of Y
        self.n_label = len(set(Y.ravel()))         # Num labels = num classes (categories of categorical var if n_target=1 or n_target if labels are binary )
        # Create the right number of copies of the predictor instance
        if len(self.predictors) != self.n_target:
            predictorInstance = self.predictors[0]
            self.predictors = [predictorInstance]
            for i in range(1, self.n_target):
                self.predictors.append(copy.copy(predictorInstance))
        # Fit all predictors
        for i in range(self.n_target):
            # Update the number of desired prodictos
            if hasattr(self.predictors[i], 'n_estimators'):
                self.predictors[i].n_estimators = self.n_estimators
            # Subsample if desired
            if self.balance:
                pos = Y[:, i] > 0
                neg = Y[:, i] <= 0
                if sum(pos) < sum(neg): 
                    chosen = pos
                    not_chosen = neg
                else: 
                    chosen = neg
                    not_chosen = pos
                num = sum(chosen)
                idx = filter(lambda(x): x[1] == True, enumerate(not_chosen))
                idx = np.array(zip(*idx)[0])
                np.random.shuffle(idx)
                chosen[idx[0:min(num, len(idx))]] = True
                # Train with chosen samples
                self.predictors[i].fit(X[chosen, :], Y[chosen, i])
            else:
                self.predictors[i].fit(X, Y[:, i])
        return
		
    def predict_proba(self, X):
        if len(X.shape) == 1: # IG modif Feb3 2015
            X = np.reshape(X, (-1, 1))   
        prediction = self.predictors[0].predict_proba(X)
        if self.n_label == 2: # Keep only 1 prediction, 1st column = (1 - 2nd column)
            prediction = prediction[:, 1]
        for i in range(1, self.n_target): # More than 1 target, we assume that labels are binary
            new_prediction = self.predictors[i].predict_proba(X)[:, 1]
            prediction = np.column_stack((prediction, new_prediction))
        return prediction
		
class RandomPredictor:
    def __init__(self, target_num):
        self.target_num = target_num
        self.n_estimators = 1
        return
        
    def __repr__(self):
        return "RandomPredictor"

    def __str__(self):
        return "RandomPredictor"
	
    def fit(self, X, Y):
        if len(Y.shape) > 1:
            assert(self.target_num == Y.shape[1])
        return self
		
    def predict_proba(self, X):
        prediction = np.random.rand(X.shape[0], self.target_num)
        return prediction
