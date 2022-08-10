from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import mlflow

def run_logistic_regression(X, y, X_test, y_test, params={}):
    """
    fit logistic regression model and return different metrics
    params:
    X - pd.DataFrame/np.array - featuers for the model
    y - pd.Series/np.array - target variable
    X_test - pd.DataFrame/np.array - test featuers
    y - pd.Series/np.array - test target
    params - dict - model params
    """
    lr = LogisticRegression(random_state=1, max_iter=1000, solver='saga', **params)
    lr.fit(X, y)
    pred = lr.predict(X_test)
    return lr, accuracy_score(y_test, pred), f1_score(y_test, pred, average='macro'), precision_score(y_test, pred, average='macro'), recall_score(y_test, pred, average='macro')

def run_random_forest(X, y, X_test, y_test, params={}):
    """
    fit random forest model and return different metrics
    params:
    X - pd.DataFrame/np.array - featuers for the model
    y - pd.Series/np.array - target variable
    X_test - pd.DataFrame/np.array - test featuers
    y - pd.Series/np.array - test target
    params - dict - model params
    """
    rf = RandomForestClassifier(random_state=1, n_jobs=4, **params)
    rf.fit(X, y)
    pred = rf.predict(X_test)
    return rf, accuracy_score(y_test, pred), f1_score(y_test, pred, average='macro'), precision_score(y_test, pred, average='macro'), recall_score(y_test, pred, average='macro')

def run_lightgbm(X, y, X_val, y_val, X_test, y_test, params={}):
    """
    fit lightgbm model and return different metrics
    params:
    X - pd.DataFrame/np.array - featuers for the model
    y - pd.Series/np.array - target variable
    X_test - pd.DataFrame/np.array - validation featuers
    y - pd.Series/np.array - validation target
    X_test - pd.DataFrame/np.array - test featuers
    y - pd.Series/np.array - test target
    params - dict - model params
    """    
    p = {**{'force_row_wise':True, 'num_threads':4, 'objective': 'multiclass', 'num_class':3, 'metric': 'multi_logloss', 'random_state':1}, **params}
    clf = lgb.LGBMClassifier(verbose=-1, **p)
    callbacks = [lgb.early_stopping(100, verbose=0), lgb.log_evaluation(period=0)]
    clf.fit(X, y, eval_set=[(X, y),(X_val, y_val)], eval_names=['train','val'], callbacks=callbacks)
    pred = clf.predict(X_test)

    return clf, accuracy_score(y_test, pred), f1_score(y_test, pred, average='macro'), precision_score(y_test, pred, average='macro'), recall_score(y_test, pred, average='macro')

def run_ml_flow_experiment(X_train, X_val, X_test, y_train, y_val, y_test, params, experiment_name='', run_section='all', model_desc=''):
    """
    fit models and track performance in mlflow
    params:
    X - pd.DataFrame/np.array - featuers for the model
    y - pd.Series/np.array - target variable
    X_test - pd.DataFrame/np.array - validation featuers
    y - pd.Series/np.array - validation target
    X_test - pd.DataFrame/np.array - test featuers
    y - pd.Series/np.array - test target
    params - dict - model params
    experiment_name - str - name of the mlflow experiment 
    run_section - str - 'all' or selection between ['lr', 'rf', 'catb']
    model_desc - str - optional description to add when logging
    """
    mlflow.set_experiment(experiment_name=experiment_name)

    if run_section=='all' or 'lr' in run_section:
        for i,p in enumerate(params['lr']):
            with mlflow.start_run() as run:
                lr, acc, f1, prec, rec = run_logistic_regression(X=np.concatenate((X_train, X_val), axis=0), 
                                                    y=np.concatenate((y_train, y_val), axis=0), 
                                                    X_test=X_test, y_test=y_test, params=p)
                print('model: logistic regression, accuracy: ',acc, 'f1: ', f1, '\n params: ', p)
                mlflow.log_params(p)
                mlflow.log_param("model", 'logistic regression'.format(model_desc))
                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("precision", prec)
                mlflow.log_metric("recall", rec)
                mlflow.log_metric("f1", f1)
                mlflow.set_tag('mlflow.runName', 'lr_{}_{}'.format(model_desc, i))

    if run_section=='all' or 'rf' in run_section:
        for i,p in enumerate(params['rf']):
            with mlflow.start_run() as run:
                rf, acc, f1, prec, rec = run_random_forest(X=np.concatenate((X_train, X_val), axis=0), 
                                                    y=np.concatenate((y_train, y_val), axis=0), 
                                                    X_test=X_test, y_test=y_test, params=p)
                print('model: random forest ','accuracy: ',acc, 'f1: ', f1, '\n params: ', p)
                mlflow.log_params(p)
                mlflow.log_param("model", 'random forest'.format(model_desc))
                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("precision", prec)
                mlflow.log_metric("recall", rec)
                mlflow.log_metric("f1", f1)
                mlflow.set_tag('mlflow.runName', 'rf_{}_{}'.format(model_desc, i))

    if run_section=='all' or 'lgb' in run_section:        
        for i,p in enumerate(params['lgb']):
            with mlflow.start_run() as run:
                lgb, acc, f1, prec, rec = run_lightgbm(X=X_train, y=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test, params=p)
                print('model: lightgbm, accuracy: ',acc, 'f1: ', f1, '\n params: ', p)
                mlflow.log_params(p)
                mlflow.log_param("model", 'lightgbm'.format(model_desc))
                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("precision", prec)
                mlflow.log_metric("recall", rec)
                mlflow.log_metric("f1", f1)
                mlflow.set_tag('mlflow.runName', 'lgb_{}_{}'.format(model_desc, i))