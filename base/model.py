import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, # roc_auc_score, roc_curve
    # log_loss, classification_report
)
from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import cross_val_score

# models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier

# neural network
# import tensorflow as tf
# import tensorflow_decision_forests as tfdf

class Model:
    """
    """
    def __init__(self,
                 model,params,name,
                 x_train,y_train,
                 x_test,y_test,
                 x_predict,
                 run_grid=False
            ):
        
        self.base_model = model
        self.params = params
        self.name = name
        
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.x_train_full = pd.concat([x_train,x_test],axis=0)
        self.y_train_full = pd.concat([y_train,y_test],axis=0)
        self.x_predict = x_predict
        
        self.run_grid = run_grid

        self.pred_train = None
        self.pred_test = None
        self.pred_train_full = None
        self.pred_main = None

        self.mse = None
        self.rmse = None
        self.mae = None

        self.label = 'SalePrice'
    
    def run(self,run_type):
        if self.name == 'neural':
            self._nn(run_type)
        elif self.base_model == CatBoostClassifier:
            model = self.base_model(
                silent=True,eval_metric='Accuracy',
                depth=6,iterations=450,learning_rate=0.1
            )
        else:
            try:
                model = self.base_model(verbose=True)
            except:
                model = self.base_model()
        
        if run_type == 'test':
            input_nm = 'train'
            x = self.x_train
            y = self.y_train
            predict = self.x_test
        elif run_type == 'main':
            input_nm = 'train_full'
            x = self.x_train_full
            y = self.y_train_full
            predict = self.x_predict
            
        if self.run_grid:
            grid = GridSearchCV(model,self.params,cv=5,scoring='accuracy',verbose=0,n_jobs=-1)
            grid.fit(x,y)
            model = grid.best_estimator_
        print('fitting model...')
        model.fit(x,y.astype(int))
        print('making predictions...')
        input_preds = np.array(model.predict(x))
        print('setting attributes...')
        predictions = np.array(model.predict(predict))
        setattr(self,f'pred_{input_nm}',pd.Series(input_preds,name=self.name,index=x.index))
        setattr(self,f'pred_{run_type}',pd.Series(predictions,name=self.name,index=predict.index))

    def eval(self):
        self.accuracy = accuracy_score(self.y_test,self.pred_test)
        self.precision = precision_score(self.y_test,self.pred_test)
        self.recall = recall_score(self.y_test,self.pred_test)
        self.f1 = f1_score(self.y_test,self.pred_test)
        self.cm = confusion_matrix(self.y_test,self.pred_test)
        # y_probabilities = model.predict_proba(X_test)[:, 1]
        # fpr, tpr, thresholds = roc_curve(y_true, y_probabilities)
        # self.auc = roc_auc_score(y_true, y_probabilities)

    def _nn(self,run_type):
        model = self.base_model(hyperparameter_template="benchmark_rank1",task = tfdf.keras.Task.REGRESSION)
        
        if run_type == 'test':
            input_nm = 'train'
            x = tfdf.keras.pd_dataframe_to_tf_dataset(
                pd.concat([self.x_train,self.y_train],axis=1),
                label=self.label,
                task=tfdf.keras.Task.CLASSIFICATION
            )
            predict = tfdf.keras.pd_dataframe_to_tf_dataset(
                self.x_test,
                task=tfdf.keras.Task.CLASSIFICATION
            )
        elif run_type == 'main':
            input_nm = 'train_full'
            x = tfdf.keras.pd_dataframe_to_tf_dataset(
                pd.concat([self.x_train_full,self.y_train_full],axis=1),
                label=self.label,
                task=tfdf.keras.Task.CLASSIFICATION
            )
            predict = tfdf.keras.pd_dataframe_to_tf_dataset(
                self.x_predict,
                task=tfdf.keras.Task.CLASSIFICATION
            )

        model.compile(metrics=['mse'])
        model.fit(x=x)
        setattr(self,f'pred_{input_nm}',pd.Series(model.predict(x),name=self.name,index=x.index))
        setattr(self,f'pred_{run_type}',pd.Series(model.predict(predict),name=self.name,index=predict.index))

MODELS = [
    # ensemble
    {
        'name': 'rf',
        'model': RandomForestClassifier,
        'params': {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, 9],
            'min_samples_split': [2, 3, 4],
            'min_samples_leaf': [1, 2, 3],
            'max_features': ['auto', 'sqrt', 'log2'],
            'bootstrap': [True, False]
        }
    },
    {
        'name': 'gb',
        'model': GradientBoostingClassifier,
        'params': {
            'loss': ['ls', 'lad', 'huber', 'quantile'],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, 9],
            'min_samples_split': [2, 3, 4],
            'min_samples_leaf': [1, 2, 3],
            'max_features': ['auto', 'sqrt', 'log2']
        }
    },
    {
        'name': 'xgb',
        'model': XGBClassifier,
        'params': {
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, 9],
            'min_child_weight': [1, 2, 3],
            'gamma': [0, 0.1, 0.2, 0.3],
            'subsample': [0.5, 0.7, 0.9],
            'colsample_bytree': [0.5, 0.7, 0.9],
            'reg_alpha': [0, 0.1, 0.2, 0.3],
            'reg_lambda': [0.5, 0.7, 0.9],
            'scale_pos_weight': [0.5, 0.7, 0.9]
        }
    },
    {
        'name': 'lgb',
        'model': LGBMClassifier,
        'params': {
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, 9],
            'num_leaves': [10, 20, 30, 40],
            'min_child_samples': [10, 20, 30, 40],
            'min_child_weight': [1, 2, 3],
            'subsample': [0.5, 0.7, 0.9],
            'colsample_bytree': [0.5, 0.7, 0.9],
            'reg_alpha': [0, 0.1, 0.2, 0.3],
            'reg_lambda': [0.5, 0.7, 0.9],
            'scale_pos_weight': [0.5, 0.7, 0.9]
        }
    },
    {
        'name': 'cat',
        'model': CatBoostClassifier,
        'params': {
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, 9],
            'subsample': [0.5, 0.7, 0.9],
            'colsample_bylevel': [0.5, 0.7, 0.9],
            'reg_lambda': [0.5, 0.7, 0.9]
        }
    },
    {
        'name': 'log',
        'model': LogisticRegression,
        'params': {
            'penalty': ['l1', 'l2', 'elasticnet', 'none'],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            'C': [0.1, 1, 10],
            'max_iter': [100, 200, 300],
            'multi_class': ['auto', 'ovr', 'multinomial'],
            'l1_ratio': [0.1, 0.2, 0.3]
        }
    },
    {
        'name': 'svm',
        'model': SVC,
        'params': {
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'degree': [2, 3, 4],
            'gamma': ['scale', 'auto'],
            'coef0': [0.0, 0.1, 0.2],
            'tol': [0.001, 0.01, 0.1],
            'C': [0.1, 1, 10],
            'epsilon': [0.1, 0.2, 0.3]
        }
    },
    {
        'name': 'knn',
        'model': KNeighborsClassifier,
        'params': {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'leaf_size': [10, 20, 30, 40],
            'p': [1, 2]
        }
    },
    {
        'name': 'dt',
        'model': DecisionTreeClassifier,
        'params': {
            'criterion': ['mse', 'friedman_mse', 'mae'],
            'splitter': ['best', 'random'],
            'max_depth': [3, 5, 7, 9],
            'min_samples_split': [0.1, 0.2, 0.3],
            'min_samples_leaf': [0.1, 0.2, 0.3],
            'max_features': ['auto', 'sqrt', 'log2']
        }
    },
    {
        'name': 'mlp',
        'model': MLPClassifier,
        'params': {
            'hidden_layer_sizes': [(100,), (200,), (300,)],
        }
    },
    # { # neural network
    #     'name': 'neural',
    #     'model': tfdf.keras.RandomForestModel,
    #     'params': {}
    # },
    # {
    #     'name': 'neural',
    #     'model': tfdf.keras.GradientBoostedTreesModel,
    #     'params': {}
    # },
    # {
    #     'name': 'neural',
    #     'model': tfdf.keras.CartModel,
    #     'params': {}
    # },
    # {
    #     'name': 'neural',
    #     'model': tfdf.keras.RandomForestModel,
    #     'params': {}
    # },
]