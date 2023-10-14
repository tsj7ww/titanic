# base
import os
# import datetime
# import re

# data
import pandas as pd
import numpy as np

# stats
# import statsmodels.api as sm
# import scipy.stats as stats
# from scipy.stats import skew, norm
from sklearn.preprocessing import StandardScaler # StandardScaler LabelEncoder RobustScaler MinMaxScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFECV,RFE
# from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OneHotEncoder # OrdinalEncoder
from sklearn.neighbors import LocalOutlierFactor

class Process:
    def __init__(self):
        self.data_dir = os.path.join(os.getcwd(),'data')
        self.test_fname = os.path.join(self.data_dir,'test.csv')
        self.train_fname = os.path.join(self.data_dir,'train.csv')
        self.submission_fname = os.path.join(self.data_dir,'submission.csv')

        self.label_col = 'Survived'
        self.cat_cols = ['Pclass','Sex','Ticket','Cabin','Embarked']
        self.cont_cols = ['Age','Fare','SibSp','Parch','Family','Alone','Mr','Mrs','Miss','Master']
        self.text_cols = ['Name']
    
    def load(self):
        self.train_df = pd.read_csv(self.train_fname,header=0,index_col=0)
        self.test_df = pd.read_csv(self.test_fname,header=0,index_col=0)
        self.submission_df = pd.read_csv(self.submission_fname,header=0,index_col=0)
        self.df = pd.concat([self.train_df,self.test_df],axis=0)
        self.df['Survived'] = self.df['Survived'].astype('Int64')
        
        self.x = self.df.drop(self.label_col,axis=1)
        self.y = self.df[self.label_col]
        self.train_idx = self.df[self.df[self.label_col].notnull()].index
        self.predict_idx = self.df[self.df[self.label_col].isnull()].index
    
    def skew(self):
        self.x['Fare'] = np.log1p(self.x['Fare']+1)

    def fill(self):
        self.x['Embarked'] = self.x['Embarked'].fillna('X')
    
    def feature_engineer(self):
        self._parse_name()
        self._parse_ticket()
        self._parse_cabin()
        self.x['Family'] = self.x['SibSp'] + self.x['Parch']
        self.x['Alone'] = self.x['Family'].apply(lambda x: 1 if x==0 else 0)
    
    def _parse_name(self):
        self.x['Mr'] = self.x.Name.str.contains('Mr.').astype(int)
        self.x['Mrs'] = self.x.Name.str.contains('Mrs.').astype(int)
        self.x['Miss'] = self.x.Name.str.contains('Miss.').astype(int)
        self.x['Master'] = self.x.Name.str.contains('Master.').astype(int)
        self.x = self.x.drop('Name',axis=1)
    
    def _parse_ticket(self):
        self.x['Ticket'] = self.x.Ticket.str.extract('(\d+)')
        self.x['Ticket'] = self.x.Ticket.fillna(0).astype(int)
        self.cont_cols.append('Ticket')
        self.cat_cols.remove('Ticket')
    
    def _parse_cabin(self):
        self.x['Cabin'] = self.x.Cabin.str.extract('(\w)')
        self.x['Cabin'] = self.x.Cabin.fillna('Z')
    
    def cap(self):
        self.x['Age'] = self.x['Age'].clip(0,70)
        self.x['Fare'] = self.x['Fare'].clip(0,100)
        self.x['SibSp'] = self.x['SibSp'].clip(0,2)
        self.x['Parch'] = self.x['Parch'].clip(0,3)
    
    def scale(self):
        scaler = StandardScaler() # StandardScaler MinMaxScaler
        scaled = scaler.fit_transform(self.x[self.cont_cols].values)
        self.x[self.cont_cols] = pd.DataFrame(scaled,columns=self.cont_cols,index=self.x.index)
    
    def encode(self):
        # self.x = pd.get_dummies(self.x,columns=self.cat_cols,drop_first=True)
        for col in self.cat_cols:
            self.x[col] = self.x[col].astype('category').cat.codes
        ohe = OneHotEncoder(categories='auto')
        ohe.fit(self.x[self.cat_cols])
        encoded = ohe.transform(self.x[self.cat_cols]).toarray()
        encoded_df = pd.DataFrame(encoded,index=self.x.index)
        self.x = pd.concat([self.x.drop(self.cat_cols,axis=1),encoded_df],axis=1)
        self.x.columns = [str(col) for col in self.x.columns]
    
    def outliers(self):
        lof = LocalOutlierFactor(n_neighbors=20,contamination=0.1)
        outliers = lof.fit_predict(self.x)
        self.x = self.x.loc[outliers==1]
        self.y = self.y.loc[outliers==1]
    
    def impute(self):
        imputer = IterativeImputer()
        imputed = imputer.fit_transform(self.x)
        self.x = pd.DataFrame(imputed, columns=self.x.columns, index=self.x.index)

        if self.x.isnull().values.any():
             print('Missing values!')
        else:
            print('No missing values')
        
    def smote(self):
        smote = SMOTE()
        self.x.loc[self.train_idx],self.y.loc[self.train_idx] = smote.fit_resample(
            self.x.loc[self.train_idx],self.y.loc[self.train_idx]
        )
    
    def rfe(self,grid=False):
        # correlation matrix?
        if grid:
            rfe = RFECV(estimator=CatBoostClassifier(silent=True),step=1,cv=5,scoring='accuracy',n_jobs=-1)
        else:
            rfe = RFE(estimator=CatBoostClassifier(silent=True),n_features_to_select=60,step=1)
        rfe.fit(self.x.loc[self.train_idx],self.y.loc[self.train_idx])
        self.x = self.x.loc[:,rfe.support_]

    def split(self):
        self.y_train = self.y.loc[self.train_idx]
        self.y_predict = self.y.loc[self.predict_idx]

        self.x_train = self.x.loc[self.train_idx]
        self.x_predict = self.x.loc[self.predict_idx]
        
        self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(
            self.x_train,self.y_train,test_size=0.2,random_state=42
        )
    
    def save(self):
        for df in ['x_train','y_train','x_test','y_test','x_predict']:
            getattr(self,df).to_csv(os.path.join(self.data_dir,f'{df}.csv'),index=True)
