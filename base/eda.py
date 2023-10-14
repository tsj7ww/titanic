import os
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from wordcloud import WordCloud
from scipy.stats import boxcox

class EDA:
    def __init__(self):
        self.data_dir = os.path.join(os.getcwd(),'data')
        self.test_fname = os.path.join(self.data_dir,'test.csv')
        self.train_fname = os.path.join(self.data_dir,'train.csv')
        self.submission_fname = os.path.join(self.data_dir,'submission.csv')

        self.label_col = 'Survived'
        self.cat_cols = ['Pclass','Sex','SibSp','Parch','Ticket','Cabin','Embarked']
        self.cont_cols = ['Age','Fare']
        self.text_cols = ['Name']
    
    def load(self):
        self.test_df = pd.read_csv(self.test_fname,header=0,index_col=0)
        self.train_df = pd.read_csv(self.train_fname,header=0,index_col=0)
        self.submission_df = pd.read_csv(self.submission_fname,header=0,index_col=0)
        self.df = pd.concat([self.train_df,self.test_df],axis=0)

        self.x = self.df.drop(self.label_col,axis=1)
        self.y = self.df[self.label_col]
        self.train_idx = self.df[self.df[self.label_col].notnull()].index
        self.test_idx = self.df[self.df[self.label_col].isnull()].index

    def print(self):
        display(self.df.head())
        display(self.df.tail())
    
    def plot(self):
        for col in [self.label_col]+self.cat_cols:
            fig,ax = plt.subplots(1,1,figsize=(5,5))
            self.df[col].fillna('None').value_counts().plot(kind='bar',ax=ax)
            plt.show()
        for col in self.cont_cols:
            fig,ax = plt.subplots(1,1,figsize=(5,5))
            self.df[col].fillna(-10).hist(ax=ax)
            # if col=='Fare':
            #     np.log1p(self.df[col]+1).hist(ax=ax)
            ax.set_title(col)
            plt.show()
        for col in self.text_cols:
            text = ' '.join(self.df[col].fillna('None'))
            wordcloud = WordCloud().generate(text)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.show()