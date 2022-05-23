#!/usr/bin/env python
# coding: utf-8

from flask import Flask, jsonify, request
import joblib
from bs4 import BeautifulSoup
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
import lightgbm as lgb
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
import re
import pickle
import sqlite3
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_curve, auc
import re
import os
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score
import gc
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

# https://www.tutorialspoint.com/flask
import flask
app = Flask(__name__)


###################################################

root_dir='D:/AAIC dataset/self case study 1'
application_test = pd.read_csv(os.path.join(root_dir,'application_test.csv'))
test_df=application_test[application_test['SK_ID_CURR']==input_id]

###################################################
def categorical_feature_preprocess(value):
    value=value.replace(':','')
    value= '_'.join(i for i in value.split() if i not in ['/','_','-']).lower()
    value=value.replace('/','_')
    return value

def preprocessing_feature_engineering_application(data):
    #removing the value as suspected as outlier
    data[data['AMT_INCOME_TOTAL']>1e8]['AMT_INCOME_TOTAL']=np.nan
    #as per the column description
    data['AMT_REQ_CREDIT_BUREAU_SUM']=(data['AMT_REQ_CREDIT_BUREAU_HOUR'])+     data[ 'AMT_REQ_CREDIT_BUREAU_DAY']+     (data['AMT_REQ_CREDIT_BUREAU_WEEK'])+(data['AMT_REQ_CREDIT_BUREAU_MON'])+     (data['AMT_REQ_CREDIT_BUREAU_QRT'])+(data['AMT_REQ_CREDIT_BUREAU_YEAR'])
    #dropping rest of the columns
    data.drop(['AMT_REQ_CREDIT_BUREAU_HOUR','AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_WEEK','AMT_REQ_CREDIT_BUREAU_MON'               ,'AMT_REQ_CREDIT_BUREAU_QRT','AMT_REQ_CREDIT_BUREAU_YEAR'],axis=1)
    #removing the value as suspected as outlier
    data[data['AMT_REQ_CREDIT_BUREAU_SUM']>250]['AMT_REQ_CREDIT_BUREAU_SUM']=np.nan
    
    #converting the days field into year
    data['DAYS_BIRTH_YEAR']=np.abs(data['DAYS_BIRTH'])
    #removing the value as suspected as outlier
    data[data['DAYS_EMPLOYED']==365243]['DAYS_EMPLOYED']=np.nan
    data['DAYS_EMPLOYED']=np.abs(data['DAYS_EMPLOYED'])
    data['DAYS_REGISTRATION']=np.abs(data['DAYS_REGISTRATION'])
    data['DAYS_ID_PUBLISH']=np.abs(data['DAYS_ID_PUBLISH'])
    data['DAYS_LAST_PHONE_CHANGE']=np.abs(data['DAYS_LAST_PHONE_CHANGE'])
       
    
    categorical_feature_name_list=data.select_dtypes('object').columns.values
    for cat_features in categorical_feature_name_list:
        data[cat_features]=data[cat_features].apply(lambda x: categorical_feature_preprocess(str(x)) if x!=np.nan                                                               else 'nan')
        
        
    #removing the value as suspected as outlier
    data[data['OBS_30_CNT_SOCIAL_CIRCLE']>340]['OBS_30_CNT_SOCIAL_CIRCLE']=np.nan
    data[data['OBS_60_CNT_SOCIAL_CIRCLE']>340]['OBS_60_CNT_SOCIAL_CIRCLE']=np.nan
    
    #how much people defaulted in their social circle 
    data['DEF_30_FRAC_DEF_SOCIAL_CIRCLE']=data['DEF_30_CNT_SOCIAL_CIRCLE']/data['OBS_30_CNT_SOCIAL_CIRCLE']
    data['DEF_60_FRAC_DEF_SOCIAL_CIRCLE']=data['DEF_60_CNT_SOCIAL_CIRCLE']/data['OBS_60_CNT_SOCIAL_CIRCLE']
    
    #dropping rest of the columns
    data.drop(['DEF_30_CNT_SOCIAL_CIRCLE','OBS_30_CNT_SOCIAL_CIRCLE','DEF_60_CNT_SOCIAL_CIRCLE'               ,'OBS_60_CNT_SOCIAL_CIRCLE'],axis=1)
    

    #new features taken from domain knowledged and kaggle discussion kernels:
    #credit amount and annuity amount ratios
    data['CREDIT_ANNUITY_RATIO']=data['AMT_CREDIT']/data['AMT_ANNUITY']
    
    #good price affordable on income
    data['GOODS_PRICE_AFFORDABLE']=data['AMT_INCOME_TOTAL']/data['AMT_GOODS_PRICE']
    
    #work experiance to clients age
    data['EMPLOYED_AGE_RATIO']=data['DAYS_EMPLOYED'] / data['DAYS_BIRTH']
    
    #mean,max,min of these 3 scores EXT_SOURCE_X
    data['EXT_SOURCE_SCORE_MEAN']=data[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis = 1)
    data['EXT_SOURCE_SCORE_MIN']=data[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].min(axis = 1)
    data['EXT_SOURCE_SCORE_MAX']=data[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].max(axis = 1)
    
    return data

def predict_missing_col(data,missing_columns):
    temp_df = pd.DataFrame(columns = ['pred_' + name for name in missing_columns])
    data_temp=data.copy()
    for col in missing_columns:
        #storing the columns value to impute
        temp_df['pred_' + col] = data[col]
        #value will be predicted on the rest of the columns
        rest_col = list(set(data.columns) - set(missing_columns)-set(['SK_ID_CURR','TARGET']))
        #converting the categorical features to one hot encoding to apply the model
        categorical_feature_name_list=data_temp.select_dtypes('object').columns.values
        #converting all the categorical columns into frequency encoding    
        for cat_features in categorical_feature_name_list:
            #calculating the frequency
            temp_dict=joblib.load(os.path.join(root_dir,str(cat_features)+'_temp_dict.pkl'))
            #converting the features to frequency
            data_temp[cat_features]=data_temp[cat_features].apply(lambda x:temp_dict.get(x,0))
            
        #light GBM regressor model to predict the missing values
        lgbmr=joblib.load(os.path.join(root_dir,'missing_col_model.pkl'))
        lgbmr = LGBMRegressor(max_depth = 9, n_estimators = 5000, n_jobs = -1, learning_rate = 0.3, 
                                      random_state = 125)
        #fitting on the data
        lgbmr.fit(X = data_temp[rest_col], y = data_temp[col])
        #adding predicted values in the temp train and test data
        temp_df.loc[data_temp[col].isnull(), 'pred_'+col] = lgbmr.predict(data_temp[rest_col])        [data_temp[col].isnull()]
        
    # memory management
    gc.enable()
    del data_temp
    gc.collect()
    return temp_df

#this will get the mean value of numerical feature alues based on their corresponding value in the categorical features
def categorical_numerical_feature_engineering(data,numerical_feature,categorical_feature):
    #apply groupby to get the mean of numerical values for each unique value of categoical feature
    temp=data[[numerical_feature,categorical_feature]].groupby([categorical_feature]).agg(['mean']).reset_index()
    #converting it to a dictionary to get the new feature
    temp_dict=dict(np.array(temp))
    return temp_dict

#we are taking the top 10 features 
def calculating_category_wise_numerical_mean(data):
    categorical_feature_name_list=data.select_dtypes('object').columns.values
    top_10_numerical_feature_list=['EXT_SOURCE_2','EXT_SOURCE_3','DAYS_BIRTH','DAYS_ID_PUBLISH',    'DAYS_REGISTRATION','AMT_ANNUITY','DAYS_EMPLOYED','DAYS_LAST_PHONE_CHANGE','AMT_CREDIT','AMT_INCOME_TOTAL']  
    #adding total 160 features to the main data set
    for categorical_feature in categorical_feature_name_list:
        for numerical_feature in top_10_numerical_feature_list:
            temp_dict=joblib.load(os.path.join(root_dir,str(cat_features)+'_temp_dict_160.pkl'))
            #creating new feature name
            new_feature_name=str(numerical_feature)+'_'+str(categorical_feature)+'_mean'
            #adding the feature in the dataframe
            data[new_feature_name]=data[categorical_feature].apply(lambda x: temp_dict.get(x,0) )
    return data   
            
            
def onehot_encoding_categorical_features(data):
    #categorical features
    categorical_feature_name_list=data.select_dtypes('object').columns.values
    temp_dict=joblib.load(os.path.join(root_dir,str(cat_features)+'one_hot_encoding_feature.pkl'))
    #converting all the categorical columns into one hot encoding   
    for cat_features in categorical_feature_name_list:
        #print(cat_features)
        #creating countervectorizer for each unique value
        #vectorizer = CountVectorizer(tokenizer=lambda txt: txt.split())
        #fitting on the train data
        #vectorizer.fit(data[cat_features].apply(lambda x: np.str_(x)))
        #temp_dict[cat_features]=vectorizer.get_feature_names()
        one_hot_encoded=np.zeros(shape=(len(data),len(temp_dict[cat_features])))
        #applying transform on the train data
        for j in range(len(data)):
            for i in range(len(temp_dict[cat_features])):
                if data[cat_features][j]==temp_dict[cat_features][i]:
                    one_hot_encoded[j][i]=1
        temp_train1=pd.DataFrame(one_hot_encoded,columns=[cat_features+'_'+i for i in temp_dict[cat_features]],index=application_train.index)

        #adding the one hot encoded feature to the main dataset
        data=pd.concat([data,temp_train1],axis=1)
        
        #dropping the original categorical features after one hot encoding them
        data.drop(cat_features,axis=1,inplace=True)
        
        
        # memory management
        gc.enable()
        #deleting temporary datas
        del temp_train1
        gc.collect()

    return data,temp_dict 

def kmeans_clustering(data):
    #normalizing the value for applying PCA
    #std_scalar=joblib.load(os.path.join(root_dir,'std_scalar.pkl'))
    std_scalar = joblib.load(os.path.join(root_dir,'std_scalar.pkl'))
    
    #applying fit and transform on the train data
    data_std=std_scalar.transform(data.drop('SK_ID_CURR',axis=1))
    #joblib.dump(std_scalar,os.path.join(root_dir,'std_scalar.pkl'))
    #Applying PCA to tranform data from 135 to 5 feature to apply k-means clustering
    #pca=joblib.load(os.path.join(root_dir,'pca.pkl'))
    pca = joblib.load(os.path.join(root_dir,'pca.pkl'))
    data_pca=pca.transform(np.nan_to_num(data_std))
    algo=joblib.load(os.path.join(root_dir,'kmeans.pkl'))
    #algo = KMeans(n_clusters=2,random_state=15)
    #fitting on the train data
    
    #joblib.dump(algo,os.path.join(root_dir,'kmeans.pkl'))
    #adding the cluster label as a feature
    data['K_MEANS_CLUSTER_LABEL']=algo.predict(data_pca)
    return data


def additional_feature_engineering(data,label):
   
    #https://www.kaggle.com/c/home-credit-default-risk/discussion/64784
    #from winner solution neighbors_target_mean_500: The mean TARGET value of the 500 closest neighbors 
    #of each row, where each neighborhood was defined by the three external sources and the credit/annuity ratio.
    #https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    #knn=joblib.load(os.path.join(root_dir,'knn.pkl'))
    #knn = KNeighborsClassifier(500)
    knn = joblib.load(os.path.join(root_dir,'knn.pkl'))
    data_knn=data[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','CREDIT_ANNUITY_RATIO']].fillna(0)
    temp=data.copy()
    temp['TARGET']=label
    #fitting only on train data
    knn.fit(data_knn, y_train)
    joblib.dump(knn,os.path.join(root_dir,'knn.pkl'))
    #500 nearest neighbors for train
    nearest_500_neighbors_train = knn.kneighbors(data_knn)[1] 

    neighbors_target_mean_500_train=[]
    for i in nearest_500_neighbors_train:
        neighbors_target_mean_500_train.append(np.mean(temp['TARGET'].iloc[i]))#transform

    data['NEIGHBORS_TARGET_MEAN_500']=neighbors_target_mean_500_train
    return data


def feature_engineering_pipeline(X_train):
    y_train=pd.read_csv(os.path.join(root_dir,'y_train.csv'))
    bureau_agg=pd.read_csv(os.path.join(root_dir,'bureau_agg.csv'))
    credit_card_balance_agg=pd.read_csv(os.path.join(root_dir,'credit_card_balance_agg.csv'))
    POS_CASH_balance_agg=pd.read_csv(os.path.join(root_dir,'POS_CASH_balance_agg.csv'))
    installments_payments_agg=pd.read_csv(os.path.join(root_dir,'installments_payments_agg.csv'))
    previous_application_agg=pd.read_csv(os.path.join(root_dir,'previous_application_agg.csv'))
    
    bureau_agg=reduce_mem_usage(bureau_agg)
    credit_card_balance_agg=reduce_mem_usage(credit_card_balance_agg)
    POS_CASH_balance_agg=reduce_mem_usage(POS_CASH_balance_agg)
    installments_payments_agg=reduce_mem_usage(installments_payments_agg)
    previous_application_agg=reduce_mem_usage(previous_application_agg)
    
    #joing all the datset with set 1 for train data
    X_train=pd.merge(X_train,previous_application_agg,how='left',on='SK_ID_CURR')
    X_train=pd.merge(X_train,POS_CASH_balance_agg,how='left',on='SK_ID_CURR')
    X_train=pd.merge(X_train,installments_payments_agg,how='left',on='SK_ID_CURR')
    X_train=pd.merge(X_train,credit_card_balance_agg,how='left',on='SK_ID_CURR')
    X_train=pd.merge(X_train,bureau_agg,how='left',on='SK_ID_CURR')
    
    X_train=preprocessing_feature_engineering_application(X_train)
    #name of the missing columns that value will be predicted
    missing_columns=['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']
    temp_df=predict_missing_col(X_train,missing_columns)
    #adding the columns with the predicted values for train and test data
    X_train[['pred_EXT_SOURCE_1','pred_EXT_SOURCE_2','pred_EXT_SOURCE_3']]=temp_df[['pred_EXT_SOURCE_1',                                                                                    'pred_EXT_SOURCE_2','pred_EXT_SOURCE_3']]
    # memory management
    gc.enable()
    del temp_df
    gc.collect()
    
    X_train=calculating_category_wise_numerical_mean(X_train)
    
    X_train, temp_dict=onehot_encoding_categorical_features(X_train)
    
    X_train=kmeans_clustering(X_train)
    
    X_train=additional_feature_engineering(X_train,y_train)    
    
    return X_train

###################################################
@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/index')
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    clf = joblib.load(os.path.join(root_dir,'final_model_lgbm.pkl'))
    X_train = feature_engineering_pipeline(X_train)
    X_train = X_train.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    pred = clf.predict(X_train.drop('SK_ID_CURR',axis=1).values)
    if pred[0]:
        prediction = "Positive"
    else:
        prediction = "Negative"

    return jsonify({'prediction': prediction})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)


