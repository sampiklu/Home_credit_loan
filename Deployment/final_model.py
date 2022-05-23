#!/usr/bin/env python
# coding: utf-8


#Importing Libraries
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

root_dir='D:/AAIC dataset/self case study 1'
application_train = pd.read_csv(os.path.join(root_dir,'application_train.csv'))
time_span_list=[2,3,6,12,96]

def categorical_feature_preprocess(value):
    value=value.replace(':','')
    value= '_'.join(i for i in value.split() if i not in ['/','_','-']).lower()
    value=value.replace('/','_')
    return value

def preprocessing_feature_engineering_application(data):
    print("Shape of dataset: ",data.shape)
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
            temp_dict=dict(data_temp.groupby(cat_features).size()/len(data_temp))
            joblib.dump(temp_dict,os.path.join(root_dir,str(cat_features)+'_temp_dict.pkl'))
            #converting the features to frequency
            data_temp[cat_features]=data_temp[cat_features].apply(lambda x:temp_dict.get(x,0))
            
        #light GBM regressor model to predict the missing values
        
        lgbmr = LGBMRegressor(max_depth = 9, n_estimators = 5000, n_jobs = -1, learning_rate = 0.3, 
                                      random_state = 125)
        #fitting on the data
        lgbmr.fit(X = data_temp[rest_col], y = data_temp[col])
        joblib.dump(lgbmr,os.path.join(root_dir,'missing_col_model.pkl'))
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
            temp_dict=categorical_numerical_feature_engineering(data,numerical_feature,categorical_feature)
            joblib.dump(temp_dict,os.path.join(root_dir,str(numerical_feature)+str(categorical_feature)+'_temp_dict_160.pkl'))
            #creating new feature name
            new_feature_name=str(numerical_feature)+'_'+str(categorical_feature)+'_mean'
            #adding the feature in the dataframe
            data[new_feature_name]=data[categorical_feature].apply(lambda x: temp_dict.get(x,0) )
    return data   
            
            
def onehot_encoding_categorical_features(data):
    #categorical features
    categorical_feature_name_list=data.select_dtypes('object').columns.values
    temp_dict={}
    #converting all the categorical columns into one hot encoding   
    for cat_features in categorical_feature_name_list:
        #print(cat_features)
        #creating countervectorizer for each unique value
        vectorizer = CountVectorizer(tokenizer=lambda txt: txt.split())
        #fitting on the train data
        vectorizer.fit(data[cat_features].apply(lambda x: np.str_(x)))
        temp_dict[cat_features]=vectorizer.get_feature_names()
        one_hot_encoded=np.zeros(shape=(len(data),len(vectorizer.get_feature_names())))
        #applying transform on the train data
        for j in range(len(data)):
            for i in range(len(vectorizer.get_feature_names())):
                if data[cat_features][j]==vectorizer.get_feature_names()[i]:
                    one_hot_encoded[j][i]=1
        temp_train1=pd.DataFrame(one_hot_encoded,columns=[cat_features+'_'+i for i in vectorizer.get_feature_names()],index=data.index)

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
    std_scalar = StandardScaler()
    
    #applying fit and transform on the train data
    data_std=std_scalar.fit_transform(data.drop('SK_ID_CURR',axis=1))
    joblib.dump(std_scalar,os.path.join(root_dir,'std_scalar.pkl'))
    #Applying PCA to tranform data from 135 to 5 feature to apply k-means clustering
    #pca=joblib.load(os.path.join(root_dir,'pca.pkl'))
    pca = PCA(n_components=5)
    #applying fit and transform on the train data
    data_pca=pca.fit_transform(np.nan_to_num(data_std))
    joblib.dump(pca,os.path.join(root_dir,'pca.pkl'))
    
    #algo=joblib.load(os.path.join(root_dir,'kmeans.pkl'))
    algo = KMeans(n_clusters=2,random_state=15)
    #fitting on the train data
    algo.fit(data_pca)
    joblib.dump(algo,os.path.join(root_dir,'kmeans.pkl'))
    #adding the cluster label as a feature
    data['K_MEANS_CLUSTER_LABEL']=algo.labels_
    return data


def additional_feature_engineering(data,label):
   
    #https://www.kaggle.com/c/home-credit-default-risk/discussion/64784
    #from winner solution neighbors_target_mean_500: The mean TARGET value of the 500 closest neighbors 
    #of each row, where each neighborhood was defined by the three external sources and the credit/annuity ratio.
    #https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    #knn=joblib.load(os.path.join(root_dir,'knn.pkl'))
    knn = KNeighborsClassifier(500)
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

#https://towardsdatascience.com/how-to-learn-from-bigdata-files-on-low-memory-incremental-learning-d377282d38ff

def reduce_mem_usage(df):
    """ 
    iterate through all the columns of a dataframe and 
    modify the data type to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print(('Memory usage of dataframe is {:.2f}' 
                     'MB').format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max <                  np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max <                   np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max <                   np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max <                   np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max <                   np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max <                   np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')
    end_mem = df.memory_usage().sum() / 1024**2
    print(('Memory usage after optimization is: {:.2f}' 
                              'MB').format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) 
                                             / start_mem))
    
    return df

# Remove duplicate columns by values
def dedup_dataframe(data):
    unique_col = np.unique(data, axis = 1, return_index = True)[1]
    data = data.iloc[:, unique_col]
    return data

#get aggregated result on numerical features
def stat_support_num(support_data,app_id,support_id,prefix):
    #create empty dataframe
    data_numeric=pd.DataFrame()
    #storing the app_id
    data_numeric[app_id]=support_data[app_id]
    #for each column numeric only
    cnt=0
    for num_col in support_data:
        if (support_data[num_col].dtype=='int64' or support_data[num_col].dtype=='float64') and num_col !=app_id and num_col != support_id:
            data_numeric[num_col]=support_data[num_col]
            cnt+=1
            
    stats_groupby = data_numeric.groupby(app_id).agg(['count', 'mean', 'max', 'min', 'sum'])
    print("No of numerical featres : {} and after aggreagation no of total features : {}".format(cnt,5*cnt))
    column_name = []
    
    for j in stats_groupby.columns.levels[0]:
        if j != app_id:
            # for each stat
            for k in stats_groupby.columns.levels[1]:
                # make a new column name 
                column_name.append('{}_{}_{}'.format(prefix,j, k))
    
    stats_groupby.columns = column_name
    
    #assigning the id column
    stats_groupby[app_id]=stats_groupby.index
    
    # Remove duplicate columns by values
    stats_groupby=dedup_dataframe(stats_groupby)
    
    print("After removing duplicate values No of numerical features :",stats_groupby.shape[1])
    return stats_groupby
         
    
#get aggregated result on categorical features
def stat_support_cat(support_data,app_id,support_id,prefix):    
    #create empty dataframe
    data_categorical=pd.DataFrame()
    #storing the app_id
    data_categorical[app_id]=support_data[app_id]
    #for each column categorical only
    cnt=0
    for cat_col in support_data:
        if (support_data[cat_col].dtype!='int64' and support_data[cat_col].dtype!='float64') and cat_col !=app_id and cat_col != support_id:
            data_categorical[cat_col]=support_data[cat_col]
            cnt+=1
    data_categorical = pd.get_dummies(data_categorical)
    stats_groupby = data_categorical.groupby(app_id).agg(['sum'])
    
    print("No of categorical featres : {} and after aggreagation no of total features : {}".format(cnt,data_categorical.shape[1]))
    column_name = []
    
    for j in stats_groupby.columns.levels[0]:
        if j != app_id:
            # for each stat
            for k in stats_groupby.columns.levels[1]:
                # Make a new column name 
                column_name.append('{}_{}_{}'.format(prefix,j, k))
    
    stats_groupby.columns = column_name
    
    #assigning the id column
    stats_groupby[app_id]=stats_groupby.index
    
    # Remove duplicate columns by values
    stats_groupby=dedup_dataframe(stats_groupby)
    
    print("After removing duplicate values No of categorical features :",stats_groupby.shape[1])

    return stats_groupby

#aggragting all the results after applyting them separately numerical/categorical data types
def agg_support(support_data,app_id,support_id,prefix):
    cnt=0
    #checking numerical feature to aggregate
    for i in support_data.select_dtypes(exclude='object').columns.values:
        if("_ID_" not in i):
            cnt+=1
    
    #checking any categorical or numerical feature to aggregate        
    if(cnt==0 and support_data.select_dtypes('object').shape[1]==0):
        print("No feature to aggregate")
    #only numerical feature to aggregate            
    elif(cnt!=0 and support_data.select_dtypes('object').shape[1]==0):
        support_data_stat_num=stat_support_num(support_data,app_id,support_id,prefix)
        after_join_num_cat=support_data_stat_num
    
    #only categorical feature to aggregate            
    elif(cnt==0 and support_data.select_dtypes('object').shape[1]!=0):
        support_data_stat_cat=stat_support_cat(support_data,app_id,support_id,prefix)
        after_join_num_cat=support_data_stat_cat
        
    #if there is no categorical column
    else:
        support_data_stat_num=stat_support_num(support_data,app_id,support_id,prefix)
        support_data_stat_cat=stat_support_cat(support_data,app_id,support_id,prefix)
        after_join_num_cat=support_data_stat_num.merge(support_data_stat_cat, on = app_id, how = 'outer')
        
        # Remove duplicate columns by values
        after_join_num_cat=dedup_dataframe(after_join_num_cat)
    
    #after_join_num_cat.reset_index()
    print("Shape of the final dataframe: ",after_join_num_cat.shape)
    # memory management
    gc.enable()
    if cnt==0:
        del support_data_stat_cat
    elif support_data.select_dtypes('object').shape[1]==0:
        del support_data_stat_num
    elif cnt!=0 and support_data.select_dtypes('object').shape[1]==0:
        del support_data_stat_num,support_data_stat_cat
    gc.collect()
    return after_join_num_cat


def agg_bureau_balance(data,time_span_list,app_id,support_id):
    cnt=0
    for i in time_span_list:
        temp=data[(data['MONTHS_BALANCE']<=i)]
        temp.drop('MONTHS_BALANCE',axis=1,inplace=True)
        temp=agg_support(temp,app_id,support_id,'month_'+str(i))
        cnt+=1
        if cnt > 1:
            bureau_balance_agg=bureau_balance_agg.merge(temp, on = app_id, how = 'outer')

        else:
            bureau_balance_agg=temp
    bureau_balance_agg=bureau_balance_agg.fillna(0)
    return bureau_balance_agg
def monthwise_dpd_mean(data):
    temp=data.copy()
    temp['DPD_FLAG_IND']=temp['DPD_FLAG'].apply(lambda x: 1 if x=='Y' else 0)
    temp['quater']=temp['MONTHS_BALANCE'].apply(lambda x: 'quater_'+str(1) if x<3                                                                 else 'quater_'+str(int(x/3)+1))
    
    temp=temp[['SK_ID_BUREAU','quater','DPD_FLAG_IND']].groupby(['SK_ID_BUREAU','quater']).agg(['sum']).reset_index()
    
    temp.columns=['SK_ID_BUREAU','quater','DPD_FLAG_SUM_quater']
    temp_quater=temp[['SK_ID_BUREAU','DPD_FLAG_SUM_quater']].    groupby(['SK_ID_BUREAU']).agg(['mean']).reset_index()
    
    temp=data.copy()
    temp['DPD_FLAG_IND']=temp['DPD_FLAG'].apply(lambda x: 1 if x=='Y' else 0)
    temp['half_year']=temp['MONTHS_BALANCE'].apply(lambda x: 'half_year_'+str(1) if x<6                                                                 else 'half_year_'+str(int(x/6)+1))
    temp=temp[['SK_ID_BUREAU','half_year','DPD_FLAG_IND']].groupby(['SK_ID_BUREAU','half_year']).agg(['sum']).reset_index()
    
    temp.columns=['SK_ID_BUREAU','half_year','DPD_FLAG_SUM_half_year']
    temp_half_year=temp[['SK_ID_BUREAU','DPD_FLAG_SUM_half_year']].    groupby(['SK_ID_BUREAU']).agg(['mean']).reset_index()
    
    temp=data.copy()
    temp['DPD_FLAG_IND']=temp['DPD_FLAG'].apply(lambda x: 1 if x=='Y' else 0)
    temp['year']=temp['MONTHS_BALANCE'].apply(lambda x: 'year_'+str(1) if x<12                                                                 else 'year_'+str(int(x/12)+1))
    
    temp=temp[['SK_ID_BUREAU','year','DPD_FLAG_IND']].groupby(['SK_ID_BUREAU','year']).agg(['sum']).reset_index()
    
    temp.columns=['SK_ID_BUREAU','year','DPD_FLAG_SUM_year']
    temp_year=temp[['SK_ID_BUREAU','DPD_FLAG_SUM_year']].    groupby(['SK_ID_BUREAU']).agg(['mean']).reset_index()  
    
    bureau_agg_add=temp_quater.merge(temp_year, on = 'SK_ID_BUREAU', how = 'outer')
    bureau_agg_add=bureau_agg_add.merge(temp_half_year, on = 'SK_ID_BUREAU', how = 'outer')
    
    
    bureau_agg_add.columns=bureau_agg_add.columns.droplevel(1)
    return bureau_agg_add

def feature_engineering_bureau_balance(bureau_balance):
    #assigning a Defaulter flag as per the description 
    bureau_balance['DPD_FLAG']=bureau_balance['STATUS'].apply(lambda x: 'Y' if x in ['5', '1', '4', '3','0', '2'] else x)
    
    bureau_balance.drop('STATUS',axis=1,inplace=True)
    #converting to positive
    bureau_balance['MONTHS_BALANCE']=np.abs(bureau_balance['MONTHS_BALANCE'])
    
    #applying the aggrgation
    bureau_balance_agg=agg_bureau_balance(bureau_balance,time_span_list,'SK_ID_BUREAU','SK_ID_CURRENT')
    
    bureau_agg_add=monthwise_dpd_mean(bureau_balance)
    
    #adding all the feature engineered dataset
    bureau_balance_agg=bureau_balance_agg.merge(bureau_agg_add, on = 'SK_ID_BUREAU', how = 'outer')
    
    return bureau_balance_agg

def feature_engineering_bureau(bureau, bureau_balance_agg,zero_feature_importance):
    bureau[bureau['DAYS_ENDDATE_FACT']<-40000]['DAYS_ENDDATE_FACT']=np.nan
    bureau[bureau['DAYS_CREDIT_UPDATE']<-40000]['DAYS_CREDIT_UPDATE']=np.nan
    bureau[bureau['AMT_CREDIT_MAX_OVERDUE']>.8e8]['AMT_CREDIT_MAX_OVERDUE']=np.nan
    bureau[bureau['AMT_CREDIT_SUM']>5e8]['AMT_CREDIT_SUM']=np.nan
    bureau[bureau['AMT_CREDIT_SUM_DEBT']>1.5e8]['AMT_CREDIT_SUM_DEBT']=np.nan 
    bureau[bureau['AMT_CREDIT_SUM_OVERDUE']>3.5e5]['AMT_CREDIT_SUM_OVERDUE']=np.nan 
    bureau[bureau['AMT_ANNUITY']>1e8]['AMT_ANNUITY']=np.nan
    
    #domain knowledge
    bureau['ANNUITY_CREDIT_RATIO'] = bureau['AMT_ANNUITY'] / bureau['AMT_CREDIT_SUM']
    #joining bureau_balance data
    bureau=pd.merge(bureau,bureau_balance_agg,how='left',on='SK_ID_BUREAU')
    print("shape after joining the bureau_balance aggregated data: ", bureau.shape)
    
    #memory management
    gc.enable()
    #removing teamporary datas
    del bureau_balance_agg
    gc.collect()
    
    
    bureau_temp_active=bureau[(bureau['CREDIT_ACTIVE']=='Active')].copy()
    bureau_temp_active['SK_ID_CURR']=bureau[(bureau['CREDIT_ACTIVE']=='Active')]['SK_ID_CURR']
    bureau_temp_active['SK_ID_BUREAU']=bureau[(bureau['CREDIT_ACTIVE']=='Active')]['SK_ID_BUREAU']
    bureau_temp_active.drop('CREDIT_ACTIVE', axis=1, inplace=True)
    bureau_active_agg=agg_support(bureau_temp_active,'SK_ID_CURR','SK_ID_BUREAU','active_bureau')
    
    bureau_temp_not_active=bureau[(bureau['CREDIT_ACTIVE']!='Active')].copy()
    bureau_temp_not_active['SK_ID_CURR']=bureau_temp_not_active[(bureau['CREDIT_ACTIVE']!='Active')]['SK_ID_CURR']
    bureau_temp_not_active['SK_ID_BUREAU']=bureau_temp_not_active[(bureau['CREDIT_ACTIVE']!='Active')]['SK_ID_BUREAU']
    bureau_temp_not_active.drop('CREDIT_ACTIVE', axis=1, inplace=True)
    bureau_not_active_agg=agg_support(bureau_temp_not_active,'SK_ID_CURR','SK_ID_BUREAU','not_active_bureau')
    
    bureau_temp_secure=bureau[bureau['CREDIT_TYPE'].isin (['Another type of loan','Cash loan (non-earmarked)','Consumer credit','Credit card',                                           'Interbank credit','Loan for business development',                                           'Loan for working capital replenishment','Microloan', 'Mobile operator loan',    'Unknown type of loan'])].copy()
    bureau_temp_secure['SK_ID_CURR']=bureau[bureau['CREDIT_TYPE'].isin (['Another type of loan','Cash loan (non-earmarked)','Consumer credit','Credit card',                                           'Interbank credit','Loan for business development',                                           'Loan for working capital replenishment','Microloan', 'Mobile operator loan',    'Unknown type of loan'])]['SK_ID_CURR']
    bureau_temp_secure['SK_ID_BUREAU']=bureau[bureau['CREDIT_TYPE'].isin (['Another type of loan','Cash loan (non-earmarked)','Consumer credit','Credit card',                                           'Interbank credit','Loan for business development',                                           'Loan for working capital replenishment','Microloan', 'Mobile operator loan',    'Unknown type of loan'])]['SK_ID_BUREAU']
    bureau_secure_agg=agg_support(bureau_temp_secure,'SK_ID_CURR','SK_ID_BUREAU','active_bureau')
    
    bureau_temp_unsecure=bureau[~bureau['CREDIT_TYPE'].isin (['Another type of loan','Cash loan (non-earmarked)','Consumer credit','Credit card',                                           'Interbank credit','Loan for business development',                                           'Loan for working capital replenishment','Microloan', 'Mobile operator loan',    'Unknown type of loan'])].copy()
    bureau_temp_unsecure['SK_ID_CURR']=bureau[~bureau['CREDIT_TYPE'].isin (['Another type of loan','Cash loan (non-earmarked)','Consumer credit','Credit card',                                           'Interbank credit','Loan for business development',                                           'Loan for working capital replenishment','Microloan', 'Mobile operator loan',    'Unknown type of loan'])]['SK_ID_CURR']
    bureau_temp_unsecure['SK_ID_BUREAU']=bureau[~bureau['CREDIT_TYPE'].isin (['Another type of loan','Cash loan (non-earmarked)','Consumer credit','Credit card',                                           'Interbank credit','Loan for business development',                                           'Loan for working capital replenishment','Microloan', 'Mobile operator loan',    'Unknown type of loan'])]['SK_ID_BUREAU']
    bureau_unsecure_agg=agg_support(bureau_temp_unsecure,'SK_ID_CURR','SK_ID_BUREAU','active_bureau')
    
    #applying aggrgation on all bureau data
    bureau_agg=agg_support(bureau,'SK_ID_CURR','SK_ID_BUREAU','all_bureau')
    
    
    # join bureau_active_agg,bureau_not_active_agg,bureau_secure_agg,bureau_unsecure_agg,bureau_agg
    bureau_agg_temp=bureau_active_agg.merge(bureau_not_active_agg, on = 'SK_ID_CURR', how = 'outer')
    bureau_agg_temp=bureau_secure_agg.merge(bureau_agg_temp, on = 'SK_ID_CURR', how = 'outer')
    bureau_agg_temp=bureau_unsecure_agg.merge(bureau_agg_temp, on = 'SK_ID_CURR', how = 'outer')
    bureau_agg=bureau_agg.merge(bureau_agg_temp, on = 'SK_ID_CURR', how = 'outer')
    
    #memory management
    gc.enable()
    #removing teamporary datas
    del bureau_active_agg,bureau_not_active_agg,bureau_secure_agg,bureau_unsecure_agg,bureau_agg_temp
    gc.collect()
    
    bureau_agg.drop(columns=zero_feature_importance,axis=1,inplace=True)
    return bureau_agg

def agg_credit_card_balance(data,time_span_list,app_id,support_id):
    cnt=0
    for i in time_span_list:
        temp=data[(data['MONTHS_BALANCE']<=i)]
        temp.drop('MONTHS_BALANCE',axis=1,inplace=True)
        temp=agg_support(temp,app_id,support_id,'month_'+str(i))
        cnt+=1
        if cnt > 1:
            credit_card_balance_agg=credit_card_balance_agg.merge(temp, on = app_id, how = 'outer')

        else:
            credit_card_balance_agg=temp
    credit_card_balance_agg=credit_card_balance_agg.fillna(0)
    return credit_card_balance_agg

def monthwise_mean_credit_card_balance(data,freq,id_col):
    l1=['DPD_FLAG','AMT_BALANCE_ZERO_FLAG','AMT_BALANCE','AMT_BALANCE_PLUS_LIMIT','LESS_PAYMENT','AMT_CREDIT_LIMIT_ACTUAL','AMT_DRAWINGS_ATM_CURRENT','AMT_DRAWINGS_CURRENT','AMT_DRAWINGS_OTHER_CURRENT','AMT_DRAWINGS_POS_CURRENT','AMT_INST_MIN_REGULARITY','AMT_PAYMENT_CURRENT','AMT_PAYMENT_TOTAL_CURRENT','AMT_RECEIVABLE_PRINCIPAL','AMT_RECIVABLE','AMT_TOTAL_RECEIVABLE', 'CNT_DRAWINGS_ATM_CURRENT','CNT_DRAWINGS_CURRENT', 'CNT_DRAWINGS_OTHER_CURRENT']
    
    temp=data[l1].copy()
    temp[id_col]=data[id_col]
    temp['month_'+str(freq)]=data['MONTHS_BALANCE'].apply(lambda x: 'month_'+str(1) if x<freq                                                                 else 'month_'+str(int(x/freq)+1))

    temp=temp.groupby([id_col,'month_'+str(freq)]).agg(['sum']).reset_index()
    temp.columns=temp.columns.droplevel(1)
    temp.columns=[str(i)+'_month_'+str(freq) if i not in [id_col,'month_'+str(freq)] else i for i in temp.columns]
    temp_quater=temp.groupby([id_col]).agg(['mean']).reset_index()
    temp_quater.columns=temp_quater.columns.droplevel(1)
    return temp_quater
    

def feature_engineering_credit_card_balance(credit_card_balance,zero_feature_importance):
    credit_card_balance['MONTHS_BALANCE']=np.abs(credit_card_balance['MONTHS_BALANCE'])
    #balance cleared to zero flag
    credit_card_balance['AMT_BALANCE_ZERO_FLAG'] = credit_card_balance['AMT_BALANCE'].apply(lambda x: 1 if x==0 else 0)
    #over balance max limit
    credit_card_balance['AMT_BALANCE_PLUS_LIMIT'] = (credit_card_balance['AMT_BALANCE'] - credit_card_balance['AMT_CREDIT_LIMIT_ACTUAL']).apply( lambda x: 1 if x>0 else 0)
    #less payment
    credit_card_balance['LESS_PAYMENT'] = (credit_card_balance['AMT_INST_MIN_REGULARITY']-credit_card_balance['AMT_PAYMENT_CURRENT']).apply( lambda x: 1 if x>0 else 0)
    #dpd flag
    credit_card_balance['DPD_FLAG'] = credit_card_balance['SK_DPD'].apply( lambda x: 1 if x>0 else 0)
    
    #appying the aggreagtion
    credit_card_balance_agg=agg_credit_card_balance(credit_card_balance,time_span_list,'SK_ID_CURR','SK_ID_PREV')
    
    temp_quater=monthwise_mean_credit_card_balance(credit_card_balance,4,'SK_ID_CURR')
    temp_half_year=monthwise_mean_credit_card_balance(credit_card_balance,6,'SK_ID_CURR')
    temp_year=monthwise_mean_credit_card_balance(credit_card_balance,12,'SK_ID_CURR')
    
    credit_card_balance_agg=credit_card_balance_agg.merge(temp_quater, on = 'SK_ID_CURR', how = 'outer')
    credit_card_balance_agg=credit_card_balance_agg.merge(temp_half_year, on = 'SK_ID_CURR', how = 'outer')
    credit_card_balance_agg=credit_card_balance_agg.merge(temp_year, on = 'SK_ID_CURR', how = 'outer')
    
    credit_card_balance_agg.drop(columns=zero_feature_importance,axis=1,inplace=True)
    
    return credit_card_balance_agg

def agg_POS_CASH_balance(data,time_span_list,app_id,support_id):
    cnt=0
    for i in time_span_list:
        temp=data[(data['MONTHS_BALANCE']<=i)]
        temp.drop('MONTHS_BALANCE',axis=1,inplace=True)
        temp=agg_support(temp,app_id,support_id,'month_'+str(i))
        cnt+=1
        if cnt > 1:
            POS_CASH_balance_agg=POS_CASH_balance_agg.merge(temp, on = app_id, how = 'outer')

        else:
            POS_CASH_balance_agg=temp
    POS_CASH_balance_agg=POS_CASH_balance_agg.fillna(0)
    return POS_CASH_balance_agg

def monthwise_mean_POS_CASH_balance(data,freq,id_col):
    l1=['WAVED_SK_DPD_RATIO','DPD_FLAG']
    
    temp=data[l1].copy()
    temp[id_col]=data[id_col]
    temp['month_'+str(freq)]=data['MONTHS_BALANCE'].apply(lambda x: 'month_'+str(1) if x<freq                                                                 else 'month_'+str(int(x/freq)+1))

    temp=temp.groupby([id_col,'month_'+str(freq)]).agg(['sum']).reset_index()
    temp.columns=temp.columns.droplevel(1)
    temp.columns=[str(i)+'_month_'+str(freq) if i not in [id_col,'month_'+str(freq)] else i for i in temp.columns]
    temp_quater=temp.groupby([id_col]).agg(['mean']).reset_index()
    temp_quater.columns=temp_quater.columns.droplevel(1)
    return temp_quater
    
    
def feature_engineering_POS_CASH_balance(POS_CASH_balance,zero_feature_importance):
    POS_CASH_balance['MONTHS_BALANCE']=np.abs(POS_CASH_balance['MONTHS_BALANCE'])
    #Total no of installments paid and to be paid
    POS_CASH_balance['TOTAL_CNT_INSTALMENT']=POS_CASH_balance['CNT_INSTALMENT']+POS_CASH_balance['CNT_INSTALMENT_FUTURE']
    #ignored dpd to dpd raio
    POS_CASH_balance['WAVED_SK_DPD_RATIO']=POS_CASH_balance['SK_DPD_DEF']/POS_CASH_balance['SK_DPD']
    #dpd-ignored dpd
    POS_CASH_balance['ACTUAL_SK_DPD']=POS_CASH_balance['SK_DPD']+POS_CASH_balance['SK_DPD_DEF']
    #paid installments
    POS_CASH_balance['INSTALLMENTS_PAID'] = POS_CASH_balance['CNT_INSTALMENT'] - POS_CASH_balance['CNT_INSTALMENT_FUTURE']
    #dpd flag
    POS_CASH_balance['DPD_FLAG'] = POS_CASH_balance['SK_DPD'].apply( lambda x: 1 if x>0 else 0)
    #applying the aggregation
    POS_CASH_balance_agg=agg_POS_CASH_balance(POS_CASH_balance,time_span_list,'SK_ID_CURR','SK_ID_PREV')
    
    
    
    POS_CASH_balance_active=POS_CASH_balance[(POS_CASH_balance['NAME_CONTRACT_STATUS']=='Active')].copy()
    
    POS_CASH_balance_agg_active=agg_support(POS_CASH_balance_active,'SK_ID_CURR','SK_ID_PREV','active_pos')
    
    POS_CASH_balance_completed=POS_CASH_balance[(POS_CASH_balance['NAME_CONTRACT_STATUS']=='Completed')].copy()
    
    POS_CASH_balance_agg_completed=agg_support(POS_CASH_balance_completed,'SK_ID_CURR','SK_ID_PREV','completed_pos')
    
    # join POS_CASH_balance_agg,POS_CASH_balance_agg_active,POS_CASH_balance_agg_completed,POS_CASH_balance_agg_overall
    temp=POS_CASH_balance_agg.merge(POS_CASH_balance_agg_active, on = 'SK_ID_CURR', how = 'outer')
    POS_CASH_balance_agg=POS_CASH_balance_agg_completed.merge(temp, on = 'SK_ID_CURR', how = 'outer')
    
    
    #memory management
    gc.enable()
    del POS_CASH_balance_agg_active,POS_CASH_balance_agg_completed,temp
    gc.collect()
    
    temp_quater=monthwise_mean_POS_CASH_balance(POS_CASH_balance,4,'SK_ID_CURR')
    temp_half_year=monthwise_mean_POS_CASH_balance(POS_CASH_balance,6,'SK_ID_CURR')
    temp_year=monthwise_mean_POS_CASH_balance(POS_CASH_balance,12,'SK_ID_CURR')
    
    
    POS_CASH_balance_agg=POS_CASH_balance_agg.merge(temp_quater, on = 'SK_ID_CURR', how = 'outer')
    POS_CASH_balance_agg=POS_CASH_balance_agg.merge(temp_half_year, on = 'SK_ID_CURR', how = 'outer')
    POS_CASH_balance_agg=POS_CASH_balance_agg.merge(temp_year, on = 'SK_ID_CURR', how = 'outer')
    
    POS_CASH_balance_agg.drop(columns=zero_feature_importance,axis=1,inplace=True)
    
    return POS_CASH_balance_agg

def feature_engineering_installments_payments(installments_payments,zero_feature_importance):

    #payment was after the installment date
    installments_payments['LATE_PAYMENT_FLAG'] = (installments_payments['DAYS_ENTRY_PAYMENT'] - installments_payments['DAYS_INSTALMENT']).apply(lambda x: 1 if x>0 else 0)
    #payment was less than the installment amount
    installments_payments['LESS_PAYMENT_FLAG'] = (installments_payments['AMT_PAYMENT'] - installments_payments['AMT_INSTALMENT']).apply(lambda x: 1 if x>0 else 0)
    
    
    #applying the aggreagtion function
    installments_payments_agg=agg_support(installments_payments,'SK_ID_CURR','SK_ID_PREV','installments_payments')
    
    installments_payments_agg.drop(columns=zero_feature_importance,axis=1,inplace=True)
    
    return installments_payments_agg

def feature_engineering_previous_application(previous_application,zero_feature_importance):
    
    #as er the EDA removing 365243 in 'DAYS_X' features
    previous_application['DAYS_FIRST_DRAWING'][previous_application['DAYS_FIRST_DRAWING'] == 365243.0] = np.nan
    previous_application['DAYS_FIRST_DUE'][previous_application['DAYS_FIRST_DUE'] == 365243.0] = np.nan
    previous_application['DAYS_LAST_DUE_1ST_VERSION'][previous_application['DAYS_LAST_DUE_1ST_VERSION'] == 365243.0] = np.nan
    previous_application['DAYS_LAST_DUE'][previous_application['DAYS_LAST_DUE'] == 365243.0] = np.nan
    previous_application['DAYS_TERMINATION'][previous_application['DAYS_TERMINATION'] == 365243.0] = np.nan
    #domain knowledge and winner solution 10th place
    #https://www.kaggle.com/c/home-credit-default-risk/discussion/64598
    previous_application['ANNUITY_CREDIT_RATIO'] = previous_application['AMT_ANNUITY'] / previous_application['AMT_CREDIT']
    previous_application['CREDIT_APPLICATION_AMT_DIFF'] = previous_application['AMT_CREDIT'] - previous_application['AMT_APPLICATION']
    previous_application['INTEREST'] = previous_application['CNT_PAYMENT']*previous_application['AMT_ANNUITY'] - previous_application['AMT_CREDIT']
    previous_application['INTEREST_RATE'] = 2*12*previous_application['INTEREST']/(previous_application['AMT_CREDIT']*(previous_application['CNT_PAYMENT']+1))
    previous_application['INTEREST_SHARE'] = previous_application['INTEREST']/previous_application['AMT_CREDIT']
    
    #applying aggregation function on previous_application data
    previous_application_agg=agg_support(previous_application,'SK_ID_CURR','SK_ID_PREV','previous')
    
    previous_application_agg.drop(columns=zero_feature_importance,axis=1,inplace=True)
    
    return previous_application_agg

##########################################################################


#storing the dependent feature
y_train = application_train['TARGET'].values
#droppiing the dependent feature
X_train = application_train.drop(['TARGET'],axis=1)

def feature_engineering_pipeline(X_train,y_train):
    bureau_balance = pd.read_csv(os.path.join(root_dir,'bureau_balance.csv'))
    #storing the files in dataframe for further analysis
    bureau = pd.read_csv(os.path.join(root_dir,'bureau.csv'))
    
    #storing the files in dataframe for further analysis
    credit_card_balance = pd.read_csv(os.path.join(root_dir,'credit_card_balance.csv'))
    
    #storing the files in dataframe for further analysis
    POS_CASH_balance = pd.read_csv(os.path.join(root_dir,'POS_CASH_balance.csv'))
    
    #storing the files in dataframe for further analysis
    installments_payments = pd.read_csv(os.path.join(root_dir,'installments_payments.csv'))
    
    #storing the files in dataframe for further analysis
    previous_application = pd.read_csv(os.path.join(root_dir,'previous_application.csv'))
    
    time_span_list=[2,3,6,12,96]
    
    with open (os.path.join(root_dir,'zero_feature_importance_bureau.pkl'), 'rb') as fp:
        zero_feature_importance_bureau = pickle.load(fp)
        
    with open (os.path.join(root_dir,'zero_feature_importance_credit_card_balance.pkl'), 'rb') as fp:
        zero_feature_importance_credit_card_balance = pickle.load(fp)
        
    with open (os.path.join(root_dir,'zero_feature_importance_POS_CASH_balance.pkl'), 'rb') as fp:
        zero_feature_importance_POS_CASH_balance = pickle.load(fp)
        
    with open (os.path.join(root_dir,'zero_feature_importance_installments_payments.pkl'), 'rb') as fp:
        zero_feature_importance_installments_payments = pickle.load(fp)
    
    with open (os.path.join(root_dir,'zero_feature_importance_previous_application.pkl'), 'rb') as fp:
        zero_feature_importance_previous_application = pickle.load(fp)    
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
    joblib.dump(temp_dict,os.path.join(root_dir,'one_hot_encoding_feature.pkl'))
    
    X_train=kmeans_clustering(X_train)
    
    X_train=additional_feature_engineering(X_train,y_train)
    
    X_train=reduce_mem_usage(X_train)
    
    bureau_balance_agg=feature_engineering_bureau_balance(bureau_balance)
    
    bureau_agg=feature_engineering_bureau(bureau, bureau_balance_agg,zero_feature_importance_bureau)
    
    credit_card_balance_agg=feature_engineering_credit_card_balance(credit_card_balance,zero_feature_importance_credit_card_balance)
    
    POS_CASH_balance_agg=feature_engineering_POS_CASH_balance(POS_CASH_balance,zero_feature_importance_POS_CASH_balance)
    
    installments_payments_agg=feature_engineering_installments_payments(installments_payments,zero_feature_importance_installments_payments)
    
    previous_application_agg=feature_engineering_previous_application(previous_application,zero_feature_importance_previous_application)
    
    bureau_agg.to_csv(os.path.join(root_dir,'bureau_agg.csv'),index=False)
    credit_card_balance_agg.to_csv(os.path.join(root_dir,'credit_card_balance_agg.csv'),index=False)
    POS_CASH_balance_agg.to_csv(os.path.join(root_dir,'POS_CASH_balance_agg.csv'),index=False)
    installments_payments_agg.to_csv(os.path.join(root_dir,'installments_payments_agg.csv'),index=False)
    previous_application_agg.to_csv(os.path.join(root_dir,'previous_application_agg.csv'),index=False)
    
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
    
    
    return X_train,y_train

def predict(X_train,y_train):
    #https://stackoverflow.com/questions/60582050/lightgbmerror-do-not-support-special-json-characters-in-feature-name-the-same
    X_train = X_train.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    train=X_train.drop('SK_ID_CURR',axis=1).values
    #harrcoding the best params and fiiting with the train data
    clf_best = lgb.LGBMClassifier(max_depth=-1, random_state=30, silent=True, n_jobs=-1,                              colsample_bytree=0.9845830012866338, metric='None',                              min_child_samples=490, min_child_weight=1e-05, n_estimators=500,                              num_leaves=21, reg_alpha=50, reg_lambda=10,                              scale_pos_weight=2, subsample=0.7032798646617617)
    clf_best.fit(train,y_train)
    
    joblib.dump(clf_best,os.path.join(root_dir,'final_model_lgbm.pkl'))
    
    y_train_pred=clf_best.predict_proba(train)
    pred_label=[]
    for pred in y_train_pred:
        if pred[0] >= 0.6:
            pred_label.append(1)
        else:
            pred_label.append(0)
    print ("train AUC score: ",roc_auc_score(y_train, y_train_pred[:,1]))
    fpr_, tpr_, _ = roc_curve(y_train, y_train_pred[:,1])
    
    plt.plot(fpr_, tpr_,label='train data')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC curve')
    plt.legend()
    plt.show()
    
    return roc_auc_score(y_train, y_train_pred[:,1])

X_train,y_train=feature_engineering_pipeline(X_train,y_train)    

auc_score=predict(X_train,y_train)