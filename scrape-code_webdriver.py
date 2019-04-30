# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 10:17:04 2019

@author: vrush
"""

import pandas
from sklearn import neighbors,cluster,preprocessing,svm,decomposition,model_selection,metrics,ensemble,feature_extraction,naive_bayes,linear_model
import numpy
import collections
import operator
import matplotlib.pyplot as plt
import scipy
import seaborn
import xgboost
import scipy.stats
seaborn.set(rc={'figure.figsize':(11.7,8.27)})
from bs4 import BeautifulSoup
import requests
import selenium
from selenium import webdriver
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import scale

def parse_amenities(am_st):
    am_st=am_st.translate(am_st.maketrans('','','{}'))
    arr=am_st.split(',')
    am=[s.translate(s.maketrans('','','"')).strip() for s in arr if s!='']
    return(am)

'''removes/replaces the rows/values having targetColumn greater than 99th percentile for each
category level in the column passed to function'''
def handleAbove99tileByCategory(df,columnName,targetColumn,replaceWithCutoff=False):
    unique_vals=df[columnName].dropna().unique()
    print('Working on Column: ',columnName,' Target Column: ',targetColumn)
    print('Category wise 99th percentile')
    for val in unique_vals:
        subset=df[df[columnName]==val]
        cutoffpercentile=numpy.nanpercentile(subset[targetColumn],q=[99])
        print(columnName.upper(),'-',val,':',numpy.ceil(cutoffpercentile[0]))
        if(replaceWithCutoff==False):
            df=df.drop(df[(df[columnName]==val) & (df[targetColumn]>cutoffpercentile[0])].index)
        else:
            df.loc[df[(df[columnName]==val) & (df[targetColumn]>cutoffpercentile[0])].index,targetColumn]=numpy.ceil(cutoffpercentile)
    
    return(df)
 
  
def updatePrice(df):
    '''update from scrapped data'''    
    merged_scrape=pandas.read_csv('C:\\Users\\gudea\\Desktop\\Python\\airbnb\\scrapped_data\\combinedscrape_180119.csv')
    merged_scrape=merged_scrape.fillna(0)#the records having NaN are no longer listed on airbnb.so we will remove them
    for url,new_price in zip(merged_scrape.listing_url,merged_scrape.new_price):    
        df.loc[df.listing_url==url,'price']=new_price
    
    #remove  0 price listings as they are no longer listed on airbnb and we are not sure of their price
    df=df[df.price>0].copy()
    return(df)


def removeUnwantedColumns(df):
    '''remove the unwanted columns.mostly those having free flowing text'''
    df=df.drop(['id','scrape_id','last_scraped','name','summary','space'
                                  ,'description','experiences_offered','access'
                                  ,'interaction','neighborhood_overview','notes','transit'
                                  ,'house_rules','thumbnail_url','medium_url','picture_url','xl_picture_url'
                                  ,'host_url','host_name','host_since','host_location'
                                  ,'host_about','host_picture_url','host_listings_count'
                                  ,'host_acceptance_rate','host_thumbnail_url'
                                  ,'neighbourhood_group_cleansed','market','country_code','country'
                                  ,'weekly_price','monthly_price','calendar_updated'
                                  ,'has_availability','availability_30','availability_60'
                                  ,'availability_365','calendar_last_scraped'
                                  ,'requires_license','license','jurisdiction_names'
                                  ,'is_business_travel_ready','require_guest_profile_picture'
                                  ,'require_guest_phone_verification','calculated_host_listings_count','host_verifications'
                                  ,'host_neighbourhood','is_location_exact'],axis=1)
    
     
    
    '''after EDA, delete some more columns'''
    '''majority are one value so delete host_has_profile_pic'''
    df=df.drop(['host_has_profile_pic',],axis=1)
    '''as we have the neighbourhood, we dont need the zipcode'''
    df=df.drop(['zipcode'],axis=1)
    '''remove square feet as majority values are blank'''
    df=df.drop(['square_feet'],axis=1)
    '''remove columns that leak future information like review.remove unwanted reviews columns. we will only keep the main one'''
    df=df.drop(['number_of_reviews','review_scores_value','first_review','last_review','review_scores_accuracy','review_scores_rating','review_scores_cleanliness','review_scores_checkin','review_scores_communication','review_scores_location','reviews_per_month'],axis=1)
    '''as we have london_borough we dont need the state'''
    df=df.drop(['state'],axis=1)
    '''now in cleaned data neighbourhood stands for actual neighbourhood whereas 
    neighbourhood_cleansed stands for the borough where this neighbourhood is located'''
    '''to avoid confustion we'll rename it to london_borough'''
    df=df.rename(index=int,columns={'neighbourhood_cleansed':'london_borough'})
    '''remove street,smart_location,city as we have london_borough'''
    df=df.drop(['street','smart_location','city'],axis=1)
    '''as we are predicting only the rental price and not the cleaning fee or security deposit.so we can remove those columns'''
    df=df.drop(['security_deposit','cleaning_fee'],axis=1)
    '''since host_response_rate and host_response_time are missing together for 35% of the records and since they dont 
    have a strong corelation with the DV, we can drop them for now.'''
    df=df.drop(['host_response_rate','host_response_time'],axis=1)
    '''as majority are real bed we will delete this column'''
    df=df.drop(['bed_type'],axis=1) 
    return(df)    


def cleanData(df):
    '''DATA CLEANING'''
    '''cleaning special characters from certain numerical columns'''
    df['price']=df['price'].str.replace('$','').str.replace(',','').astype('float')
    df['extra_people']=df['extra_people'].str.replace('$','').str.replace(',','').astype('float')
    
    '''convert binary variables to numerical'''
    df.host_is_superhost=df.host_is_superhost.map({'f':0,'t':1})
    df.host_identity_verified=df.host_identity_verified.map({'f':0,'t':1})
    df.instant_bookable=df.instant_bookable.map({'f':0,'t':1})
         
    '''delete the rows having 0 as price as they are noise'''
    df=df[df.price>0].copy()   
    '''the bedrooms which are actually marked as 0 are actually studio apartments.
     so replace the number of bedrooms by 1'''
    df.loc[df.bedrooms==0,'bedrooms']=1
    '''many listings have misleading info they are giving one bedroom for rent but have mentioned the total number of rooms in the house'''
    df.loc[(df.bedrooms>1)&(df.room_type=='Private room'),'bedrooms' ]=1
    '''similar problem they have mentioned how many people the house can accommodate but the price mentioned is for 1 person'''
    df.loc[(df.bedrooms==1)&(df.room_type=='Shared room'),'accommodates']=1
    '''we are restricting the scope to 5 bedrooms'''
    df=df[df.bedrooms<=5].copy()
    '''the hostels have many shared bathrooms which can affect the model. so for hostels we will cap bathroom to 1'''
    df.loc[df.property_type=='Hostel','bathrooms']=1
    return(df)


def featureEngineeringOfAmenities(df):
    '''clean the amenities field and convert into list'''
    df['amenities']=df.apply(lambda x:parse_amenities(x.amenities),axis=1)
    '''OHE the data of ammenities'''
    '''we cannot use getdummies here as each row has a list of amenities.so we are using MultiLabelBinarizer '''
    mlb=preprocessing.MultiLabelBinarizer()
    amenities=pandas.DataFrame(mlb.fit_transform(df['amenities']),columns=mlb.classes_, index=df.index)    
    amenities=amenities.drop(['translation missing: en.hosting_amenity_49','translation missing: en.hosting_amenity_50'],axis=1)
    
    '''check corelation between amenities'''
    cor_amn=pandas.DataFrame(amenities.corr())
    for col in cor_amn.columns:
        cor_amn.loc[col,col]=numpy.nan
    high_cor=cor_amn.where(cor_amn.abs().gt(.8))
    high_cor=high_cor.dropna(axis=1,how='all')
    high_cor=high_cor.dropna(axis=0,how='all')
    
    '''highly corelated with bathroom essentials. so remove them'''
    amenities=amenities.drop(['Bath towel','Bedroom comforts','Body soap','Toilet paper'],axis=1)
    '''highly corelated with cooking basics. so remove them'''
    amenities=amenities.drop(['Dishes and silverware','Oven','Refrigerator','Stove','Microwave'],axis=1)
    '''highly corelated with self check in.so remove them'''
    amenities=amenities.drop(['Lockbox'],axis=1)
    '''highly corelated to toilet so remove'''
    amenities=amenities.drop(['Wide clearance to shower'],axis=1)
    
    '''delete original amenities column'''
    df=df.drop(['amenities'],axis=1) 
    '''merge amenities with original data'''
    df=pandas.DataFrame(pandas.concat([df,amenities],axis=1))
    
    '''remove amenities which are most common or most uncommon'''
    amenities_dist=dict()
    unbalanced_amenities=list()
    for i in amenities.columns:
        freq=df[i].sum().item()
        amenities_dist.update({i:freq})
        if(freq<1500 or freq>70000):
            unbalanced_amenities.append(i)
    '''sort by most common'''
    amenities_dist=dict(sorted(amenities_dist.items(),key=operator.itemgetter(1),reverse=True))
    '''get rid of amenities which have less than 3% of 0's or 1's in each column'''
    df=df.drop(unbalanced_amenities,axis=1)
    return(df)
    
def reducePropertyTypeLevels(df):
    Property_Type_Count=collections.Counter(df.property_type)
    '''Counting the number of properties which are below and equal to 200'''
    Property_Count_Below_100=list()
    Property_Count_Below_100=[key for key,value in Property_Type_Count.items() if value<=100]
    '''Replacing the value of properties with others where count is below or equal to 10'''        
    df['property_type'].replace(Property_Count_Below_100,"Other",inplace=True)
    return(df)

def missingValueImpute(df):
    df.bathrooms=df.bathrooms.fillna(round(df.bathrooms.mean()))
    df.host_is_superhost=df.host_is_superhost.fillna(0)
    df.host_identity_verified=df.host_identity_verified.fillna(0)
    df.host_total_listings_count=df.host_total_listings_count.fillna(0)
    return(df)



'''LOGIC STARTS''' 
#load the data
london_data=pandas.DataFrame(pandas.read_csv('C:\\Users\\gudea\\Desktop\\Python\\airbnb\\london_listings.csv',low_memory=False,na_values=['',' ',numpy.NAN,numpy.NaN,'NA','N/A']))
#remove unwanted columns
london_data=removeUnwantedColumns(london_data)
#clean data
london_data=cleanData(london_data)
#update the prices we have scraped for some records to check their authenticity
london_data=updatePrice(london_data)

'''CHECK CORRELATION'''
corr=round(london_data.corr(),2)
seaborn.heatmap(corr,annot=True)
#accomodates and beds has strong corelation.so drop beds
london_data=london_data.drop(['beds'],axis=1) 



'''VISUALIZATION'''
seaborn.pointplot(x=london_data.bedrooms,y=london_data.price)
seaborn.barplot(y=london_data.london_borough,x=london_data.price)
seaborn.barplot(y=london_data.property_type,x=london_data.price)
seaborn.boxplot(x=london_data.bedrooms)
seaborn.FacetGrid(london_data[['latitude','longitude','london_borough']],hue='london_borough').map(plt.scatter,'latitude','longitude').add_legend()
seaborn.regplot(x=london_data.bedrooms,y=london_data.price)
seaborn.distplot(london_data.price)
seaborn.regplot(x=london_data.bedrooms,y=london_data.bathrooms)


'''OUTLIER TREATMENT'''
#extreme outliers removal based on the number of bedrooms
london_data=handleAbove99tileByCategory(london_data,'bedrooms','price')
#handle outliers in case of number of bathrooms for other properties
london_data=handleAbove99tileByCategory(london_data,'bedrooms','bathrooms',True)


'''FEATURE ENGINEERING'''
#reset index the dataframe as we have deleted some rows
london_data=london_data.reset_index(drop=True)
#Feature Engineering of Amenities
london_data=featureEngineeringOfAmenities(london_data)
#Reduce levels in Property Type
seaborn.barplot(y=london_data.property_type,x=london_data.price)
london_data=reducePropertyTypeLevels(london_data)
#remove the columns which were kept for debuging.remove neighbourhood as we will be doing clustering on lat,long
london_data=london_data. drop(['host_id','neighbourhood','listing_url'],axis=1) 
#bin the price column
bins=[0,58,110,210,2001]
names=[1,2,3,4]
price_bins=pandas.cut(london_data.price,bins=bins,labels=names).astype('int')
london_data['price_bins']=price_bins
#OHE the categorical columns
london_data=pandas.get_dummies(london_data)

'''SPLIT INTO TRAIN AND TEST'''
X=london_data.drop(['price','price_bins'],axis=1).copy()
Y=london_data[['price','price_bins']].copy()
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y,random_state = 18,train_size=0.7)

#reset index for train and test
x_train=x_train.reset_index(drop=True)
y_train=y_train.reset_index(drop=True)
x_test=x_test.reset_index(drop=True)
y_test=y_test.reset_index(drop=True)

'''IMPUTE MISSING VALUES FOR TRAIN AND TEST SEPERATELY'''
x_train=missingValueImpute(x_train)
x_test=missingValueImpute(x_test)


'''CLUSTERING'''
#instead of using the neighbourhood categorical columns having many levels. we can create location
#clusters as it will avoid the need of creating many OHE columns
#run kmeans on train data
kmeans=KMeans(n_clusters=100).fit(x_train[['latitude','longitude']])
labels_train=kmeans.labels_
x_train['location_cluster']=pandas.Series(labels_train)

#then use this to predict the test location clusters
labels_test=kmeans.predict(x_test[['latitude','longitude']])
x_test['location_cluster']=pandas.Series(labels_test)

#as we have cluster, we can drop latitude and longitude for both train and test
x_train=x_train.drop(['latitude','longitude'],axis=1)
x_test=x_test.drop(['latitude','longitude'],axis=1)



'''CLASSIFICATION OF THE PRICE BINS'''
#Most of the time people have an idea about the price range in which their rental will fail. 
#For such people we can skip the classification stage and go directly to the price bucket specific regression .
#For users who have no idea about the price range, we can first classify in which price bucket their rental can fall in
#and then do the bucket specific regression

#RandomForest Classifier to predict the price bins
randomForestClassifier=ensemble.RandomForestClassifier(n_estimators=200,max_features=100,max_depth=15,min_samples_leaf=5)
randomForestClassifier.fit(x_train,y_train['price_bins'])
print(randomForestClassifier.score(x_train,y_train['price_bins']))
print(randomForestClassifier.score(x_test,y_test['price_bins']))
y_pred=randomForestClassifier.predict(x_test)
report=metrics.classification_report(y_test['price_bins'],y_pred)
print(report)

#Logistic Regression
logistic=linear_model.LogisticRegression().fit(scale(x_train),y_train['price_bins'])
print(logistic.score(scale(x_train),y_train['price_bins']))
print(logistic.score(scale(x_test),y_test['price_bins']))
y_pred=logistic.predict(scale(x_test))
report=metrics.classification_report(y_test['price_bins'],y_pred)
print(report)


'''REGRESSION'''

#A regression model is built for each of the price bins. So total 4 regression models will be built.
#Based on the output of the classification model, we will call the appropriate regression model

#Model for Price Bin 1#







"""END"""






















#rf=ensemble.RandomForestRegressor(n_estimators=100,max_features=100,min_samples_leaf=20)
#rf.fit(x_train,y_train)
#numpy.sqrt(metrics.mean_squared_error(y_train,rf.predict(x_train)))
#rf.score(x_train,y_train)
#rf.score(x_test,y_test)
#
#y_predicted=rf.predict(x_test)
#metrics.median_absolute_error(y_test,y_predicted)
#numpy.sqrt(metrics.mean_squared_error(y_test,y_predicted))
#
#residuals=y_test-y_predicted
#print(scipy.stats.skew(residuals))
#seaborn.distplot(residuals)
#'''Heteroskedasticity(non constant variance.error increases as price increases) can be seen from the plot (Funnel Shape)'''
#seaborn.residplot(x=y_predicted,y=residuals)
#
#imps=dict()
#for feature, importance in zip(x_train.columns, rf.feature_importances_):
#    imps[feature] = importance #add the name/value pair 
#imps=list(sorted(imps.items(),key=operator.itemgetter(1),reverse=True))
#
#rf_log_transformed.get_params
#rf_log_transformed=ensemble.RandomForestRegressor(n_estimators=200)
#rf_log_transformed.fit(x_train,scipy.special.boxcox(y_train,0))
#print(rf_log_transformed.score(x_train,scipy.special.boxcox(y_train,0)))
#print(rf_log_transformed.score(x_test,scipy.special.boxcox(y_test,0)))
#y_predicted_logtransformed=rf_log_transformed.predict(x_test)
#residuals_log=scipy.special.boxcox(y_test,0)-y_predicted_logtransformed
#print(scipy.stats.skew(residuals_log))
#sum(y_predicted_logtransformed)
#seaborn.distplot(residuals_log)
#seaborn.residplot(x=y_predicted_logtransformed,y=residuals_log).set(xlabel='Fitted',ylabel='Residuals')
#seaborn.regplot(x=scipy.special.inv_boxcox(y_predicted_logtransformed,0),y=y_test).set(xlabel='Predicted Price',ylabel='Actual Price')
#
#
#metrics.median_absolute_error(y_test,scipy.special.inv_boxcox(y_predicted_logtransformed,0))
#numpy.sqrt(metrics.mean_squared_error(y_test,scipy.special.inv_boxcox(y_predicted_logtransformed,0)))
#
#imps_rf=dict()
#for feature, importance in zip(x_train.columns, rf_log_transformed.feature_importances_):
#    imps_rf[feature] = importance #add the name/value pair 
#imps_rf=list(sorted(imps_rf.items(),key=operator.itemgetter(1),reverse=True))
#
#actual=y_test.reset_index(drop=True)
#normal_predict=pandas.Series(y_predicted).reset_index(drop=True)
#log_predict=pandas.Series(scipy.special.inv_boxcox(y_predicted_logtransformed,0)).reset_index(drop=True)
#
#compare=pandas.DataFrame(pandas.concat([actual,normal_predict,log_predict],axis=1))
#compare.columns=['Actual','Predicted','PredictedLog']
#
#
#
#from sklearn.pipeline import make_pipeline
#svr = make_pipeline(preprocessing.RobustScaler(), linear_model.Lasso())
#
#svr.fit(x_train,scipy.special.boxcox(y_train,0))
#
#print(svr.score(x_test,scipy.special.boxcox(y_test,0)))
#y_predicted_ridge=svr.predict(x_test)
#print(metrics.median_absolute_error(y_test,scipy.special.inv_boxcox(y_predicted_ridge,0)))
#numpy.sqrt(metrics.mean_squared_error(y_test,scipy.special.inv_boxcox(y_predicted_ridge,0)))
#seaborn.regplot(y=y_predicted_ridge,x=x_test)
#
#actual=y_test.reset_index(drop=True)
#ridge_predict=pandas.Series(scipy.special.inv_boxcox(y_predicted_ridge,0)).reset_index(drop=True)
#log_predict=pandas.Series(scipy.special.inv_boxcox(y_predicted_logtransformed,0)).reset_index(drop=True)
#
#compare=pandas.DataFrame(pandas.concat([actual,ridge_predict,log_predict],axis=1))
#compare.columns=['Actual','Ridge','RFLog']
#
#
#ridge=linear_model.BayesianRidge()
#ridge.fit(x_train,scipy.special.boxcox(y_train,0))
#print(ridge.score(x_train,scipy.special.boxcox(y_train,0)))
#print(ridge.score(x_test,scipy.special.boxcox(y_test,0)))
#
#
#seaborn.regplot(x=temp.bedrooms,y=temp.price)
#xgb=xgboost.XGBRegressor(max_depth=10,eta=)
#
#xgb=xgb.fit(x_train,scipy.special.boxcox(y_train,0))
#print(xgb.score(x_train,scipy.special.boxcox(y_train,0)))
#print(xgb.score(x_test,scipy.special.boxcox(y_test,0)))
#y_predicted_xgb=xgb.predict(x_test)
#print(metrics.median_absolute_error(y_test,scipy.special.inv_boxcox(y_predicted_xgb,0)))
#print(numpy.sqrt(metrics.mean_squared_error(y_test,scipy.special.inv_boxcox(y_predicted_xgb,0))))
#
#from xgboost  import plot_tree,plot_importance,to_graphviz
#cols=x_train.columns
#to_graphviz=to_graphviz(xgb)
#cols=[x.replace(' ','') for x in cols]
#plot_tree(xgb,feature_names=cols)
#plot_importance(xgb)
#import numpy
#import numpy.core.multiarray
#import shap
#numpy.version.version
#
#shap.initjs()
#
#explainer = shap.TreeExplainer(xgb)
#import site; site.getsitepackages()
#shap_values = explainer.shap_values(X)
#
## visualize the first prediction's explanation
#shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:])
#plt.savefig('C:\\Users\\gudea\\Desktop\\Python\\imp_weight.jpg')
#d=plt.rcParams['figure.figsize']
#d[0]=30
#d[1]=30
#plot_importance(xgb,importance_type='cover')
#plt.savefig('C:\\Users\\gudea\\Desktop\\Python\\imp_cover.jpg')
#import graphviz
#from sklearn.tree import export_graphviz
#export_graphviz(xgb,
#                feature_names=x_train.columns,
#                filled=True,
#                rounded=True)
#
#param_grid = {'xgbregressor__learning_rate': numpy.arange(0.05,0.2,0.05), 
#              'xgbregressor__max_depth': numpy.arange(5,9,1),
#              'xgbregressor__n_estimators': [100, 300],
#              'xgbregressor__min_child_weight ': numpy.arange(1,4,1),
#              'xgbregressor__alpha ': [0.005,0.05]}
#
## Instantiate the grid search model
#grid_search = model_selection.GridSearchCV(estimator = xgb,
#                           param_grid = param_grid, 
#                           cv = 3, n_jobs = -1, verbose = 2, 
#                           scoring = 'neg_mean_squared_error')
#grid_search.fit(x_train, scipy.special.boxcox(y_train,0))
#grid_search.best_params_
#
#y_predicted_xgb=xgb.predict(x_test)
#print(metrics.median_absolute_error(y_test,scipy.special.inv_boxcox(y_predicted_xgb,0)))
#print(numpy.sqrt(metrics.mean_squared_error(y_test,scipy.special.inv_boxcox(y_predicted_xgb,0))))
#print(numpy.sqrt(metrics.mean_squared_error(y_train,scipy.special.inv_boxcox(xgb.predict(x_train),0))))
#xgb.get_params()
#imps_xgb=dict()
#for feature, importance in zip(x_train.columns, xgb.feature_importances_):
#    imps_xgb[feature] = importance #add the name/value pair 
#imps_xgb=list(sorted(imps_xgb.items(),key=operator.itemgetter(1),reverse=True))
#
#x=x_test.reset_index(drop=True)
#actual=y_test.reset_index(drop=True)
#predict=pandas.Series(scipy.special.inv_boxcox(y_predicted_xgb,0)).reset_index(drop=True)
#diff=numpy.abs(actual-predict)
#diffpercent=(diff/actual)*100
#within5percent=diffpercent[diffpercent<=10]
#
#compare=pandas.DataFrame(pandas.concat([actual,predict,diff,x],axis=1))
#ind=['Actual','Predicted','Difference']
#ind.extend(x.columns)
#compare.columns=ind
#compare.to_csv('analyze_predictions.csv',index=False)
#
#
#
#
#'''BEDROOM BASED MODELS'''
#
#X_1bhk=london_data_bkp[london_data_bkp.cluster==0].drop(['price','cluster'],axis=1)
#X_1bhk=X_1bhk.drop(cols,axis=1)
#Y_1bhk=london_data_bkp[london_data_bkp.cluster==0]['price']
#x_train_1bhk, x_test_1bhk, y_train_1bhk, y_test_1bhk = model_selection.train_test_split(X_1bhk, Y_1bhk,random_state = 25,train_size=0.7)
#xgb_1bhk=xgboost.XGBRegressor(max_depth=8,n_estimators=100,reg_lambda=,reg_alpha=0.05)
#xgb_1bhk.fit(x_train_1bhk,scipy.special.boxcox(y_train_1bhk,0))
#print(xgb_1bhk.score(x_train_1bhk,scipy.special.boxcox(y_train_1bhk,0)))
#print(xgb_1bhk.score(x_test_1bhk,scipy.special.boxcox(y_test_1bhk,0)))
#print(numpy.sqrt(metrics.mean_squared_error(y_test_1bhk,scipy.special.inv_boxcox(xgb_1bhk.predict(x_test_1bhk),0))))
#print(metrics.median_absolute_error(y_test_1bhk,scipy.special.inv_boxcox(xgb_1bhk.predict(x_test_1bhk),0)))
##rmse 25.9
#
#,reg_lambda =30,reg_alpha=0.6
#

'''clusters'''
london_data_imputed.to_csv('C:\\Users\\gudea\\Desktop\\Python\\airbnb\\london_clean_4.csv',index=True)





