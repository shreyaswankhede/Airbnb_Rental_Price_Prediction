# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 10:17:04 2019

@Team members: vrushank,shehzada,sameer,shreyas,sagar
"""

import pandas
from sklearn import preprocessing,model_selection,metrics,ensemble,linear_model
import numpy
import collections
import operator
import matplotlib.pyplot as plt
import scipy
import seaborn
import xgboost
import scipy.stats
from pdpbox import pdp,info_plots
from sklearn.preprocessing import scale
seaborn.set(rc={'figure.figsize':(11.7,8.27)})

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
    merged_scrape=pandas.read_csv('C:\\Users\\Shreyas\\combinedscrape_180119.csv')
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


def plotFeatureImportances(model,columns):
    imps_bin1=dict()
    for feature, importance in zip(columns, model.feature_importances_):
        imps_bin1[feature] = importance #add the name/value pair 
    imps_bin1=list(sorted(imps_bin1.items(),key=operator.itemgetter(1),reverse=True))
    top_15_imp=imps_bin1[:15]
    labels=[]
    vals=[]
    for f in top_15_imp:
        labels.append(f[0])
        vals.append(f[1]*100)
    
    seaborn.barplot(y=labels,x=vals)
    
#One time use function
def scrapeUpdatedPricesFromAirbnb():
    import pandas
    import numpy
    from bs4 import BeautifulSoup
    from selenium import webdriver
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
    from selenium.webdriver.chrome.options import Options 
    import collections
    '''read processed data'''
    london_data=pandas.read_csv('C:\\Users\\Shreyas\\processed_data.csv')
    
    '''scrape'''
    dollar_rate=70.38
    chrome_options = Options()  
    chrome_options.add_argument("--headless")  
    new_prices=pandas.DataFrame(data=None,columns=['listing_url','new_price'])
    capa = DesiredCapabilities.CHROME
    capa["pageLoadStrategy"] = "none"
                
    index=0
    '''change below condition according to what you want to scrape'''    
    subset=london_data.loc[(london_data.price<40) & (london_data.bedrooms==5),: ]
    
    for url in subset.listing_url:          
        try:            
            driver = webdriver.Chrome('C:\\Users\\Shreyas\\chromedriver.exe',desired_capabilities=capa,chrome_options=chrome_options)
            driver.get(url)                
            wait = WebDriverWait(driver, 10)#10 seconds wait.increase if your net is slow    
            wait.until(EC.presence_of_element_located((By.CLASS_NAME, '_doc79r')))
            
            driver.execute_script("window.stop();")        
            source=driver.page_source
            soup=BeautifulSoup(source,'html.parser')
            spandiv=soup.find_all('span',attrs={'class':'_doc79r'}) 
            if(spandiv!=None and len(spandiv)>0):
                price=spandiv[0].text
                price=price.replace(',','').replace('₹','')
                price=int(price)
                price=numpy.ceil(price/dollar_rate)
                new_prices.loc[index,:]=(url,price)
                index+=1                    
                driver.close()
            else:
                driver.close()
        except:
            print('Timeout exception:',url)
            new_prices.loc[index,:]=(url,'NaN')
            index+=1
            driver.close()
        print(index)
    
    '''write file.add ur name to the file'''
    new_prices.to_csv('scrapped_file.csv',index=False)
    

def generateInsight(model,features,data):
    pdp_airbnb = pdp.pdp_isolate(model=model,
                               dataset=data,
                               model_features=data.columns,
                               feature=features)
    fig, axes = pdp.pdp_plot(pdp_isolate_out=pdp_airbnb,
                             feature_name=features,
                             plot_pts_dist=True, 
                             )



'''LOGIC STARTS''' 
#load the data
london_data=pandas.DataFrame(pandas.read_csv('C:\\Users\\Shreyas\\london_listings.csv',low_memory=False,na_values=['',' ',numpy.NAN,numpy.NaN,'NA','N/A']))
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



'''OUTLIER TREATMENT'''
#extreme outliers removal based on the number of bedrooms
london_data=handleAbove99tileByCategory(london_data,'bedrooms','price')
#handle outliers in case of number of bathrooms for other properties
london_data=handleAbove99tileByCategory(london_data,'bedrooms','bathrooms',True)

'''VISUALIZATION'''
seaborn.pointplot(x=london_data.bedrooms,y=london_data.price)
seaborn.barplot(y=london_data.london_borough,x=london_data.price)
seaborn.jointplot(x=london_data.bedrooms, y=london_data.price, kind="hex", color="#4CB391")
seaborn.barplot(y=london_data.property_type,x=london_data.price)
seaborn.boxplot(x=london_data.bedrooms)
seaborn.FacetGrid(london_data[['latitude','longitude','london_borough']],hue='london_borough').map(plt.scatter,'latitude','longitude').add_legend()
seaborn.regplot(x=london_data.bedrooms,y=london_data.price)
seaborn.distplot(london_data.price)
seaborn.regplot(x=london_data.bedrooms,y=london_data.bathrooms)


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
bins=[0,100,2001]
names=[1,2]
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



'''CLASSIFICATION OF THE PRICE BINS'''
#Most of the time people have an idea about the price range in which their rental will fail. 
#For users who have no idea about the price range, we can first classify in which price bucket their rental can fall in
#and then do the bucket specific regression

#RandomForest Classifier to predict the price bins
randomForestClassifier=ensemble.RandomForestClassifier(n_estimators=200,max_features='auto',max_depth=15,min_samples_leaf=7,random_state=25,class_weight='balanced')
randomForestClassifier.fit(scale(x_train),y_train['price_bins'])
print(randomForestClassifier.score(scale(x_train),y_train['price_bins']))
print(randomForestClassifier.score(scale(x_test),y_test['price_bins']))
y_pred=randomForestClassifier.predict(scale(x_test))
report=metrics.classification_report(y_test['price_bins'],y_pred)
print(report)

#Logistic Regression
logistic=linear_model.LogisticRegression(random_state=23,class_weight='balanced')
logistic.fit(scale(x_train),y_train['price_bins'])
print(logistic.score(scale(x_train),y_train['price_bins']))
print(logistic.score(scale(x_test),y_test['price_bins']))
y_pred=logistic.predict(scale(x_test))
report=metrics.classification_report(y_test['price_bins'],y_pred)
print(report)


#Vote Classifier
voteClassifier=ensemble.VotingClassifier(estimators=[('lr', logistic), ('rf', randomForestClassifier)],voting='soft')
voteClassifier.fit(scale(x_train),y_train['price_bins'])
print(voteClassifier.score(scale(x_train),y_train['price_bins']))
print(voteClassifier.score(scale(x_test),y_test['price_bins']))
y_pred=voteClassifier.predict(scale(x_test))
report=metrics.classification_report(y_test['price_bins'],y_pred)
print(report)


'''REGRESSION'''

#A regression model is built for each of the price bins. So total 2 regression models will be built.
#Based on the output of the classification model, we will call the appropriate regression model

#########Model for Price Bin 1################
y_train_bin1=y_train.loc[y_train.price_bins==1,'price']
y_test_bin1=y_test.loc[y_test.price_bins==1,'price']
x_train_bin1=x_train.loc[y_train_bin1.index,:]
x_test_bin1=x_test.loc[y_test_bin1.index,:]

y_train_bin1=y_train_bin1.reset_index(drop=True)
y_test_bin1=y_test_bin1.reset_index(drop=True)
x_train_bin1=x_train_bin1.reset_index(drop=True)
x_test_bin1=x_test_bin1.reset_index(drop=True)



xgb_bin1=xgboost.sklearn.XGBRegressor(max_depth=9,n_estimators=200,min_child_weight=5,reg_alpha=0.3,random_state=23)
xgb_bin1.fit(x_train_bin1,scipy.special.boxcox(y_train_bin1,0))
y_pred_bin1_train=scipy.special.inv_boxcox(xgb_bin1.predict(x_train_bin1),0)
y_pred_bin1_test=scipy.special.inv_boxcox(xgb_bin1.predict(x_test_bin1),0)
#error on train
print('MEDIAN ABSOLUTE ERROR TRAIN: ',metrics.median_absolute_error(y_train_bin1,y_pred_bin1_train))
print('RMSE TRAIN: ',numpy.sqrt(metrics.mean_squared_error(y_train_bin1,y_pred_bin1_train)))
#error on test
print('MEDIAN ABSOLUTE ERROR TEST: ',metrics.median_absolute_error(y_test_bin1,y_pred_bin1_test))
print('RMSE TEST: ',numpy.sqrt(metrics.mean_squared_error(y_test_bin1,y_pred_bin1_test)))


#Understanding the model
generateInsight(xgb_bin1,'latitude',x_train_bin1)
generateInsight(xgb_bin1,'longitude',x_train_bin1)
generateInsight(xgb_bin1,'accommodates',x_train_bin1)
generateInsight(xgb_bin1,'extra_people',x_train_bin1)
generateInsight(xgb_bin1,'minimum_nights',x_train_bin1)
generateInsight(xgb_bin1,'guests_included',x_train_bin1)
generateInsight(xgb_bin1,[col for col in x_train_bin1 if col.startswith('room_type')],x_train_bin1)
generateInsight(xgb_bin1,[col for col in x_train_bin1 if col.startswith('property_type')],x_train_bin1)

#interaction plot
feats = ['latitude', 'longitude']
p = pdp.pdp_interact(xgb_bin1, x_train_bin1, x_train_bin1.columns,feats)
pdp.pdp_interact_plot(p, feats)#the model is correctly able to find to classify the centre to london close to Buckingham palace as most expensive

fig, axes, summary_df = info_plots.actual_plot_interact(
    model=xgb_bin1, X=x_train_bin1, 
    features=['accommodates', [col for col in x_train_bin1 if col.startswith('property_type')]], 
    feature_names=['accommodates', 'Property Type'],
    
)

#Plot Feature importances
plotFeatureImportances(xgb_bin1,x_train_bin1.columns)



##############Model for Price Bin 2#####################
y_train_bin2=y_train.loc[(y_train.price_bins==2),'price']
y_test_bin2=y_test.loc[y_test.price_bins==2,'price']
x_train_bin2=x_train.loc[y_train_bin2.index,:]
x_test_bin2=x_test.loc[y_test_bin2.index,:]

y_train_bin2=y_train_bin2.reset_index(drop=True)
y_test_bin2=y_test_bin2.reset_index(drop=True)
x_train_bin2=x_train_bin2.reset_index(drop=True)
x_test_bin2=x_test_bin2.reset_index(drop=True)




xgb_bin2=xgboost.XGBRegressor(max_depth=7,n_estimators=200,min_child_weight=6,reg_alpha=0.9,random_state=23,learning_rate=0.1)
xgb_bin2.fit(x_train_bin2,scipy.special.boxcox(y_train_bin2,0))
y_pred_bin2_train=scipy.special.inv_boxcox(xgb_bin2.predict(x_train_bin2),0)
y_pred_bin2_test=scipy.special.inv_boxcox(xgb_bin2.predict(x_test_bin2),0)
#error on train
print('MEDIAN ABSOLUTE ERROR TRAIN: ',metrics.median_absolute_error(y_train_bin2,y_pred_bin2_train))
print('RMSE TRAIN: ',numpy.sqrt(metrics.mean_squared_error(y_train_bin2,y_pred_bin2_train)))
#error on test
print('MEDIAN ABSOLUTE ERROR TEST: ',metrics.median_absolute_error(y_test_bin2,y_pred_bin2_test))
print('RMSE TEST: ',numpy.sqrt(metrics.mean_squared_error(y_test_bin2,y_pred_bin2_test)))


generateInsight(xgb_bin2,'accommodates',x_train_bin2)
generateInsight(xgb_bin2,'extra_people',x_train_bin2)
generateInsight(xgb_bin2,'minimum_nights',x_train_bin2)
generateInsight(xgb_bin2,'guests_included',x_train_bin2)
generateInsight(xgb_bin2,[col for col in x_train_bin2 if col.startswith('room_type')],x_train_bin2)
generateInsight(xgb_bin2,[col for col in x_train_bin2 if col.startswith('property_type')],x_train_bin2)

#interaction plot
feats = ['latitude', 'longitude']
p = pdp.pdp_interact(xgb_bin2, x_train_bin2, x_train_bin2.columns,feats)
pdp.pdp_interact_plot(p, feats)#the model is correctly able to find to classify the centre to london close to Buckingham palace as most expensive

fig, axes, summary_df = info_plots.actual_plot_interact(
    model=xgb_bin2, X=x_train_bin2, 
    features=['accommodates', [col for col in x_train_bin2 if col.startswith('property_type')]], 
    feature_names=['accommodates', 'Property Type'],
    
)



plotFeatureImportances(xgb_bin2,x_train_bin2.columns)






"""END"""


























