import pandas as pd
import fastai as fastai
import numpy as np
import autogluon as ag
from autogluon import TabularPrediction as task

from fastai.tabular.transform import Categorify, FillMissing, Normalize, add_datepart
from sklearn import metrics
import math
inpDf = pd.read_csv('/data2/home/prasannaiyer/Projects/TT_AG/Dataset/OTD_2019_PU.csv')
inpDf = inpDf[(inpDf['Days Late']<20)&((inpDf['Location Type']=='Plant')\
    |(inpDf['Location Type']=='Port')|(inpDf['Location Type']=='Keen'))]

### CREATION OF NEW FIELDS ###
inpDf['Delivery_Date'] = pd.to_datetime(inpDf['Shipment End Date'])
inpDf['Pickup_Date'] = pd.to_datetime(inpDf['First shipped date (first YT26 action) Date'])
inpDf['OutputTransitTime'] = (inpDf['Delivery_Date']-inpDf['Pickup_Date']).dt.days
inpDf.rename(columns = {'Shipment Secure Resources Upd Dt':'Tender_Date'},inplace = True)
inpDf['Tender_Date'] = pd.to_datetime(inpDf['Tender_Date'])
inpDf['OD'] = [1 if OD > 0 else 0 for OD in inpDf['Shipment OD Day']]

### REMOVE OUTLIERS ###
inpDf = inpDf[(inpDf['OutputTransitTime'] <16) & (inpDf['OutputTransitTime'] > 0)]
inpDf = inpDf[(inpDf['Shipment Loaded Distance']>200) & (inpDf['Shipment Loaded Distance']<2501)]
### DROP NULL VALUES ###
inpDf.dropna(subset=['OutputTransitTime'],inplace=True)
#inpDf['OutputTransitTime'] = inpDf['OutputTransitTime'].astype('int32')

### DROP COLUMNS NOT NEEDED FOR PREDICTION ###
inpDf.drop(['Shipment GID', 'Delivery Status','Shipment Source Location GID',\
    'Shipment Source Province Code','Shipment Destination Location GID',\
        'Shipment Destination Location Name', 'Shipment Destination City',\
            'Shipment Enroute Status Upd Dt','Vehicle Outbound Delivery tendered and accepted Date',\
                'First shipped date (first YT26 action) Date', 'Shipment End Date',\
                    'Expected Days to Deliver', 'Actual Days', 'Days Late','Year',\
                        'Month','Shipment OD Day','Pickup_Date','Delivery_Date'],\
                            inplace = True,axis=1)

### CREATION OF DATE FIELDS ###
inpDf = add_datepart(inpDf,'Tender_Date',drop=False)

### SORT THE DATAFRAME ###
### Next step is to split the data into training and testing dataset. 
### Since this data involves dates, dataset is sorted by the specific date feature. 
### This ensures that older data is used for training and later data is used in testing. 
### This is important as the model will be used to predict future values of the dependent variable
inpDf = inpDf.sort_values(by=['Tender_Date'])
inpDf = inpDf.drop(['Tender_Date'],axis=1)
data_count = inpDf.shape[0]
data_train_count = int(data_count*0.8)
data_val_count = data_count - data_train_count
train_data = inpDf[:data_train_count]
test_data = inpDf[data_train_count:]


labelColumn = 'OutputTransitTime'

### Autogluon requires directories for saving the output
modelDir = '/home/prasannaiyer/Projects/AGModel1'
modelDir1 = '/home/prasannaiyer/Projects/AGModel2'

### HYPER PARAMETERS FOR VARIOUS ML ALGORITHMS
nn_options = { 'num_epochs': 10}
#hyperparameters = {'NN': nn_options}
problemType = 'regression'
hyperparameters={'NN': {'num_epochs':20},'GBM': {'num_boost_round':1000},'CAT': {'iterations':10000},'RF': {'n_estimators':300},\
    'XT': {'n_estimators':300},'KNN': {},'custom': ['GBM']}

hp1={'GBM': {'num_boost_round':10000},'CAT': {'iterations':10000},'RF': {'n_estimators':500},\
    'XT': {'n_estimators':500},'KNN': {},'custom': ['GBM']}

time_limits = 2*60 

### OPTION 1: TRAIN ON INDIVIDUAL ML ALGORITHMS
predModel = task.fit(train_data = train_data,label = labelColumn,hyperparameters = hp1,\
    output_directory = modelDir,problem_type = problemType)

### OPTION 1: TRAIN AN ENSEMBLE MODEL
predModel1 = task.fit(train_data = train_data,label = labelColumn,hyperparameters = hp1,\
    output_directory = modelDir1,problem_type = problemType,num_bagging_folds=5,stack_ensemble_levels=1)

### PREDICT THE OUTPUT ON TEST DATA
test_data_nolabel = test_data.drop(labels = [labelColumn],axis=1)

predOutput = predModel.predict(test_data_nolabel)
predOutputEnsemble = predModel1.predict(test_data_nolabel)

### Calculate the a
x = math.sqrt(metrics.mean_squared_error(test_data[labelColumn], predOutput))

