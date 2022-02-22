import pandas as pd
import fastai as fastai
import numpy as np
from fastai import tabular
from fastai.imports import *
from fastai import *
from fastai.tabular import *
from fastai.tabular.transform import Categorify, FillMissing, Normalize, add_datepart
from fastai.tabular.data import TabularDataBunch, TabularList
from sklearn import metrics
import sklearn as sk
from fastai.metrics import accuracy, rmse
from fastai.tabular.learner import tabular_learner
procs = [FillMissing,Categorify,Normalize]
procs1 = [FillMissing,Categorify]
inpDf = pd.read_csv('/data2/home/prasannaiyer/Projects/TT_Fastai/Dataset/OTD_2019_PU.csv')
inpDf = inpDf[(inpDf['Days Late']<20)&((inpDf['Location Type']=='Plant')\
    |(inpDf['Location Type']=='Port')|(inpDf['Location Type']=='Keen'))]

### CREATION OF NEW FIELDS ###
inpDf['Delivery_Date'] = pd.to_datetime(inpDf['Shipment End Date'])
inpDf['Pickup_Date'] = pd.to_datetime(inpDf['First shipped date (first YT26 action) Date'])
inpDf['OutputTransitTime'] = (inpDf['Delivery_Date']-inpDf['Pickup_Date']).dt.days
inpDf.rename(columns = {'Shipment Secure Resources Upd Dt':'Tender_Date'},inplace = True)
inpDf['Tender_Date'] = pd.to_datetime(inpDf['Tender_Date'])
inpDf['OD'] = [1 if OD > 0 else 0 for OD in inpDf['Shipment OD Day']]

### NORMALISE OutputTransitTime ###
inpDf = inpDf[(inpDf['OutputTransitTime'] <16) & (inpDf['OutputTransitTime'] > 0)]
inpDf = inpDf[inpDf['Shipment Loaded Distance']>200]
### DROP NULL VALUES ###
inpDf.dropna(subset=['OutputTransitTime'],inplace=True)
#inpDf['OutputTransitTime'] = inpDf['OutputTransitTime'].astype('int32')

### DROP COLUMNS ###
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
inpDf = inpDf.sort_values(by=['Tender_Date'])
inpDf = inpDf.drop(['Tender_Date'],axis=1)

### CATEGORICAL COLUMNS ###
cat_columns = []
for c in inpDf.columns:
  if inpDf[c].dtype in ['object']:
    cat_columns.append(c) 
    #print(c,' +++ ',inpDf[c].dtype)

### CONTINUOUS OR NUMERICAL COLUMNS ###
cont_columns = []
for c in inpDf.columns:
  if inpDf[c].dtype not in ['object']:
    cont_columns.append(c) 

### CREATE EMBEDDINGS ###
emb_sizes={}
for c in cat_columns:
    if inpDf[c].dtypes == 'O':
        cat_count = int(inpDf[c].nunique()/2)
        if cat_count>1:
            emb_sizes.update({c:cat_count})

### BOOLEAN COLUMNS ###
for bcol in cont_columns:
    #print(bcol,'--',inpDf[bcol].dtypes)
    if inpDf[bcol].dtypes == 'bool':
        inpDf[bcol] = inpDf[bcol].astype('int32')

### TRAIN/TEST DATA SPLIT ###
data_count = inpDf.shape[0]
data_train_count = int(data_count*0.8)
data_val_count = data_count - data_train_count
train_data = inpDf[:data_train_count]
test_data = inpDf[data_train_count:]
val_idx = range(data_train_count,data_count-1)
val_idx1 = range(data_train_count,data_count)

dep_var = 'OutputTransitTime'
cont_columns.remove(dep_var)
path = ''
tab_databunch = TabularDataBunch.from_df(path,inpDf,dep_var,valid_idx=val_idx,\
    cat_names=cat_columns,procs = procs,cont_names=cont_columns) 


### NO NORMALIZATION ###
tab_databunch1 = TabularDataBunch.from_df(path,inpDf,dep_var,valid_idx=val_idx1,\
    cat_names=cat_columns,procs = procs1,cont_names=cont_columns) 

#check = TabularList.from_df(df, path=path, cat_names=cat_names, \
 #   cont_names=cont_names, procs=procs).split_by_idx(valid_idx))

learner1 = tabular_learner(tab_databunch1,layers=[1000,500],\
     emb_szs=emb_sizes,metrics=rmse)

### Train 20 epochs
learner1.fit_one_cycle(20, 1e-2)

### Prediction details
pred1,y1,loss1 = learner1.get_preds(with_loss=True)
x = rmse(pred1,y1)


