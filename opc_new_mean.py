#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import all required libraries
import pandas as pd # for data manipulation purpose
import numpy as np # for numerical calculation
#%matplotlib inline
import matplotlib.pyplot as plt # for data visualization
import seaborn as sns # for advance data visualization
from sklearn.preprocessing import LabelEncoder # for converting categorical data into numeric 
from sqlalchemy import create_engine # to connect the sql database 
import datetime # to work on date time
from feature_engine.outliers import Winsorizer # To remove the outliers
import sweetviz # Auto EDA
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing # holts winters Exponential Smoothing
import statsmodels.graphics.tsaplots as tsa_plots # for plot the ACF and PACF plots
from statsmodels.tsa.arima.model import ARIMA # To build the ARIMA models
import pmdarima as pm # for Auto ARIMA
import statsmodels.formula.api as smf

import warnings
warnings.filterwarnings('ignore') 


# In[6]:


# load thd data set
cement = pd.read_csv(r"C:\Users\DELL\OneDrive\Desktop\cement_new_llllllllll.csv") # load the dataset 


# In[7]:


# Data base connection
user = "root" # user name 
pwd = '9963973155' # password
db = 'cement' # data base
con = create_engine(f'mysql+pymysql://{user}:{pwd}@localhost/{db}') # conntion for database 


# In[8]:


# load the data set into data base
cement.to_sql('cement', con = con, if_exists = 'replace', chunksize = 1000, index = False) # pull the data into the database
sql = 'select * from cement' # sql query
cement = pd.read_sql_query(sql, con = con) # load the data from database
cement.head()


# In[9]:


# rename the columns 
cement = cement.iloc[:,[ 0, 1, 2, 3, 5, 6, 7, 8, 9]] # drop the column ( Quantity in Ton)
cement.columns = ['Date', 'CementType', 'Price_Before_VAT', 'Quantity_In_Quintal', 'Transportation', 'Sales', 'Tax', 'Advat_cost','Fuel_cost'] # rename the columns
cement.head()


# ## Data Preprocessing and EDA ####

# In[10]:



cement.info() # shows the summary and details of the data 


# In[11]:


cement.isna().sum() # check for null values and it's total


# In[12]:


cement.dropna(inplace = True) # drop the null values


# In[13]:


cement.duplicated().sum() # check the duplicate values and their sum


# In[14]:


cement.drop_duplicates(inplace = True) # drop the duplicated values


# In[21]:


labelencode = LabelEncoder() # intialize the label Encoder

# apply label encodeing on cementType column ppc_bags = 2, opc_bags = 0, opc_bulk = 1
cement['CementType'] = labelencode.fit_transform(cement['CementType'])  
opc_bags =  cement.query('CementType == 1') # filtering with query method separate opc_bags 
opc_bags.reset_index(drop = True) # reset the index


# In[16]:


opc_bags.describe() # describe the stastical values


# In[23]:


## data visualization
#  Line plot for each slaes and Quantity_In_Quintal
for column in ['Quantity_In_Quintal','Sales']:
    plt.figure(figsize = (15, 8))
    plt.plot(opc_bags[column])
    plt.title(f'Line_plot of {column}')
    plt.show()


# In[26]:


## Histogram
a = [ 'Price_Before_VAT', 'Quantity_In_Quintal', 'Transportation', 'Sales', 'Tax', 'Advat_cost','Fuel_cost']
for column in a:
    plt.figure(figsize = (8,6))
    plt.hist(opc_bags[column])
    plt.title(f'Histogram of {column}')
    plt.show()


# In[31]:


# bar plot for each column
for column in a:
    plt.figure(figsize =(15,8))
    plt.bar(x = np.arange(0, 655, 1), height = opc_bags[column])
    plt.title(f"bar_plot of {column} ")
    plt.show()


# In[32]:


## box plot
for column in a:
    plt.figure(figsize = (15,8))
    plt.boxplot(opc_bags[column])
    plt.title(f'Box plot of {column}')
    plt.show()


# In[33]:


opc_bags.plot(kind = 'box', subplots = True, sharey = False, figsize = (15,10))
plt.subplots_adjust(wspace = 0.75)
plt.show()


# In[34]:


from feature_engine.outliers import Winsorizer # intialize the winsorizer
winsor = Winsorizer(capping_method = 'iqr', tail = 'both', fold = 1.5, 
                    variables =[ 'Price_Before_VAT', 'Quantity_In_Quintal',
                           'Transportation', 'Sales', 'Tax', 'Advat_cost', 'Fuel_cost'])


# In[36]:


# apply winsorization on opc_bags data
opc_bags_w = winsor.fit_transform(opc_bags [['Date',  'Price_Before_VAT', 'Quantity_In_Quintal',
       'Transportation', 'Sales', 'Tax', 'Advat_cost', 'Fuel_cost']])


# In[37]:


opc_bags_w.plot(kind = 'box', subplots = True, sharey = False, figsize = (15,10))
plt.subplots_adjust(wspace = 0.75)
plt.show()


# In[38]:


# pair plot
sns.pairplot(opc_bags_w)
plt.show()


# In[39]:


# corelation 
corr_matrix = opc_bags_w.corr()


# In[40]:


# heatmap
sns.heatmap(corr_matrix, cmap = 'coolwarm', annot = True)


# In[42]:


################################### AUTO EDA ################################################

import sweetviz
my_report = sweetviz.analyze([opc_bags,'opc_bags_w'])
my_report.show_html('report1.html')
my_report.show_notebook()


# In[43]:


opc_bags_w['Date'] = pd.to_datetime(opc_bags_w['Date']) # Convert the 'date' column to a datetime data type
opc_bags_m = opc_bags_w.resample('M',  on = 'Date').mean() # using resample daily data into monthly based on average

opc_bags1 = opc_bags_m.iloc[:,[1,3]]


# In[47]:
opc_bags_w.info()

plt.plot(opc_bags1['Sales'])  # line plot of sales


# In[50]:





# In[49]:


# Creating a function to calculate the MAPE value for test data 
def MAPE(pred, org):
    temp = np.abs((pred-org)/org)*100
    return np.mean(temp)


# # Model Building
# # Data Driven models
# 

# In[62]:


# spliting the data into training and testing
train = opc_bags1.head(36)
test = opc_bags1.tail(12)


# In[68]:


# Simple Exponential model
ses_model = SimpleExpSmoothing(train['Sales']).fit() # trianing the model on train data 
pred_ses = ses_model.predict(start = test.index[0], end = test.index[-1]) # predict on test data 
MAPE(pred_ses, test.Sales) # check the error percent between testing and predicted values on test
print('simple_exp_sm_MAPE:',MAPE(pred_ses, test.Sales) )
rmse_ses = np.sqrt((np.mean(np.array(test['Sales'])- np.array(pred_ses))**2))
print('rmse_simple_exp_sm:',rmse_ses)


# In[72]:


# Holts method 
hw_model = Holt(train['Sales']).fit()
pred_hw = hw_model.predict(start = test.index[0], end = test.index[-1])
MAPE(pred_hw, test.Sales)
print('holt_MAPE:',MAPE(pred_hw, test.Sales))
rmse_Hol = np.sqrt((np.mean(np.array(test['Sales'])- np.array(pred_hw))**2))
print('rmse_Holt:',rmse_Hol)


# In[76]:


## Holts winter exponential smoothing with additive seasonality and additive trend
hwe_model_add_add = ExponentialSmoothing(train['Sales'], seasonal = "add", trend = "add", seasonal_periods = 12).fit()
pred_hwe_add_add = hwe_model_add_add.predict(start = test.index[0], end = test.index[-1])
MAPE(pred_hwe_add_add, test.Sales)
print('hwes_add_add_MAPE:',MAPE(pred_hwe_add_add, test.Sales))
rmse_Hol_w = np.sqrt((np.mean(np.array(test['Sales'])- np.array(pred_hwe_add_add))**2))
print('rmse_hwes_add_add:',rmse_Hol_w)


# In[150]:


# Holts winter exponential smoothing with multiplicative seasonality and additive trend
hwe_model_mul_add = ExponentialSmoothing(train['Sales'], seasonal = "mul", trend = "add", seasonal_periods = 12).fit()
pred_hwe_mul_add = hwe_model_mul_add.predict(start = test.index[0], end = test.index[-1])
print('hwe_mul_add_MAPE:', MAPE(pred_hwe_mul_add, test.Sales ))
rmse_Hol_wma = np.sqrt((np.mean(np.array(test['Sales'])- np.array(pred_hwe_mul_add))**2))
print('rmse_hwe_mul_add:', rmse_Hol_wma)


# In[88]:


#To Build  AR Model Check the lags using ACF plot
tsa_plots.plot_acf(opc_bags1.Sales)
tsa_plots.plot_pacf(opc_bags1.Sales)


# In[91]:


AR_model = ARIMA(train.Sales, order = (1, 0, 0)) # Initialize the model
res = AR_model.fit() # fit the model
print(res.summary())
# Forecast for next 12 months
pred_ar = res.predict(start = len(train), end = len(train) + 11)
rmse_AR = np.sqrt((np.mean(np.array(test['Sales'])- np.array(pred_ar))**2))
print('rmse_AR:', rmse_AR)
print('AR_MAPE:', MAPE(pred_ar, test.Sales ))


# In[93]:


# Build the MA model
MA_model = ARIMA(train.Sales, order = (0, 0, 1)) # Initialize the model
res = MA_model.fit() # fit the model
print(res.summary())
# Forecast for next 12 months
pred_MA = res.predict(start = len(train), end = len(train) + 11)
rmse_MA = np.sqrt((np.mean(np.array(test['Sales'])- np.array(pred_MA))**2))
print('rmse_MA:', rmse_MA)
print('MA_MAPE:', MAPE(pred_MA, test.Sales ))


# In[94]:


# Build the ARMA model
ARMA_model = ARIMA(train.Sales, order = (1, 0, 1)) # Initialize the model
res = AR_model.fit() # fit the model
print(res.summary())
# Forecast for next 12 months
pred_arma = res.predict(start = len(train), end = len(train) + 11)
rmse_ARMA = np.sqrt((np.mean(np.array(test['Sales'])- np.array(pred_arma))**2))
print('rmse_ARMA:', rmse_ARMA)
print('ARMA_MAPE:', MAPE(pred_arma, test.Sales ))


# In[96]:


# Build the ARIMA model
AR_model = ARIMA(train.Sales, order = (1, 1, 1)) # Initialize the model
res = AR_model.fit() # fit the model
print(res.summary())
# Forecast for next 12 months
pred_arima = res.predict(start = len(train), end = len(train) + 11)
rmse_ARIMA = np.sqrt((np.mean(np.array(test['Sales'])- np.array(pred_arima))**2))
print('rmse_ARIMA:', rmse_ARIMA)
print('ARIMA_MAPE:', MAPE(pred_arima, test.Sales ))


# In[97]:


# # Auto-ARIMA - Automatically discover the optimal order for an ARIMA model.
# # pip install pmdarima --user
ar_model = pm.auto_arima(train.Sales, start_p=0, start_q=0,
                      max_p=12, max_q=12, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal= True,   #  Seasonality
                        trace=True,
                      error_action='warn', stepwise=True)


# In[98]:


# Build the Random Walk model
Rw_model = ARIMA(train.Sales, order = (0, 1, 0)) # Initialize the model
res = Rw_model.fit() # fit the model
print(res.summary())
# Forecast for next 12 months
pred_Rw = res.predict(start = len(train), end = len(train) + 11)
rmse_RW = np.sqrt((np.mean(np.array(test['Sales'])- np.array(pred_Rw))**2))
print('rmse_RW:', rmse_RW)
print('ARIMA_MAPE:', MAPE(pred_Rw, test.Sales ))


# In[109]:



# model Based 

month =['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
dates = pd.date_range(start = '2019-11-01', freq = 'MS', periods = len(opc_bags1))
opc_bags1['Months'] = dates.month


# In[111]:


import calendar
opc_bags1['Months'] = opc_bags1['Months'].apply(lambda x: calendar.month_abbr[x])
opc_bags1 = opc_bags1[['Months','Sales'  ]]


# In[113]:




opc_bags1['t'] = np.arange(1,39)

opc_bags1['t_square'] = opc_bags1['t'] * opc_bags1['t']

opc_bags1['log_Sales'] = np.log(opc_bags1['Sales'])


# In[115]:



dummy = pd.DataFrame(pd.get_dummies(opc_bags1['Months']))
opc_bags1 = pd.concat((opc_bags1, dummy), axis = 1)
 


# In[117]:



Train = opc_bags1.head(26)
Test = opc_bags1.tail(12)


# In[123]:



# linear model
linear = smf.ols('Sales ~ t', data = Train).fit()
predic = pd.Series(linear.predict(pd.DataFrame(Test['t'])))
rmse_lin = np.sqrt((np.mean(np.array(test['Sales'])- np.array(predic))**2))
print('rmse_lin:',rmse_lin)
print('MAPE_lin:',MAPE(predic, Test['Sales']))


# In[125]:


# qudratic model
quad = smf.ols('Sales ~ t + t_square', data = Train).fit()
pred_quad = pd.Series(quad.predict(pd.DataFrame(Test[['t', 't_square']])))
rmse_quad = np.sqrt((np.mean(np.array(Test['Sales'])- np.array(pred_quad))**2))
print('rmse_quad:',rmse_quad)
print('MAPE_quad:',MAPE(pred_quad, Test['Sales']))


# In[127]:


# Exponential model
expo = smf.ols('log_Sales ~ t', data = Train).fit()
predic_exp = pd.Series(np.exp(expo.predict(pd.DataFrame(Test['t']))))
rmse_exp = np.sqrt((np.mean(np.array(Test['Sales'])- np.array(predic_exp))**2))
print('rmse_exp:', rmse_exp)
print('MAPE_exp:',MAPE(predic_exp, Test['Sales']))


# In[ ]:





# In[158]:


# out of the all models Holts Winter Exponential smoothings has less so 100% build on this model and forecast for next 12 months
# Holts winter exponential smoothing with multiplicative seasonality and additive trend
hwe_model_mul_add_final = ExponentialSmoothing(opc_bags1['Sales'], seasonal = "multiplicative", trend =  "additive", seasonal_periods = 12).fit()
pred_hwe_mul_add1 = hwe_model_mul_add_final.predict(start = len(opc_bags1), end = len(opc_bags1)+11)
 


# In[146]:


#pred_hwe_mul_add
len(opc_bags1)
pred_hwe_mul_add1


# In[160]:


plt.plot(pred_hwe_mul_add1)
plt.plot(opc_bags1['Sales'])
plt.show()


# In[162]:


# Save the best model
hwe_model_mul_add_final.save('hwe_model_mul_add_fina.pickle') # save the model
import pickle
pickle.dump(hwe_model_mul_add_final, open('hwe_model_mul_add_final.pickle', 'wb'))
# In[ ]:




