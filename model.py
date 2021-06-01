import pandas as pd
from data import conn, sql

data = pd.io.sql.read_sql(sql, conn)

#remove metadata
df=data.drop(['VISIT_TYPE','CSN_ID','PAT_ID','ADDRESS_PAT','CITY','STATE','ZIP_PAT','CENTER','EPIC_ID','GL','APPT_STATUS_NAME'], axis=1)

#remove Nulls
df=df.dropna()

#remove spaces
#df_clean.str.strip(['VISIT_TYPE','CSN_ID','PAT_ID','ADDRESS_PAT','CITY','STATE','ZIP_PAT','CENTER','EPIC_ID','GL','APPT_STATUS_NAME'])


############################################### Part 3: Descriptive Statistics############################################


df.head()
df.describe()
df.std()

print(list(df.columns))
    



############################################ Part 4: Log Regression (StatsModel)##########################################

import sckitlearn as sk

import statsmodels.api as sm
##import pylab as pl
##import numpy as np


#Split dataset into dependant and independant variables\

train_cols=['SEX','RACE','LANGUAGE','EMPLY_STATUS','MARITAL_STATUS','AGE_AT_ENCOUNTER','ENCOUNTER_PAYOR','APPT_MADE_DATETIME','APPT_MADE_DATE','APPT_TIME','APPT_WEEK','APPT_MONTH','APPT_DAY','APPT_HOUR','APPT_TIME_AMPM','ENCOUNTER_DATE','DAYS_TO_APPT','NS_HX']


X=df[train_cols]
Y=df.APPT_STATUS

logit = sm.Logit(Y,X.astype(float))
result=logit.fit()

print(result.summary())


#Export data frame to CSV
#df.to_csv (r'C:\Users\ZhaoF\Documents\DS_Proj_1_PredNoShows\df_clean.csv', index = None, header=True)
