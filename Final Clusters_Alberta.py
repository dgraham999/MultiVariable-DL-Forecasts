# Databricks notebook source
import pandas as pd
import numpy as np
import seaborn as sns
#import altair as alt
import matplotlib.pyplot as plt
import snowflake.connector
#import plotly
#import plotly.express as px
#import plotly.graph_objects as go
#from plotly.subplots import make_subplots
from dateutil.relativedelta import relativedelta
import datetime
import sklearn
from sklearn.model_selection import train_test_split
import math
from sklearn.linear_model import LinearRegression
import gc
import random 
from pyspark.sql.functions import *

pd.set_option("display.max_rows", None)
import pyspark.pandas as ps
import scipy.stats as stats
from bioinfokit.analys import stat
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Import module for k-protoype cluster
from kmodes.kprototypes import KPrototypes

# Import module for data visualization
from plotnine import *
import plotnine

# COMMAND ----------

# MAGIC %run ./supportingfunctions/Supportingfunctions

# COMMAND ----------

## Table Name
CPO_POS_STORE_GROUPING_CAN = "CPO_POS_STORE_GROUPING_CAN"
FSA_LAT_LONG = "LOAD_RGM_CPO_FSA_LAT_LONG"

# Read in library  
POS_Data = readPySparkdf_DEV(CPO_POS_STORE_GROUPING_CAN).toPandas()
# FSA_LAT_LONG = readPySparkdf_DEV(FSA_LAT_LONG).toPandas()
# FSA_LAT_LONG['FSA'] = FSA_LAT_LONG['FSA'].str[1:]

# COMMAND ----------

## Removing the special character from the column names
POS_Data.columns = POS_Data.columns.str.replace('["]', '')

## Keeping only few columns that are required
POS_Data = POS_Data[['PROVINCE_OF_TAX','FSA','CYCLE_WEEK_END_DATE__YYYY_MM_DD','TOTAL_STICK_SOLD','PRICE_PER_STICK','Total Population']]

## Aggregating the Stick/PRICE as the BANNER and Segment have been removed
x = POS_Data.groupby(['PROVINCE_OF_TAX','FSA','CYCLE_WEEK_END_DATE__YYYY_MM_DD'],as_index = False)['TOTAL_STICK_SOLD'].sum()
y = POS_Data.groupby(['PROVINCE_OF_TAX','FSA','CYCLE_WEEK_END_DATE__YYYY_MM_DD'],as_index = False)['PRICE_PER_STICK','Total Population'].mean()
POS_Data = x.merge(y, on = ['PROVINCE_OF_TAX','FSA','CYCLE_WEEK_END_DATE__YYYY_MM_DD'], how = 'left')

## Sorting the data by Date, Province and FSA
POS_Data.sort_values(by=['PROVINCE_OF_TAX','FSA','CYCLE_WEEK_END_DATE__YYYY_MM_DD'],inplace=True)
POS_Data.head()

# COMMAND ----------

# DBTITLE 1,ENVIRONICS DATA
# ## Table Name
# df_2016 = "LOAD_RGM_DEMOSTATS2021_16"
# df_2017 = "LOAD_RGM_DEMOSTATS2021_17"
# df_2018 = "LOAD_RGM_DEMOSTATS2021_18"
# df_2019 = "LOAD_RGM_DEMOSTATS2021_19"
# df_2020 = "LOAD_RGM_DEMOSTATS2021_20"
# df_2021 = "LOAD_RGM_DEMOSTATS2021_21"

# # Read in library  
# df_2016 = readPySparkdf(df_2016).to_pandas_on_spark()
# df_2017 = readPySparkdf(df_2017).to_pandas_on_spark()
# df_2018 = readPySparkdf(df_2018).to_pandas_on_spark()
# df_2019 = readPySparkdf(df_2019).to_pandas_on_spark()
# df_2020 = readPySparkdf(df_2020).to_pandas_on_spark()
# df_2021 = readPySparkdf(df_2021).to_pandas_on_spark()

## Importing the Switching Information
df_2016 = spark.read.csv("dbfs:/FileStore/Environics/DemoStats2021_16_GEO.csv", header="true", inferSchema="true").toPandas()
df_2017 = spark.read.csv("dbfs:/FileStore/Environics/DemoStats2021_17_GEO.csv", header="true", inferSchema="true").toPandas()
df_2018 = spark.read.csv("dbfs:/FileStore/Environics/DemoStats2021_18_GEO.csv", header="true", inferSchema="true").toPandas()
df_2019 = spark.read.csv("dbfs:/FileStore/Environics/DemoStats2021_19_GEO.csv", header="true", inferSchema="true").toPandas()
df_2020 = spark.read.csv("dbfs:/FileStore/Environics/DemoStats2021_20_GEO.csv", header="true", inferSchema="true").toPandas()
df_2021 = spark.read.csv("dbfs:/FileStore/Environics/DemoStats_2021_TRENDS_GEO.csv", header="true", inferSchema="true").toPandas()


## Rename the dataframe with their actual column names
df_2016.rename(columns={"EHYBASPOP":"Total Population",
                        "EHYPTA2024":"Total 20 To 24",
                       "EHYPTA2529":"Total 25 To 29",
                       "EHYPTA3034":"Total 30 To 34",
                       "EHYPTA3539":"Total 35 To 39",
                       "EHYPTA4044":"Total 40 To 44",
                        "EHYPTA4549":"Total 45 To 49",
                        "EHYPTA5054":"Total 50 To 54",
                        "EHYPTA5559":"Total 55 To 59",
                        "EHYPTA6064":"Total 60 To 64",
                        "EHYPTA6569":"Total 65 To 69",
                        "EHYPTA7074":"Total 70 To 74",
                        "EHYPTA7579":"Total 75 To 79",
                        "EHYPTA8084":"Total 80 To 84",
                        "EHYPTA85P":"Total 85 Or Older",
                        "EHYTENHHD":"Total Households For Tenure",
                        "EHYTENOWN":"Owned",
                        "EHYTENRENT":"Rented",
                        "EHYTENBAND":"Band Housing",
                        "EHYHRI_020":"Household Income $0 To $19,999 (Constant Year 2015 $)",
                        "EHYHRI2040":"Household Income $20,000 To $39,999 (Constant Year 2015 $)",
                        "EHYHRI4060":"Household Income $40,000 To $59,999 (Constant Year 2015 $)",
                        "EHYHRI6080":"Household Income $60,000 To $79,999 (Constant Year 2015 $)",
                        "EHYHRIX100":"Household Income $80,000 To $99,999 (Constant Year 2015 $)",
                        "EHYHRI100P":"Household Income $100,000 Or Over (Constant Year 2015 $)",
                        "EHYHRIX125":"Household Income $100,000 To $124,999 (Constant Year 2015 $)",
                        "EHYHRIX150":"Household Income $125,000 To $149,999 (Constant Year 2015 $)",
                        "EHYHRIX200":"Household Income $150,000 To $199,999 (Constant Year 2015 $)",
                        "EHYHRI200P":"Household Income $200,000 Or Over (Constant Year 2015 $)",
                        "EHYHRIX300":"Household Income $200,000 To $299,999 (Constant Year 2015 $)",
                        "EHYHRI300P":"Household Income $300,000 Or Over (Constant Year 2015 $)",
                        "EHYEDUNCDD":"No Certificate, Diploma Or Degree",
                        "EHYEDUHSCE":"High School Certificate Or Equivalent",
                        "EHYEDUATCD":"Apprenticeship Or Trades Certificate Or Diploma",
                        "EHYEDUCOLL":"College, CEGEP Or Other Non-University Certificate Or Diploma",
                        "EHYEDUUDBB":"University Certificate Or Diploma Below Bachelor",
                        "EHYEDUUD":"University Degree",
                        "EHYEDUUDBD":"Bachelor's Degree",
                        "EHYEDUUDBP":"Above Bachelor's",
                        "ECYPTAAVG":"Average AGE",
                        "ECYHRIAVG":"Average Income"},inplace=True)


df_2017.rename(columns={"Y17BASPOP":"Total Population",
"Y17PTA2024":"Total 20 To 24",
"Y17PTA2529":"Total 25 To 29",
"Y17PTA3034":"Total 30 To 34",
"Y17PTA3539":"Total 35 To 39",
"Y17PTA4044":"Total 40 To 44",
"Y17PTA4549":"Total 45 To 49",
"Y17PTA5054":"Total 50 To 54",
"Y17PTA5559":"Total 55 To 59",
"Y17PTA6064":"Total 60 To 64",
"Y17PTA6569":"Total 65 To 69",
"Y17PTA7074":"Total 70 To 74",
"Y17PTA7579":"Total 75 To 79",
"Y17PTA8084":"Total 80 To 84",
"Y17PTA85P":"Total 85 Or Older",
"Y17TENHHD":"Total Households For Tenure",
"Y17TENOWN":"Owned",
"Y17TENRENT":"Rented",
"Y17TENBAND":"Band Housing",
"Y17HRI_020":"Household Income $0 To $19,999 (Constant Year 2015 $)",
"Y17HRI2040":"Household Income $20,000 To $39,999 (Constant Year 2015 $)",
"Y17HRI4060":"Household Income $40,000 To $59,999 (Constant Year 2015 $)",
"Y17HRI6080":"Household Income $60,000 To $79,999 (Constant Year 2015 $)",
"Y17HRIX100":"Household Income $80,000 To $99,999 (Constant Year 2015 $)",
"Y17HRI100P":"Household Income $100,000 Or Over (Constant Year 2015 $)",
"Y17HRIX125":"Household Income $100,000 To $124,999 (Constant Year 2015 $)",
"Y17HRIX150":"Household Income $125,000 To $149,999 (Constant Year 2015 $)",
"Y17HRIX200":"Household Income $150,000 To $199,999 (Constant Year 2015 $)",
"Y17HRI200P":"Household Income $200,000 Or Over (Constant Year 2015 $)",
"Y17HRIX300":"Household Income $200,000 To $299,999 (Constant Year 2015 $)",
"Y17HRI300P":"Household Income $300,000 Or Over (Constant Year 2015 $)",
"Y17EDUNCDD":"No Certificate, Diploma Or Degree",
"Y17EDUHSCE":"High School Certificate Or Equivalent",
"Y17EDUATCD":"Apprenticeship Or Trades Certificate Or Diploma",
"Y17EDUCOLL":"College, CEGEP Or Other Non-University Certificate Or Diploma",
"Y17EDUUDBB":"University Certificate Or Diploma Below Bachelor",
"Y17EDUUD":"University Degree",
"Y17EDUUDBD":"Bachelor's Degree",
"Y17EDUUDBP":"Above Bachelor's",
"Y17PTAAVG":"Average AGE",
"Y17HRIAVG":"Average Income"},inplace=True)

df_2018.rename(columns={"Y18BASPOP":"Total Population",
"Y18PTA2024":"Total 20 To 24",
"Y18PTA2529":"Total 25 To 29",
"Y18PTA3034":"Total 30 To 34",
"Y18PTA3539":"Total 35 To 39",
"Y18PTA4044":"Total 40 To 44",
"Y18PTA4549":"Total 45 To 49",
"Y18PTA5054":"Total 50 To 54",
"Y18PTA5559":"Total 55 To 59",
"Y18PTA6064":"Total 60 To 64",
"Y18PTA6569":"Total 65 To 69",
"Y18PTA7074":"Total 70 To 74",
"Y18PTA7579":"Total 75 To 79",
"Y18PTA8084":"Total 80 To 84",
"Y18PTA85P":"Total 85 Or Older",
"Y18TENHHD":"Total Households For Tenure",
"Y18TENOWN":"Owned",
"Y18TENRENT":"Rented",
"Y18TENBAND":"Band Housing",
"Y18HRI_020":"Household Income $0 To $19,999 (Constant Year 2015 $)",
"Y18HRI2040":"Household Income $20,000 To $39,999 (Constant Year 2015 $)",
"Y18HRI4060":"Household Income $40,000 To $59,999 (Constant Year 2015 $)",
"Y18HRI6080":"Household Income $60,000 To $79,999 (Constant Year 2015 $)",
"Y18HRIX100":"Household Income $80,000 To $99,999 (Constant Year 2015 $)",
"Y18HRI100P":"Household Income $100,000 Or Over (Constant Year 2015 $)",
"Y18HRIX125":"Household Income $100,000 To $124,999 (Constant Year 2015 $)",
"Y18HRIX150":"Household Income $125,000 To $149,999 (Constant Year 2015 $)",
"Y18HRIX200":"Household Income $150,000 To $199,999 (Constant Year 2015 $)",
"Y18HRI200P":"Household Income $200,000 Or Over (Constant Year 2015 $)",
"Y18HRIX300":"Household Income $200,000 To $299,999 (Constant Year 2015 $)",
"Y18HRI300P":"Household Income $300,000 Or Over (Constant Year 2015 $)",
"Y18EDUNCDD":"No Certificate, Diploma Or Degree",
"Y18EDUHSCE":"High School Certificate Or Equivalent",
"Y18EDUATCD":"Apprenticeship Or Trades Certificate Or Diploma",
"Y18EDUCOLL":"College, CEGEP Or Other Non-University Certificate Or Diploma",
"Y18EDUUDBB":"University Certificate Or Diploma Below Bachelor",
"Y18EDUUD":"University Degree",
"Y18EDUUDBD":"Bachelor's Degree",
"Y18EDUUDBP":"Above Bachelor's",
"Y18PTAAVG":"Average AGE",
"Y18HRIAVG":"Average Income"},inplace=True)

df_2019.rename(columns={"Y19BASPOP":"Total Population",
"Y19PTA2024":"Total 20 To 24",
"Y19PTA2529":"Total 25 To 29",
"Y19PTA3034":"Total 30 To 34",
"Y19PTA3539":"Total 35 To 39",
"Y19PTA4044":"Total 40 To 44",
"Y19PTA4549":"Total 45 To 49",
"Y19PTA5054":"Total 50 To 54",
"Y19PTA5559":"Total 55 To 59",
"Y19PTA6064":"Total 60 To 64",
"Y19PTA6569":"Total 65 To 69",
"Y19PTA7074":"Total 70 To 74",
"Y19PTA7579":"Total 75 To 79",
"Y19PTA8084":"Total 80 To 84",
"Y19PTA85P":"Total 85 Or Older",
"Y19TENHHD":"Total Households For Tenure",
"Y19TENOWN":"Owned",
"Y19TENRENT":"Rented",
"Y19TENBAND":"Band Housing",
"Y19HRI_020":"Household Income $0 To $19,999 (Constant Year 2015 $)",
"Y19HRI2040":"Household Income $20,000 To $39,999 (Constant Year 2015 $)",
"Y19HRI4060":"Household Income $40,000 To $59,999 (Constant Year 2015 $)",
"Y19HRI6080":"Household Income $60,000 To $79,999 (Constant Year 2015 $)",
"Y19HRIX100":"Household Income $80,000 To $99,999 (Constant Year 2015 $)",
"Y19HRI100P":"Household Income $100,000 Or Over (Constant Year 2015 $)",
"Y19HRIX125":"Household Income $100,000 To $124,999 (Constant Year 2015 $)",
"Y19HRIX150":"Household Income $125,000 To $149,999 (Constant Year 2015 $)",
"Y19HRIX200":"Household Income $150,000 To $199,999 (Constant Year 2015 $)",
"Y19HRI200P":"Household Income $200,000 Or Over (Constant Year 2015 $)",
"Y19HRIX300":"Household Income $200,000 To $299,999 (Constant Year 2015 $)",
"Y19HRI300P":"Household Income $300,000 Or Over (Constant Year 2015 $)",
"Y19EDUNCDD":"No Certificate, Diploma Or Degree",
"Y19EDUHSCE":"High School Certificate Or Equivalent",
"Y19EDUATCD":"Apprenticeship Or Trades Certificate Or Diploma",
"Y19EDUCOLL":"College, CEGEP Or Other Non-University Certificate Or Diploma",
"Y19EDUUDBB":"University Certificate Or Diploma Below Bachelor",
"Y19EDUUD":"University Degree",
"Y19EDUUDBD":"Bachelor's Degree",
"Y19EDUUDBP":"Above Bachelor's",
"Y19PTAAVG":"Average AGE",
"Y19HRIAVG":"Average Income"},inplace=True)

df_2020.rename(columns={"Y20BASPOP":"Total Population",
"Y20PTA2024":"Total 20 To 24",
"Y20PTA2529":"Total 25 To 29",
"Y20PTA3034":"Total 30 To 34",
"Y20PTA3539":"Total 35 To 39",
"Y20PTA4044":"Total 40 To 44",
"Y20PTA4549":"Total 45 To 49",
"Y20PTA5054":"Total 50 To 54",
"Y20PTA5559":"Total 55 To 59",
"Y20PTA6064":"Total 60 To 64",
"Y20PTA6569":"Total 65 To 69",
"Y20PTA7074":"Total 70 To 74",
"Y20PTA7579":"Total 75 To 79",
"Y20PTA8084":"Total 80 To 84",
"Y20PTA85P":"Total 85 Or Older",
"Y20TENHHD":"Total Households For Tenure",
"Y20TENOWN":"Owned",
"Y20TENRENT":"Rented",
"Y20TENBAND":"Band Housing",
"Y20HRI_020":"Household Income $0 To $19,999 (Constant Year 2015 $)",
"Y20HRI2040":"Household Income $20,000 To $39,999 (Constant Year 2015 $)",
"Y20HRI4060":"Household Income $40,000 To $59,999 (Constant Year 2015 $)",
"Y20HRI6080":"Household Income $60,000 To $79,999 (Constant Year 2015 $)",
"Y20HRIX100":"Household Income $80,000 To $99,999 (Constant Year 2015 $)",
"Y20HRI100P":"Household Income $100,000 Or Over (Constant Year 2015 $)",
"Y20HRIX125":"Household Income $100,000 To $124,999 (Constant Year 2015 $)",
"Y20HRIX150":"Household Income $125,000 To $149,999 (Constant Year 2015 $)",
"Y20HRIX200":"Household Income $150,000 To $199,999 (Constant Year 2015 $)",
"Y20HRI200P":"Household Income $200,000 Or Over (Constant Year 2015 $)",
"Y20HRIX300":"Household Income $200,000 To $299,999 (Constant Year 2015 $)",
"Y20HRI300P":"Household Income $300,000 Or Over (Constant Year 2015 $)",
"Y20EDUNCDD":"No Certificate, Diploma Or Degree",
"Y20EDUHSCE":"High School Certificate Or Equivalent",
"Y20EDUATCD":"Apprenticeship Or Trades Certificate Or Diploma",
"Y20EDUCOLL":"College, CEGEP Or Other Non-University Certificate Or Diploma",
"Y20EDUUDBB":"University Certificate Or Diploma Below Bachelor",
"Y20EDUUD":"University Degree",
"Y20EDUUDBD":"Bachelor's Degree",
"Y20EDUUDBP":"Above Bachelor's",
"Y20PTAAVG":"Average AGE",
"Y21HRIAVG":"Average Income"},inplace=True)

df_2021.rename(columns={"ECYBASPOP":"Total Population",
"ECYPTA2024":"Total 20 To 24",
"ECYPTA2529":"Total 25 To 29",
"ECYPTA3034":"Total 30 To 34",
"ECYPTA3539":"Total 35 To 39",
"ECYPTA4044":"Total 40 To 44",
"ECYPTA4549":"Total 45 To 49",
"ECYPTA5054":"Total 50 To 54",
"ECYPTA5559":"Total 55 To 59",
"ECYPTA6064":"Total 60 To 64",
"ECYPTA6569":"Total 65 To 69",
"ECYPTA7074":"Total 70 To 74",
"ECYPTA7579":"Total 75 To 79",
"ECYPTA8084":"Total 80 To 84",
"ECYPTA85P":"Total 85 Or Older",
"ECYTENHHD":"Total Households For Tenure",
"ECYTENOWN":"Owned",
"ECYTENRENT":"Rented",
"ECYTENBAND":"Band Housing",
"ECYHRI_020":"Household Income $0 To $19,999 (Constant Year 2015 $)",
"ECYHRI2040":"Household Income $20,000 To $39,999 (Constant Year 2015 $)",
"ECYHRI4060":"Household Income $40,000 To $59,999 (Constant Year 2015 $)",
"ECYHRI6080":"Household Income $60,000 To $79,999 (Constant Year 2015 $)",
"ECYHRIX100":"Household Income $80,000 To $99,999 (Constant Year 2015 $)",
"ECYHRI100P":"Household Income $100,000 Or Over (Constant Year 2015 $)",
"ECYHRIX125":"Household Income $100,000 To $124,999 (Constant Year 2015 $)",
"ECYHRIX150":"Household Income $125,000 To $149,999 (Constant Year 2015 $)",
"ECYHRIX200":"Household Income $150,000 To $199,999 (Constant Year 2015 $)",
"ECYHRI200P":"Household Income $200,000 Or Over (Constant Year 2015 $)",
"ECYHRIX300":"Household Income $200,000 To $299,999 (Constant Year 2015 $)",
"ECYHRI300P":"Household Income $300,000 Or Over (Constant Year 2015 $)",
"ECYEDUNCDD":"No Certificate, Diploma Or Degree",
"ECYEDUHSCE":"High School Certificate Or Equivalent",
"ECYEDUATCD":"Apprenticeship Or Trades Certificate Or Diploma",
"ECYEDUCOLL":"College, CEGEP Or Other Non-University Certificate Or Diploma",
"ECYEDUUDBB":"University Certificate Or Diploma Below Bachelor",
"ECYEDUUD":"University Degree",
"ECYEDUUDBD":"Bachelor's Degree",
"ECYEDUUDBP":"Above Bachelor's",
"Y21PTAAVG":"Average AGE",
"Y21HRIAVG":"Average Income"},inplace=True)

## Filter the data with the FSA and joining the data
df_2016['YEAR'] = 2016
df_2017['YEAR'] = 2017
df_2018['YEAR'] = 2018
df_2019['YEAR'] = 2019
df_2020['YEAR'] = 2020
df_2021['YEAR'] = 2021

pdList = [df_2016, df_2017, df_2018, df_2019, df_2020, df_2021]  # List of your dataframes
environics_df = pd.concat(pdList)

#Filter the FSA data only from socio economic data
environics_df = environics_df[environics_df['GEO'] == 'FSAQ420']
environics_df = environics_df.drop(columns=['GEO'],axis=1)

environics_df = environics_df[['CODE','YEAR','Total Population','Total 20 To 24','Total 25 To 29','Total 30 To 34','Total 35 To 39','Total 40 To 44','Total 45 To 49','Total 50 To 54','Total 55 To 59','Total 60 To 64','Total 65 To 69','Total 70 To 74','Total 75 To 79','Total 80 To 84','Total 85 Or Older','Total Households For Tenure','Owned','Rented',
'Band Housing',
'Household Income $0 To $19,999 (Constant Year 2015 $)',
'Household Income $20,000 To $39,999 (Constant Year 2015 $)',
'Household Income $40,000 To $59,999 (Constant Year 2015 $)',
'Household Income $60,000 To $79,999 (Constant Year 2015 $)',
'Household Income $80,000 To $99,999 (Constant Year 2015 $)',
'Household Income $100,000 Or Over (Constant Year 2015 $)',
'Household Income $100,000 To $124,999 (Constant Year 2015 $)',
'Household Income $125,000 To $149,999 (Constant Year 2015 $)',
'Household Income $150,000 To $199,999 (Constant Year 2015 $)',
'Household Income $200,000 Or Over (Constant Year 2015 $)',
'Household Income $200,000 To $299,999 (Constant Year 2015 $)',
'Household Income $300,000 Or Over (Constant Year 2015 $)',
'No Certificate, Diploma Or Degree',
'High School Certificate Or Equivalent',
'Apprenticeship Or Trades Certificate Or Diploma',
'College, CEGEP Or Other Non-University Certificate Or Diploma',
'University Certificate Or Diploma Below Bachelor',
'University Degree',
"Bachelor's Degree",
"Above Bachelor's",
'Average AGE',
'Average Income']]

## Removing the columns which are not required
#environics_df = environics_df.drop(columns=['SOURCE_SYSTEM','INSERT_TS','GEO'],axis=1)

environics_df.shape

# COMMAND ----------

## Converting the environics data to pandas dataframe to convert the dtypes 
#environics_df = environics_df.to_pandas()

## Conversion of the dtypes
cols = environics_df.columns.drop(['CODE','YEAR'])
environics_df[cols] = environics_df[cols].apply(pd.to_numeric, axis = 1)
environics_df['CODE'] = environics_df['CODE'].astype(object)
environics_df['YEAR'] = environics_df['YEAR'].astype(int)

## Convertion from pandas dataframe to pyspark-pandas dataframe
#environics_df = ps.from_pandas(environics_df)

# COMMAND ----------

##Interpolating the Population from yearly to weekly
environics_df['YEAR'] = pd.to_datetime(environics_df.YEAR, format='%Y')

## Selecting the list of variables from environics data set
#environics_df = environics_df[['CODE','Total Population','YEAR']]
environics_df = environics_df[['CODE','YEAR','Total 20 To 24','Total 25 To 29','Total 30 To 34','Total 35 To 39','Total 40 To 44','Total 45 To 49','Total 50 To 54','Total 55 To 59','Total 60 To 64','Total 65 To 69','Total 70 To 74','Total 75 To 79','Total 80 To 84','Total 85 Or Older','Household Income $0 To $19,999 (Constant Year 2015 $)','Household Income $20,000 To $39,999 (Constant Year 2015 $)','Household Income $40,000 To $59,999 (Constant Year 2015 $)','Household Income $60,000 To $79,999 (Constant Year 2015 $)','Household Income $80,000 To $99,999 (Constant Year 2015 $)','Household Income $100,000 Or Over (Constant Year 2015 $)','Household Income $100,000 To $124,999 (Constant Year 2015 $)','Household Income $125,000 To $149,999 (Constant Year 2015 $)','Household Income $150,000 To $199,999 (Constant Year 2015 $)','Household Income $200,000 Or Over (Constant Year 2015 $)','Household Income $200,000 To $299,999 (Constant Year 2015 $)','Household Income $300,000 Or Over (Constant Year 2015 $)','Average AGE','Average Income']]


# environics_df = (environics_df.set_index("YEAR")).groupby("CODE")[['Total 20 To 24','Total 25 To 29','Total 30 To 34','Total 35 To 39','Total 40 To 44','Total 45 To 49','Total 50 To 54','Total 55 To 59','Total 60 To 64','Total 65 To 69','Total 70 To 74','Total 75 To 79','Total 80 To 84','Total 85 Or Older','Household Income $0 To $19,999 (Constant Year 2015 $)','Household Income $20,000 To $39,999 (Constant Year 2015 $)','Household Income $40,000 To $59,999 (Constant Year 2015 $)','Household Income $60,000 To $79,999 (Constant Year 2015 $)','Household Income $80,000 To $99,999 (Constant Year 2015 $)','Household Income $100,000 Or Over (Constant Year 2015 $)','Household Income $100,000 To $124,999 (Constant Year 2015 $)','Household Income $125,000 To $149,999 (Constant Year 2015 $)','Household Income $150,000 To $199,999 (Constant Year 2015 $)','Household Income $200,000 Or Over (Constant Year 2015 $)','Household Income $200,000 To $299,999 (Constant Year 2015 $)','Household Income $300,000 Or Over (Constant Year 2015 $)']].resample('W').ffill().interpolate().reset_index()

environics_df  = (environics_df.set_index("YEAR")).groupby("CODE")[['Total 20 To 24','Total 25 To 29','Total 30 To 34','Total 35 To 39','Total 40 To 44','Total 45 To 49','Total 50 To 54','Total 55 To 59','Total 60 To 64','Total 65 To 69','Total 70 To 74','Total 75 To 79','Total 80 To 84','Total 85 Or Older','Household Income $0 To $19,999 (Constant Year 2015 $)','Household Income $20,000 To $39,999 (Constant Year 2015 $)','Household Income $40,000 To $59,999 (Constant Year 2015 $)','Household Income $60,000 To $79,999 (Constant Year 2015 $)','Household Income $80,000 To $99,999 (Constant Year 2015 $)','Household Income $100,000 Or Over (Constant Year 2015 $)','Household Income $100,000 To $124,999 (Constant Year 2015 $)','Household Income $125,000 To $149,999 (Constant Year 2015 $)','Household Income $150,000 To $199,999 (Constant Year 2015 $)','Household Income $200,000 Or Over (Constant Year 2015 $)','Household Income $200,000 To $299,999 (Constant Year 2015 $)','Household Income $300,000 Or Over (Constant Year 2015 $)','Average AGE','Average Income']].resample('W').mean().interpolate(method='linear').reset_index()

environics_df.head()

# COMMAND ----------

environics_df[environics_df['Average AGE'].isnull()]['CODE'].unique()

# COMMAND ----------

environics_df['Average AGE'].isnull().sum()

# COMMAND ----------

## Merging the POS and Environics Data together
POS_Data = pd.merge(POS_Data, environics_df, left_on=['FSA','CYCLE_WEEK_END_DATE__YYYY_MM_DD'], right_on=['CODE','YEAR'], how='left' )
POS_Data.head()

# COMMAND ----------

## Creating the groups of variables for clustering
POS_Data['Total_20_39'] = POS_Data['Total 20 To 24'] + POS_Data['Total 25 To 29'] + POS_Data['Total 30 To 34'] + POS_Data['Total 35 To 39']
POS_Data['Total_40_59'] = POS_Data['Total 40 To 44'] + POS_Data['Total 45 To 49'] + POS_Data['Total 50 To 54'] + POS_Data['Total 55 To 59']
POS_Data['Total 60+'] = POS_Data['Total 60 To 64'] + POS_Data['Total 65 To 69'] + POS_Data['Total 70 To 74'] + POS_Data['Total 75 To 79'] + POS_Data['Total 80 To 84'] + POS_Data['Total 85 Or Older']

POS_Data['Household Income <$60000'] = POS_Data['Household Income $0 To $19,999 (Constant Year 2015 $)']+POS_Data['Household Income $20,000 To $39,999 (Constant Year 2015 $)'] + POS_Data['Household Income $40,000 To $59,999 (Constant Year 2015 $)']
POS_Data['Household Income $60000 - $99000'] = POS_Data['Household Income $60,000 To $79,999 (Constant Year 2015 $)']+POS_Data['Household Income $80,000 To $99,999 (Constant Year 2015 $)']
POS_Data['Household Income >=$100000'] = POS_Data['Household Income $100,000 Or Over (Constant Year 2015 $)'] + POS_Data['Household Income $100,000 To $124,999 (Constant Year 2015 $)'] + POS_Data['Household Income $125,000 To $149,999 (Constant Year 2015 $)'] + POS_Data['Household Income $150,000 To $199,999 (Constant Year 2015 $)'] + POS_Data['Household Income $200,000 Or Over (Constant Year 2015 $)'] + POS_Data['Household Income $200,000 To $299,999 (Constant Year 2015 $)'] + POS_Data['Household Income $300,000 Or Over (Constant Year 2015 $)']

POS_Data = POS_Data[['PROVINCE_OF_TAX','FSA','CYCLE_WEEK_END_DATE__YYYY_MM_DD','TOTAL_STICK_SOLD','PRICE_PER_STICK','Total_20_39','Total_40_59',
                     'Total 60+','Household Income <$60000','Household Income $60000 - $99000','Household Income >=$100000','Average AGE','Average Income']]

# COMMAND ----------

POS_Data.isnull().sum()

# COMMAND ----------

## Removing the null values as the environics_df does not data after 2021-01-03. Hence removing those rows
POS_Data.dropna(inplace=True)
POS_Data.isnull().sum()

# COMMAND ----------

## Keeping the relvant columns and renaming it
#POS_Data.drop(columns=['Total Population','CODE','YEAR'],axis =1, inplace=True)
#POS_Data = POS_Data[['PROVINCE_OF_TAX','FSA','CYCLE_WEEK_END_DATE__YYYY_MM_DD','TOTAL_STICK_SOLD','PRICE_PER_STICK','Total Population_y']]
#POS_Data.rename(columns = {'Total Population_y':'Total Population'},inplace=True)
POS_Data.head()

# COMMAND ----------

## Replacing the spaace with "_" in the column names and calulating the % change
POS_Data.columns = [c.replace(' ', '_') for c in POS_Data.columns]
POS_Data.head()

# COMMAND ----------

## Converting the column from absolute values to percentage change values
# list_chng = list(['TOTAL_STICK_SOLD','PRICE_PER_STICK','Total_20_To_24','Total_25_To_29','Total_30_To_34','Total_35_To_39','Total_40_To_44','Total_45_To_49','Total_50_To_54','Total_55_To_59','Total_60_To_64','Total_65_To_69','Total_70_To_74','Total_75_To_79','Total_80_To_84','Total_85_Or_Older'])

# for i in list_chng:
#   POS_Data[i] = POS_Data.sort_values(['PROVINCE_OF_TAX','FSA','CYCLE_WEEK_END_DATE__YYYY_MM_DD']).groupby(['PROVINCE_OF_TAX', 'FSA'])[i].pct_change()

list_chng = list(['TOTAL_STICK_SOLD','Total_20_39','Total_40_59','Total_60+'])

for i in list_chng:
  POS_Data[i] = POS_Data.sort_values(['PROVINCE_OF_TAX','FSA','CYCLE_WEEK_END_DATE__YYYY_MM_DD']).groupby(['PROVINCE_OF_TAX', 'FSA'])[i].pct_change()

POS_Data.head()

# COMMAND ----------

## Removing the rows which have NAN in % change variables
POS_Data.dropna(inplace=True)
print("Shape of POS_Data", POS_Data.shape)
POS_Data.isnull().sum()

# COMMAND ----------

POS_Data.PROVINCE_OF_TAX.value_counts()

# COMMAND ----------

## Dropping the rows where there is np.inf values because of the change percentage change as the base value is 0
print("Shape of the data before: ",POS_Data.shape)

POS_Data.replace([np.inf, -np.inf], np.nan, inplace=True)
# POS_Data.dropna(subset=["TOTAL_STICK_SOLD", "PRICE_PER_STICK", "Total_20_To_24", "Total_25_To_29","Total_30_To_34", "Total_35_To_39", "Total_40_To_44","Total_45_To_49", "Total_50_To_54", "Total_55_To_59", "Total_60_To_64", "Total_65_To_69", "Total_70_To_74", "Total_75_To_79","Total_80_To_84","Total_85_Or_Older"],axis = 0, inplace=True)

POS_Data.dropna(subset=["TOTAL_STICK_SOLD", "PRICE_PER_STICK", "Total_20_39", "Total_40_59","Total_60+", "Household_Income_<$60000", "Household_Income_$60000_-_$99000","Household_Income_>=$100000","Average_AGE","Average_Income"],axis = 0, inplace=True)

print("Shape of the data after: ",POS_Data.shape)

# COMMAND ----------

# DBTITLE 1,Model Building Alberta - Analysis 1
POS_Province = POS_Data[POS_Data['PROVINCE_OF_TAX']=='Alberta']
POS_Province.drop(['CYCLE_WEEK_END_DATE__YYYY_MM_DD','PROVINCE_OF_TAX'],axis =1, inplace=True)
POS_Province = POS_Province[['FSA','TOTAL_STICK_SOLD','PRICE_PER_STICK','Household_Income_<$60000','Household_Income_$60000_-_$99000','Household_Income_>=$100000','Average_Income']]
POS_Province.head()

# COMMAND ----------

## Min Max Scaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
POS_Province[['TOTAL_STICK_SOLD','PRICE_PER_STICK','Household_Income_<$60000','Household_Income_$60000_-_$99000','Household_Income_>=$100000','Average_Income']] = scaler.fit_transform(POS_Province[['TOTAL_STICK_SOLD','PRICE_PER_STICK','Household_Income_<$60000','Household_Income_$60000_-_$99000','Household_Income_>=$100000','Average_Income']])

#['TOTAL_STICK_SOLD','Household_Income_<$60000','Household_Income_$60000_-_$99000','Household_Income_>=$100000']

print("************************************************")
## Shape of dataframe
print("POS DataFrame shape is: ",POS_Province.shape)
print("************************************************")

# COMMAND ----------

# Correlation on Provincial and FSA level data
# provincial_fsa_correlation = POS_Ontario.groupby(['FSA'],as_index = True).corr()
# provincial_fsa_correlation = provincial_fsa_correlation.reset_index(level=['FSA'])
provincial_fsa_correlation = POS_Province.corr()

#provincial_fsa_correlation.drop(['FSA'],axis=1,inplace=True)
# Rename the columns for pairplots
provincial_fsa_correlation.rename(columns={'Household_Income_<$60000':"<60000",'Household_Income_$60000_-_$99000':'60000_to_99000',
       'Household_Income_>=$100000':'>100000'},inplace=True)
display(provincial_fsa_correlation)

sns.set(rc = {'figure.figsize':(15,8)})
sns.heatmap(provincial_fsa_correlation, 
        xticklabels=provincial_fsa_correlation.columns,
        yticklabels=provincial_fsa_correlation.columns, cmap="YlGnBu")

# COMMAND ----------

# Get the position of categorical columns
catColumnsPos = [POS_Province.columns.get_loc(col) for col in list(POS_Province.select_dtypes('object').columns)]
print('Categorical columns           : {}'.format(list(POS_Province.select_dtypes('object').columns)))
print('Categorical columns position  : {}'.format(catColumnsPos))

# Convert dataframe to matrix
dfMatrix = POS_Province.to_numpy()
dfMatrix

# COMMAND ----------

# # Import module for k-protoype cluster
# from kmodes.kprototypes import KPrototypes

# # Import module for data visualization
# from plotnine import *
# import plotnine

# #Choose optimal K using Elbow method
# cost = []
# for cluster in range(1, 10):
#   try:
#       kprototype = KPrototypes(n_jobs = -1, n_clusters = cluster, init = 'Huang', random_state = 0)
#       kprototype.fit_predict(dfMatrix, categorical = catColumnsPos)
#       cost.append(kprototype.cost_)
#       print('Cluster initiation: {}'.format(cluster))
#   except:
#       break
# # Converting the results into a dataframe and plotting them
# df_cost = pd.DataFrame({'Cluster':range(1, 10), 'Cost':cost})
# #Data viz
# plotnine.options.figure_size = (8, 4.8)
# (
#     ggplot(data = df_cost)+
#     geom_line(aes(x = 'Cluster',
#                   y = 'Cost'))+
#     geom_point(aes(x = 'Cluster',
#                    y = 'Cost'))+
#     geom_label(aes(x = 'Cluster',
#                    y = 'Cost',
#                    label = 'Cluster'),
#                size = 10,
#                nudge_y = 1000) +
#     labs(title = 'Optimal number of cluster with Elbow Method')+
#     xlab('Number of Clusters k')+
#     ylab('Cost')+
#     theme_minimal()
# )

# COMMAND ----------

# # Fit the cluster ( Model Training )
kprototype = KPrototypes(n_jobs = -1, n_clusters = 2, init = 'Huang', random_state = 0)
kprototype.fit_predict(dfMatrix, categorical = catColumnsPos)

# Cluster centorid
print("Cluster Centroids: ",kprototype.cluster_centroids_)
# Check the iteration of the clusters created
print("n_iter: ",kprototype.n_iter_)
# Check the cost of the clusters created
print("cost: ",kprototype.cost_)

# Add the cluster to the dataframe
POS_Province['Cluster Labels'] = kprototype.labels_
POS_Province['Segment'] = POS_Province['Cluster Labels'].map({0:'First', 1:'Second'})
# Order the cluster
POS_Province['Segment'] = POS_Province['Segment'].astype('category')
POS_Province['Segment'] = POS_Province['Segment'].cat.reorder_categories(['First','Second'])
POS_Province.head()

# COMMAND ----------

## Section 1
## Check if there are any overlap between teh clusters
first = list(POS_Province[POS_Province['Segment']=='First']['FSA'].unique())
second =  list(POS_Province[POS_Province['Segment']=='Second']['FSA'].unique())
#third =  list(POS_Province[POS_Province['Segment']=='Third']['FSA'].unique())

print("Common FSA between First and Second Segment: ",len(set(first) & set(second)))
# print("Common FSA between First and Third Segment: ",len(set(first) & set(third)))
# print("Common FSA between Second and Third Segment: ",len(set(second) & set(third)))

## Section 2
## Removing the common FSA and giving prefrences to First then Second and then Third cluster
list1 = set(first) & set(second)
# list2 = set(first) & set(third)
# list3 = set(second) & set(third)

POS_Province.loc[POS_Province.FSA.isin(list1), 'Segment'] = "First"
#POS_Province.loc[POS_Province.FSA.isin(list2), 'Segment'] = "First"
#POS_Province.loc[POS_Province.FSA.isin(list3), 'Segment'] = "Second"

## Section 3
## Check if there are any common FSA between the clusters
first = list(POS_Province[POS_Province['Segment']=='First']['FSA'].unique())
second =  list(POS_Province[POS_Province['Segment']=='Second']['FSA'].unique())
#third =  list(POS_Province[POS_Province['Segment']=='Third']['FSA'].unique())

# COMMAND ----------

## Invert the MinMax Scaled data
POS_Province[['TOTAL_STICK_SOLD','PRICE_PER_STICK','Household_Income_<$60000','Household_Income_$60000_-_$99000','Household_Income_>=$100000','Average_Income']] = scaler.inverse_transform(POS_Province[['TOTAL_STICK_SOLD','PRICE_PER_STICK','Household_Income_<$60000','Household_Income_$60000_-_$99000','Household_Income_>=$100000','Average_Income']])
POS_Province.head()

# COMMAND ----------

# Cluster interpretation
POS_Province.rename(columns = {'Cluster Labels':'Total Count'}, inplace = True)
POS_Province.groupby('Segment').agg(
  {
    'Total Count':'count',
    'FSA': lambda x: x.value_counts().index[0],
    'TOTAL_STICK_SOLD': 'mean',
    'PRICE_PER_STICK': 'mean',
    'Household_Income_<$60000': 'mean',
    'Household_Income_$60000_-_$99000': 'mean',
    'Household_Income_>=$100000': 'mean',
    'Average_Income': 'mean',
    }).reset_index()

# COMMAND ----------

# Cluster interpretation
POS_Province.rename(columns = {'Cluster Labels':'Total Count'}, inplace = True)
POS_Province.groupby('Segment').agg(
  {
    'Total Count':'count',
    'FSA': lambda x: x.value_counts().index[0],
    'TOTAL_STICK_SOLD': 'min',
    'PRICE_PER_STICK': 'min',
    'Household_Income_<$60000': 'min',
    'Household_Income_$60000_-_$99000': 'min',
    'Household_Income_>=$100000': 'min',
    'Average_Income': 'min',
    }).reset_index()

# COMMAND ----------

# Cluster interpretation
POS_Province.rename(columns = {'Cluster Labels':'Total Count'}, inplace = True)
POS_Province.groupby('Segment').agg(
  {
    'Total Count':'count',
    'FSA': lambda x: x.value_counts().index[0],
    'TOTAL_STICK_SOLD': 'max',
    'PRICE_PER_STICK': 'max',
    'Household_Income_<$60000': 'max',
    'Household_Income_$60000_-_$99000': 'max',
    'Household_Income_>=$100000': 'max',
    'Average_Income': 'max',
    }).reset_index()

# COMMAND ----------

#Visualize K-Prototype clustering 
clusters = kprototype.fit_predict(dfMatrix, categorical = catColumnsPos)
df_Cancelled=pd.DataFrame(POS_Province)
df_Cancelled['Cluster_id_K_Prototype']=clusters
print(df_Cancelled['Cluster_id_K_Prototype'].value_counts())

# Rename the columns for pairplots
df_Cancelled.rename(columns={'Household_Income_<$60000':"<60000",'Household_Income_$60000_-_$99000':'60000-99000',
       'Household_Income_>=$100000':'>100000'},inplace=True)

#df_Cancelled.drop(columns=['Segment','y_pred'],axis=0,inplace=True)

sns.pairplot(df_Cancelled,hue='Cluster_id_K_Prototype',palette='Dark2',diag_kind='kde',markers=["o", "s"])
#sns.pairplot(df_Cancelled, diag_kind="hist", hue='Cluster_id_K_Prototype',palette='Dark2',markers=["o", "s", "D"],corner = True)

# COMMAND ----------

## List of FSA in Segment 
print("List of FSA in segment First are: \n\n",POS_Province[POS_Province['Segment']=='First']['FSA'].unique())
print("**************************************************************")
print("List of FSA in segment Second are: \n\n",POS_Province[POS_Province['Segment']=='Second']['FSA'].unique())
print("**************************************************************")
# print("List of FSA in segment Third are: \n\n",POS_Province[POS_Province['Segment']=='Third']['FSA'].unique())
# print("**************************************************************")
#print("List of FSA in segment Fourth are: \n\n",df[df['Segment']=='Fourth']['FSA'].unique())

# COMMAND ----------

## Proportion of each cluster in terms of FSA
print("Proportion of FSA in First Cluster: ",len(POS_Province[POS_Province['Segment']=='First']['FSA'].unique())/len(POS_Province['FSA'].unique()))
print("Proportion of FSA in Second Cluster: ",len(POS_Province[POS_Province['Segment']=='Second']['FSA'].unique())/len(POS_Province['FSA'].unique()))
#print("Proportion of FSA in Third Cluster: ",len(POS_Province[POS_Province['Segment']=='Third']['FSA'].unique())/len(POS_Province['FSA'].unique()))

# COMMAND ----------

POS_Province.head()

# COMMAND ----------

## Writting the POS data to snowflake
spark.conf.set("spark.sql.execution.arrow.enabled","true")
POS_Province = spark.createDataFrame(POS_Province)
writePySparkDF(POS_Province,"CPO_POS_STORE_GROUPING_ALBERTA")
