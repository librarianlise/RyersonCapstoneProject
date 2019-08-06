#-------------------- SET UP ENVIRONMENT AND READ DATA --------------------#

#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#set number of columns and rows to display
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 100)

#read in data file
df=pd.read_csv('C:/Users/lised/Desktop/Capstone/cchs20152016.csv', sep=',')

#confirm it's been read in
df.head(20)
df.info()

#-------------------- CLEAN/FORMAT CLASS VARIABLE --------------------#

#class variables of flu shots in last year: replace all Dont Know, Refusal, Not stated with null, and then remove those null rows
#replace all values of '1 year to less than 2 years ago', '2 years ago or more', and 'valid skip' (i.e., never had a flu shot) with 0='No'
df.FLU_010.value_counts()
df['flu_past_year'] = df['FLU_010'].replace([7,8,9], np.nan)
df['flu_past_year'] = df['flu_past_year'].replace([2,3,6], 0)
df = df.dropna(subset=['flu_past_year'])

#check how new variable displays and check type; change to integer
df.dtypes
df['flu_past_year'].value_counts()
df['flu_past_year']=df['flu_past_year'].astype(int)

#unweighted count and percentages of people getting flu shot in past year
df['flu_past_year'].value_counts().sort_index()
((df['flu_past_year'].value_counts().sort_index()*100)/len(df)).round(decimals=1)

#weighted percentage of people getting a flu shot in the past year
(df['flu_past_year']*df['WTS_M']).sum()/(df['WTS_M'].sum())

#-------------------- CLEAN/FORMAT INDEPENDENT VARIABLES --------------------#

#age variable (combine categories and recode; no values to remove)
df.DHHGAGE.value_counts().sort_index()
mapping_age = {1:1, 2:1, 3:1, 4:1, 5:2, 6:2, 7:3, 8:3, 9:3, 10:4, 11:4, 12:5, 13:5, 14:6, 15:6, 16:6}               
df['age'] = df['DHHGAGE'].map(lambda x : mapping_age[x]).astype(int)
df['age'].value_counts().sort_index()

#sex variable (recode; no values to remove)
df.DHH_SEX.value_counts()
df['sex'] = df['DHH_SEX'].replace(2, 0)
df.sex.value_counts().sort_index()

#perceived health variable (combine categories and recode; remove rows with non-responses)
df.GEN_005.value_counts()
df['self_health']=df.GEN_005.replace([1,2,3,4,5,7,8,9], [4,3,2,1,1, np.nan, np.nan, np.nan])
df = df.dropna(subset=['self_health']).astype(int)
df.self_health.value_counts().sort_index()

# education variable
df.EHG2DVH3.value_counts()

##because of large number of non-responses, check proportions of class variable for those with education variable = 9
(pd.crosstab(df.EHG2DVH3, df.flu_past_year, normalize = 'index')*100).round(decimals=1)

##proportion similar; remove rows with non-responses
df['education']=df.EHG2DVH3.replace(9, np.nan)
df = df.dropna(subset=['education'])

df.education = df.education.astype(int)
df.education.value_counts().sort_index()

#has a regular doctor variable (recode; remove rows with non-responses)
df.PHC_020.value_counts()
df['has_doctor']=df.PHC_020.replace([2,7,8,9], [0, np.nan, np.nan, np.nan])
df = df.dropna(subset=['has_doctor'])

df.has_doctor = df.has_doctor.astype(int)
df.has_doctor.value_counts().sort_index()

##immigrant status variable
df.SDCDVIMM.value_counts()

##because of large number of non-responses, check proportions of class variable for those with immigrant variable = 9
(pd.crosstab(df.SDCDVIMM, df.flu_past_year, normalize = 'index')*100).round(decimals=1)

##proportion similar; remove rows with non-responses
df['is_immigrant']=df.SDCDVIMM.replace([2,9], [0, np.nan])
df = df.dropna(subset=['is_immigrant'])

df.is_immigrant = df.is_immigrant.astype(int)
df.is_immigrant.value_counts().sort_index()

#province variable (replace with text abbreviations)
df.GEO_PRV.value_counts()
df['province']=df.GEO_PRV.astype('int')

df['province']=df.province.replace([10, 11, 12, 13, 24, 35, 46, 47, 48, 59, 60, 61, 62], ['NF', 'PE', 'NS', 'NB', 'QC', 'ON', 'MB', 'SK', 'AB', 'BC', 'YK', 'NW', 'NV'])

df.province.value_counts().sort_index()

#income level variable (create three levels according to household income and number of people living in household
#following article by Chen(2007); remove rows with non-responses)

##first clean income variable; remove rows with non-responses
df.INCDGHH.value_counts().sort_index()
df['income']=df.INCDGHH.replace(9, np.nan)
df = df.dropna(subset=['income'])

df.income= df.income.astype(int)
df.income.value_counts().sort_index()

##second clean household size variable; remove rows with non-responses
df.DHHDGHSZ.value_counts().sort_index()
df['household_size']=df.DHHDGHSZ.replace(9, np.nan)
df = df.dropna(subset=['household_size'])

df.household_size= df.household_size.astype(int)
df.household_size.value_counts().sort_index()

##set new income_level variable for low income
df.loc[((df['household_size'] < 3) & (df['income'] == 1)) | ((df['household_size'] == 3) & (df['income'] < 3)) | ((df['household_size'] == 4) & (df['income'] < 3)) | ((df['household_size'] > 4) & (df['income'] < 4)), 'income_level'] = 1

##set new income_level variable for high income
df.loc[(((df['household_size'] < 3) & (df['income'] > 3)) | ((df['household_size'] > 2) & (df['income'] > 4))), 'income_level'] = 3

##set new income_level variable for medium/middle income
df.loc[((df['income_level'] != 1) & (df['income_level'] != 3)), 'income_level'] = 2

df.income_level=df.income_level.astype(int)
df.income_level.value_counts()

# marital status variable (replace with text abbreviations and remove rows with non-responses)
df.DHHGMS.value_counts()
df['marital_status'] = df.DHHGMS.replace(9, np.nan)
df = df.dropna(subset=['marital_status'])

df['marital_status']=df.marital_status.replace([1, 2, 3, 4], ['Married', 'Common-law', 'Widowed/Divorced/Separated', 'Single'])

df.marital_status.value_counts()

#children in household variable (verify)
df.DHHDGL12.value_counts()
df['children_in_household'] = df.DHHDGL12.astype('int')

df['children_in_household'].value_counts()

#-------------------- CREATE FINAL DATA FRAME FOR ANALYSIS --------------------#

#see names of last 15 variables to confirm range of variables needed for new data frame
df[df.columns[-15:]].tail(1)

#check data types
df.dtypes

#create new data frame with cleaned up variables
df2=df[df.columns[-14:]]
df2.dtypes

#drop intermediary variables used to create income_level
df2=df2.drop(['income', 'household_size'], axis=1)
df2.dtypes

#save cleaned data for use in analysis in R
df2.to_csv('C:/Users/lised/Desktop/Capstone/CapstoneDataCleaned.csv')



#-------------------- CREATE CORRELATION MATRIX AND PLOT --------------------#


df2=pd.read_csv('C:/Users/lised/Desktop/Capstone/CapstoneDataCleaned.csv', sep=',')

(pd.crosstab([df2.age, df2.has_doctor, df2.self_health, df2.children_in_household], df2.flu_past_year, normalize = 'index')*100).round(decimals=1)
pd.crosstab([df2.age, df2.], df2.flu_past_year, normalize = 'index')
pd.crosstab([df2.has_dc], df2.flu_past_year)
pd.crosstab([df2.has_doctor], df2.flu_past_year, normalize = 'index')






#create temporary data frame by removing the two categorical variables
df3=df2.drop(['Unnamed: 0', 'WTS_M', 'province', 'marital_status'], axis=1)

#create correlation matrix
corr = df3.corr()
corr

#draw correlation plot for all variables in data frame
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(df3.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(df3.columns)
ax.set_yticklabels(df3.columns)
plt.show()