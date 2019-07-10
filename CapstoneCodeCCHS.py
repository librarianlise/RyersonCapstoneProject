#-------------------- SET UP ENVIRONMENT AND READ DATA --------------------#

# import libraries
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
%matplotlib inline 


#set number of columns and rows to display
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 250)

#read in data file
df=pd.read_csv('C:/Users/lised/Desktop/Capstone/cchs20152016.csv', sep=',')

#confirm it's been read in
df.head(20)
df.info()

#-------------------- FORMAT CLASS VARIABLES SO THAT 1=YES AND 0=NO --------------------#

#class variables of flu shots: replace all Dont Know, Refusal, Not stated with null, and then remove those rows.  Change 'no' value from 2 to 0
df.FLU_005.value_counts()
df['flu_ever'] = df['FLU_005'].replace([7,8,9], np.nan)
df['flu_ever'] = df['flu_ever'].replace([2], 0)
df = df.dropna(subset=['flu_ever'])

df['flu_ever']=df['flu_ever'].astype('int')#.astype('category')
df['flu_ever'].value_counts()

#unweighted percentages of people getting flu shot
df['flu_ever'].value_counts().sort_index()/len(df)

#weighted average of percentage of people who have ever received a flu shot
mean_flu_ever=(df['flu_ever']*df['WTS_M']).sum()/(df['WTS_M'].sum())
mean_flu_ever

#class variables of flu shots in last year: replace all Dont Know, Refusal, Not stated with null, and then remove those rows
#replace all values of '1 year to less than 2 years ago', '2 years ago or more', and 'valid skip' (i.e., never had a flu shot) with 0='No'
df.FLU_010.value_counts()
df['flu_past_year'] = df['FLU_010'].replace([7,8,9], np.nan)
df['flu_past_year'] = df['flu_past_year'].replace([2,3,6], 0)
df = df.dropna(subset=['flu_past_year'])

df['flu_past_year']=df['flu_past_year'].astype('int')#.astype('category')
df['flu_past_year'].value_counts()

#unweighted percentages of people getting flu shot in past year
df['flu_past_year'].value_counts().sort_index()/len(df)


#weighted average of percentage of people who received a flu shot in the past year
mean_flu_past_year=(df['flu_past_year']*df['WTS_M']).sum()/(df['WTS_M'].sum())
mean_flu_past_year

#-------------------- CLEAN INDEPENDENT VARIABLES --------------------#

#AGE variable (DHHGAGE): look at age variable; aggregate values to create six age ranges based on data dictionary describing age ranges
df.DHHGAGE.value_counts().sort_index()
mapping_age = {1:1, 2:1, 3:1, 4:1, 5:2, 6:2, 7:3, 8:3, 9:3, 10:4, 11:4, 12:5, 13:5, 14:6, 15:6, 16:6}               
df['mapped_age'] = df['DHHGAGE'].map(lambda x : mapping_age[x]).astype(int)
df['mapped_age'].value_counts().sort_index()

age_unweighted = df.mapped_age.value_counts()/(df.mapped_age.count())
age_unweighted.sort_index()

(pd.crosstab(df.mapped_age, df.flu_ever, normalize = 'index')*100).round(decimals=1)
(pd.crosstab(df.mapped_age, df.flu_past_year, normalize = 'index')*100).round(decimals=1)


#SEX variable (DHH_SEX):
df.DHH_SEX.value_counts()
df['sex'] = df['DHH_SEX'].replace(2, 0)

df.sex.value_counts().sort_index()
df.sex.value_counts().sort_index()/(df.sex.count())

(pd.crosstab(df.sex, df.flu_ever, normalize = 'index')*100).round(decimals=1)
(pd.crosstab(df.sex, df.flu_past_year, normalize = 'index')*100).round(decimals=1)


#PERCEIVED HEALTH variable

df.GEN_005.value_counts()
df['self_health']=df.GEN_005.replace([1,2,3,4,5,7,8,9], [4,3,2,1,1, np.nan, np.nan, np.nan])
df = df.dropna(subset=['self_health'])

df.self_health = df.self_health.astype(int)
df.self_health.value_counts().sort_index()

(pd.crosstab(df.self_health, df.flu_ever, normalize = 'index')*100).round(decimals=1)
(pd.crosstab(df.self_health, df.flu_past_year, normalize = 'index')*100).round(decimals=1)



# EDUCATION variable
df.EHG2DVH3.value_counts()
df['education']=df.EHG2DVH3.replace(9, np.nan)
df = df.dropna(subset=['education'])

df.education = df.education.astype(int)
df.education.value_counts().sort_index()

(pd.crosstab(df.education, df.flu_ever, normalize = 'index')*100).round(decimals=1)
(pd.crosstab(df.education, df.flu_past_year, normalize = 'index')*100).round(decimals=1)



# has a regular doctor (1 = yes, 2 > set to 0 = no, remove 'no response' etc)

df.PHC_020.value_counts()
df['has_doctor']=df.PHC_020.replace([2,7,8,9], [0, np.nan, np.nan, np.nan])
df = df.dropna(subset=['has_doctor'])

df.has_doctor = df.has_doctor.astype(int)
df.has_doctor.value_counts().sort_index()

(pd.crosstab(df.has_doctor, df.flu_ever, normalize = 'index')*100).round(decimals=1)
(pd.crosstab(df.has_doctor, df.flu_past_year, normalize = 'index')*100).round(decimals=1)


#immigrant status

df.SDCDVIMM.value_counts()
df['is_immigrant']=df.SDCDVIMM.replace([2,9], [0, np.nan])
df = df.dropna(subset=['is_immigrant'])

df.is_immigrant = df.is_immigrant.astype(int)
df.is_immigrant.value_counts().sort_index()

(pd.crosstab(df.is_immigrant, df.flu_ever, normalize = 'index')*100).round(decimals=1)
(pd.crosstab(df.is_immigrant, df.flu_past_year, normalize = 'index')*100).round(decimals=1)


# PROVINCE variable (set to categorical)
df.GEO_PRV.value_counts()
df['province']=df.GEO_PRV.astype('category')

df.province.value_counts().sort_index()
df.province.value_counts().sort_index()/(df.province.count())

(pd.crosstab(df.province, df.flu_ever, normalize = 'index')*100).round(decimals=1)
(pd.crosstab(df.province, df.flu_past_year, normalize = 'index')*100).round(decimals=1)


##################################################################################33
# to do: INCOME (three levels according to household income and number of people living in household, following article by Chen(2007))
#df.INCDGHH.value_counts()
#df.DHHDGHSZ.value_counts()


#see names of last 15 variables to confirm range of variables needed for new data frame
df[df.columns[-15:]].tail(1)

#check data types
df.dtypes

#create new data frame with cleaned up variables
df2=df[df.columns[-9:]]
df2 = df2.dropna()
df2.dtypes

#draw correlation plot for all variables in data frame

corr = df2.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(df2.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(df2.columns)
ax.set_yticklabels(df2.columns)
plt.show()

#create correlation matrix
df2.corr()

df2.to_csv('CapstoneData.csv')


