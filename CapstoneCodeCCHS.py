#-------------------- SET UP ENVIRONMENT AND READ DATA --------------------#

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#set number of columns and rows to display
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 250)

#read in data file
df=pd.read_csv('C:/Users/lised/Desktop/Capstone/cchs20152016.csv', sep=',')

#confirm it's been read in
df.head(20)
df.info()

#-------------------- FORMAT CLASS VARIABLE SO THAT 1=YES AND 0=NO --------------------#

#class variables of flu shots in last year: replace all Dont Know, Refusal, Not stated with null, and then remove those rows
#replace all values of '1 year to less than 2 years ago', '2 years ago or more', and 'valid skip' (i.e., never had a flu shot) with 0='No'
df.FLU_010.value_counts()
df['flu_past_year'] = df['FLU_010'].replace([7,8,9], np.nan)
df['flu_past_year'] = df['flu_past_year'].replace([2,3,6], 0)
df = df.dropna(subset=['flu_past_year'])

df['flu_past_year']=df['flu_past_year'].astype(int)

#count and percentages of people getting flu shot in past year
df['flu_past_year'].value_counts().sort_index()
df['flu_past_year'].value_counts().sort_index()/len(df)


#-------------------- CLEAN INDEPENDENT VARIABLES --------------------#

#age variable
df.DHHGAGE.value_counts().sort_index()
mapping_age = {1:1, 2:1, 3:1, 4:1, 5:2, 6:2, 7:3, 8:3, 9:3, 10:4, 11:4, 12:5, 13:5, 14:6, 15:6, 16:6}               
df['age'] = df['DHHGAGE'].map(lambda x : mapping_age[x]).astype(int)
df['age'].value_counts().sort_index()
df['age'].value_counts().sort_index()/len(df)

(pd.crosstab(df.age, df.flu_past_year, normalize = 'index')*100).round(decimals=1)

#sex variable:
df.DHH_SEX.value_counts()
df['sex'] = df['DHH_SEX'].replace(2, 0)

df.sex.value_counts().sort_index()
df.sex.value_counts().sort_index()/(df.sex.count())

(pd.crosstab(df.sex, df.flu_past_year, normalize = 'index')*100).round(decimals=1)

#perceived health variable
df.GEN_005.value_counts()
df['self_health']=df.GEN_005.replace([1,2,3,4,5,7,8,9], [4,3,2,1,1, np.nan, np.nan, np.nan])
df = df.dropna(subset=['self_health'])

df.self_health = df.self_health.astype(int)
df.self_health.value_counts().sort_index()

(pd.crosstab(df.self_health, df.flu_past_year, normalize = 'index')*100).round(decimals=1)

# education variable
df.EHG2DVH3.value_counts()
df['education']=df.EHG2DVH3.replace(9, np.nan)
df = df.dropna(subset=['education'])

df.education = df.education.astype(int)
df.education.value_counts().sort_index()

(pd.crosstab(df.education, df.flu_past_year, normalize = 'index')*100).round(decimals=1)

(pd.crosstab(df.age, df.education, normalize = 'index')*100).round(decimals=1)

# has a regular doctor (1 = yes, 2 > set to 0 = no, remove 'no response' etc)
df.PHC_020.value_counts()
df['has_doctor']=df.PHC_020.replace([2,7,8,9], [0, np.nan, np.nan, np.nan])
df = df.dropna(subset=['has_doctor'])

df.has_doctor = df.has_doctor.astype(int)
df.has_doctor.value_counts().sort_index()

(pd.crosstab(df.has_doctor, df.flu_past_year, normalize = 'index')*100).round(decimals=1)

#immigrant status
df.SDCDVIMM.value_counts()
df['is_immigrant']=df.SDCDVIMM.replace([2,9], [0, np.nan])
df = df.dropna(subset=['is_immigrant'])

df.is_immigrant = df.is_immigrant.astype(int)
df.is_immigrant.value_counts().sort_index()

(pd.crosstab(df.is_immigrant, df.flu_past_year, normalize = 'index')*100).round(decimals=1)

# PROVINCE variable (set to categorical)
df.GEO_PRV.value_counts()
df['province']=df.GEO_PRV.astype('int')

#df['province']=df.province.replace([10, 11, 12, 13, 24, 35, 46, 47, 48, 59, 60, 61, 62], ['NF', 'PE', 'NS', 'NB', 'QC', 'ON', 'MB', 'SK', 'AB', 'BC', 'YK', 'NW', 'NV'])

df.province.value_counts().sort_index()
df.province.value_counts().sort_index()/(df.province.count())

(pd.crosstab(df.province, df.flu_past_year, normalize = 'index')*100).round(decimals=1)


##################################################################################33
#income levels (three levels according to household income and number of people living in household, following article by Chen(2007))
df.INCDGHH.value_counts().sort_index()
df['income']=df.INCDGHH.replace(9, np.nan)
df = df.dropna(subset=['income'])

df.income= df.income.astype(int)
df.income.value_counts().sort_index()

df.DHHDGHSZ.value_counts().sort_index()
df['household_size']=df.DHHDGHSZ.replace(9, np.nan)
df = df.dropna(subset=['household_size'])

df.household_size= df.household_size.astype(int)
df.household_size.value_counts().sort_index()


pd.crosstab(df.household_size, df.income)


#set low income
df.loc[((df['household_size'] < 3) & (df['income'] == 1)) | ((df['household_size'] == 3) & (df['income'] < 3)) | ((df['household_size'] == 4) & (df['income'] < 3)) | ((df['household_size'] > 4) & (df['income'] < 4)), 'income_level'] = 1

#set high income
df.loc[(((df['household_size'] < 3) & (df['income'] > 3)) | ((df['household_size'] > 2) & (df['income'] > 4))), 'income_level'] = 3

#everything else is middle income
df.loc[((df['income_level'] != 1) & (df['income_level'] != 3)), 'income_level'] = 2

df.income_level.value_counts()
df.income_level=df.income_level.astype(int)
df.income_level.value_counts()

(pd.crosstab(df.income_level, df.flu_past_year, normalize = 'index')*100).round(decimals=1)


# marital status
df.DHHGMS.value_counts()
df['marital_status'] = df.DHHGMS.replace(9, np.nan)
df = df.dropna(subset=['marital_status'])

df.marital_status = df.marital_status.astype('int')

(pd.crosstab(df.marital_status, df.flu_past_year, normalize = 'index')*100).round(decimals=1)
(pd.crosstab(df.marital_status, df.age, normalize = 'index')*100).round(decimals=1)

#any children 0-11 in the household, yes no

df.DHHDGL12.value_counts()
df['children_in_household'] = df.DHHDGL12.astype('int')

df['children_in_household'].value_counts()

(pd.crosstab(df.children_in_household, df.flu_past_year, normalize = 'index')*100).round(decimals=1)


#see names of last 15 variables to confirm range of variables needed for new data frame
df[df.columns[-15:]].tail(1)

#check data types
df.dtypes

#create new data frame with cleaned up variables
df2=df[df.columns[-13:]]
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
df2.to_csv('C:/Users/lised/Desktop/Capstone/CapstoneDataCleaned.csv')



#CLASSIFIERS



# start ---------- LOGISTIC REGRESSION CLASSIFIER, no balancing ------------#

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

df2.columns


#make a new data frame with the independent variables #2-13
X=df2[df2.columns[1:13]]
X.columns

#create dataframe of dummy variables for the marital status and province variables, as they are the only two non-binary nominal variables
X = pd.get_dummies(X, columns=['marital_status', 'province'])
X.columns

#make marital status of married and province of Ontario the baseline measurements by removing them
X = X.drop(['marital_status_1', 'province_35'], axis=1)
X.columns

#class variable to predict (flu_last_year)
y = df2.iloc[:,0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

classifier = LogisticRegression(random_state=4)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

#accuracy
classifier.score(X_test, y_test)

print(classification_report(y_test, y_pred))

# /end ---------- LOGISTIC REGRESSION CLASSIFIER ------------#

# start ---------------- RANDOM FOREST CLASSIFIER -------------#

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

X_trainRF, X_testRF, y_trainRF, y_testRF = train_test_split(X, y, test_size=0.33, random_state=0)

clf=RandomForestClassifier(n_estimators=100)
clf.fit(X_trainRF,y_trainRF)

y_predRF=clf.predict(X_testRF)

confusion_matrixRF = confusion_matrix(y_testRF, y_predRF)
print(confusion_matrixRF)


clf.score(X_testRF, y_testRF)

print(classification_report(y_testRF, y_predRF))


clf.score(X_testRF, y_testRF)

# end ---------------- RANDOM FOREST CLASSIFIER -------------#


# start ---------------- NAIVE BAYES CLASSIFIER -------------#



#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix


X_trainNB, X_testNB, y_trainNB, y_testNB = train_test_split(X, y, test_size=0.33, random_state=0)



#Create a Gaussian Classifier
model = GaussianNB()

# Train the model using the training sets
model.fit(X_trainNB,y_trainNB)

y_predNB = model.predict(X_testNB)


confusion_matrixNB = confusion_matrix(y_testNB, y_predNB)

print(confusion_matrixNB)

print(classification_report(y_testNB, y_predNB))


model.score(X_testRF, y_testRF)


#tn, fp, fn, tp = confusion_matrix(y_testNB, y_predNB).ravel()
#(tn, fp, fn, tp)


# end ---------------- NAIVE BAYES CLASSIFIER -------------#





# start ---------- DECISION TREE RULES ------------#

from sklearn import datasets
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree.export import export_text
from sklearn import tree



#using my own data
df3 = pd.get_dummies(df2, columns=['marital_status', 'province'])

df_ind = df3.iloc[:, 1:27]
df_class = df3.iloc[:, 0]

dtree = tree.DecisionTreeClassifier(criterion='entropy')
dtree.fit(X, y)

dotfile = open("dtree.dot", 'w')
tree.export_graphviz(dtree, out_file=dotfile, feature_names=df_ind.columns)
dotfile.close()
