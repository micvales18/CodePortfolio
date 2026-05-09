# -*- coding: utf-8 -*-
"""
Created on Sat May  9 16:43:06 2026

@author: mvale
"""

# data manipulation and visualization libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# model building-related libraries
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import statsmodels.stats.api as sms
from statsmodels.stats.outliers_influence import variance_inflation_factor

# for table readability
pd.set_option("display.max_rows", 100)

# for getting different metric scores
from sklearn import metrics
from sklearn.metrics import (f1_score,
                            accuracy_score,
                            recall_score,
                            precision_score,
                            confusion_matrix,
                            classification_report,
                            roc_auc_score,
                            precision_recall_curve,
                            roc_curve,
                            make_scorer)


import warnings
warnings.filterwarnings('ignore')

# read in data
data = pd.read_csv(["REDACTED"])

# check for duplicate leads
data.ID.nunique()

# given that all entries are unique, we can drop the ID column
data.drop(['ID'], axis=1, inplace=True)

# summary descriptive statistics
data.describe().T

# make a list of categorical variables
catcols = list(data.select_dtypes('object').columns)

# print list of categorical varialbes and their counts
for i in catcols:
    print(data[i].value_counts())
    print('__'*20)

# define histogram/boxplot function for visualization
def hist_box(data, col):
    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': (0.15, 0.85)}, figsize=(12, 6))
    # include each of the graphs in the figure
    sns.boxplot(data=data, x=col, ax=ax_box, showmeans=True)
    sns.histplot(data=data, x=col, kde=True, ax=ax_hist)
    plt.show()

# visualize website visits distribution
hist_box(data, "website_visits")

# visualize time spent on website distribution
hist_box(data, "time_spent_on_website")

# visualize page views per visit distribution
hist_box(data, "page_views_per_visit")

# current occupation
sns.countplot(x=data['current_occupation'])

# get percentages for each category
data['current_occupation'].value_counts(normalize='True')

# first interactions
sns.countplot(x=data['first_interaction'])

data['first_interaction'].value_counts(normalize=True)

# profile completed
sns.countplot(x=data['profile_completed'])
data['profile_completed'].value_counts(normalize=True)

# last activity
sns.countplot(x=data['last_activity'])
data['last_activity'].value_counts(normalize=True)

# print media type 1
sns.countplot(x=data['print_media_type1'])
data['print_media_type1'].value_counts(normalize=True)

# print media type 2
sns.countplot(x=data['print_media_type2'])
data['print_media_type2'].value_counts(normalize=True)

# digital media
sns.countplot(x=data['digital_media'])
data['digital_media'].value_counts(normalize=True)

# educational channels
sns.countplot(x=data['educational_channels'])
data['educational_channels'].value_counts(normalize=True)

# referral
sns.countplot(x=data['referral'])
data['referral'].value_counts(normalize=True)

# status
sns.countplot(x=data['status'])
data['status'].value_counts(normalize=True)

# create a list of numeric variables
numcols = data.select_dtypes(include=np.number).columns.tolist()

# plot correlation matrix heatmap
sns.heatmap(data[numcols].corr(), annot=True)
plt.show()

# Create a stacked barplot function to help visualize multivariate analysis
def stacked_barplot(data,predictor,target,figsize=(10,6)):
    (pd.crosstab(data[predictor],data[target],normalize='index')*100).plot(kind='bar',figsize=figsize,stacked=True)
    plt.legend(loc='lower right')
    plt.ylabel(target)

# How does current occupation affect lead status?
# Plot current occupation and status
stacked_barplot(data,'current_occupation','status')

# understand the ages of different occupation types
data.groupby(['current_occupation'])['age'].describe()

# Do first channels of interaction have an impact on lead status?
# Plot first interaction and status
stacked_barplot(data,'first_interaction','status')

# What forms of interactions are most effective with prospects?
# Plot last activity and status
stacked_barplot(data,'last_activity','status')

# Which channels have the highest lead conversion rates?
# create a list of possible advert channels
channels = ['print_media_type1','print_media_type2','digital_media','educational_channels','referral']

# plot status against each channel
for i in channels:
    stacked_barplot(data,i,'status')


# Does having more detail about a prospect increase the chance of conversion?
# plot profile completed and status
stacked_barplot(data,'profile_completed','status')

# no missing values
# no feature engineering necessary
# Outlier detection:
#    as visualized in the univariate analysis,
#    variables "website_visits" and "page_views_per_visit"
#    have a fair number of outliers.
# Separate independent and dependent variables
predictors = data.drop(columns='status')
target = data['status']
# split data into train and test sets
xtrain, xtest, ytrain, ytest = train_test_split(predictors,target,test_size=0.30,random_state=1,stratify=target)
# get list of categorical predictor variables
xcatcols = list(predictors.select_dtypes('object').columns)

# get dummies for categorical variables
xtrain = pd.get_dummies(xtrain, columns=xcatcols, drop_first=True)
xtest = pd.get_dummies(xtest, columns=xcatcols, drop_first=True)

# get summary of independent variables
xtrain.info()

# quick sanity check on dummy variable encoding
xtrain.head()
xtrain.describe()

# build decision tree model
# create a metric function for use in the model evaluation process
def metrics_score(actual, predicted):
    print(classification_report(actual,predicted))
         
    cm = confusion_matrix(actual,predicted)
    plt.figure(figsize=(8,5))
         
    sns.heatmap(cm, annot=True, fmt='.2f', xticklabels=['Not Converted', 'Converted'], yticklabels=['Not Converted','Converted'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
         
    plt.show()

# set up the Decision Tree classifier
dtmodel = DecisionTreeClassifier(random_state=1)
dtmodel.fit(xtrain, ytrain)

# assess model performance on training set
predict_dttrain = dtmodel.predict(xtrain)
metrics_score(ytrain, predict_dttrain)

# Check model performance on the test dataset
predict_dttest = dtmodel.predict(xtest)
metrics_score(ytest, predict_dttest)

# Set up for GridSearch CV

# set up classifier, prioritize class 1
estimator = DecisionTreeClassifier(random_state=1, class_weight = {0:0.3, 1:0.7})

# grid of parameters
parameters = {
    "max_depth":np.arange(2,10),
    "criterion": ['gini','entropy'],
    "min_samples_split": [5, 10, 20, 25]
}

# scoring used to compare parameter combinations
scorer = metrics.make_scorer(recall_score, pos_label = 1)

# run grid search
grid_obj = GridSearchCV(estimator, parameters, cv=5, scoring=scorer)

# fit the grid search to the training data
grid_obj = grid_obj.fit(xtrain, ytrain)

# set classifier to best parameters
estimator = grid_obj.best_estimator_

# fit the best estimator to the data
estimator.fit(xtrain,ytrain)

# check performance of the tuned model on the training data
dttuned_train = estimator.predict(xtrain)
metrics_score(ytrain,dttuned_train)


# Check tuned model performance on the test dataset
dttuned_test = estimator.predict(xtest)
metrics_score(ytest, dttuned_test)

# visualize the decision tree
feature_names = list(xtrain.columns)
plt.figure(figsize=(20,10))
out = tree.plot_tree(
    estimator,
    feature_names=feature_names,
    filled=True,
    fontsize=9,
    node_ids=True,
    class_names=None
)
plt.show()

# visualize importance of features in the tuned tree
importances = estimator.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(8,8))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()

# Build random forest model
# set up classifier
rf_estimator = RandomForestClassifier(random_state=1)

# fit classifier to training data
rf_estimator.fit(xtrain,ytrain)
# check performance of the RF model on training data
rf_predict_train = rf_estimator.predict(xtrain)
metrics_score(ytrain,rf_predict_train)


# check performance on test data
rf_predict_test = rf_estimator.predict(xtest)
metrics_score(ytest,rf_predict_test)


# set up classifier
rftuned_estimator = RandomForestClassifier(class_weight={0:0.30, 1:0.70}, criterion = 'entropy', random_state=1)

# grid of potential parameters
params_rf =  {
    'n_estimators': [100, 110],
    'min_samples_leaf': [20, 25, 30],
    'max_features': [0.7, 0.9, 'auto'],
    'max_depth': [4,6,8,10]
}

# scoring used to compare parameter combinations, prioritize recall for class 1
scorer = metrics.make_scorer(recall_score, pos_label=1)

# run the grid search
grid_obj = GridSearchCV(rftuned_estimator, params_rf, scoring=scorer, cv=5)
grid_obj = grid_obj.fit(xtrain, ytrain)

# Set the classifier to the best combination of parameters
rftuned_estimator = grid_obj.best_estimator_
# fit tuned model to training data
rftuned_estimator.fit(xtrain, ytrain)
# check performance on training data
predict_rftuned_train = rftuned_estimator.predict(xtrain)
metrics_score(ytrain, predict_rftuned_train)

# check performance on test data
predict_rftuned_test = rftuned_estimator.predict(xtest)
metrics_score(ytest, predict_rftuned_test)

# check feature importances of the tuned RF model
importances = rftuned_estimator.feature_importances_
columns = xtrain.columns
importance_df = pd.DataFrame(importances, index=columns,
                            columns=['Importance']).sort_values(by='Importance',
                                                               ascending=False)
plt.figure(figsize=(8,8))
plt.title('Feature Importances')
sns.barplot(x=importance_df.Importance, y = importance_df.index)


# CONCLUSIONS AND RECOMMENDATIONS

# Conclusions:
# I have built multiple decision tree and random forest models, and ultimately determined that the time a lead spends on the website, the first interaction, profile completion, and age are the most important factors in predicting whether a lead will convert.
# Model performance between the tuned random forest and tuned decision tree models is comparable. Models may be able to be tuned further. However, if recall -- or limiting the number of lost customers -- is most important to the company, the tuned decision tree model may be preferred, as it is also more interpretable.
# Recommendations:
# As mentioned earlier in this report, based on the tuned decision tree, the company should prioritize and take actions to increase website interactions, take measures that encourage leads to complete at least 50-75% of their profiles, and make follow-up phone calls or texts to leads following initial website interactions. These factors tend to be most positively related to lead conversions to customers.
# While the occurrence of referred leads is low and only accounts for 2% of the leads in this dataset, the conversion rate for those with a referral is high (~65%). The company could consider methods for increasing referrals, either through incentives or website UI modifications to make the referral process easier, if these investments are not too costly.
# The time a lead spends on the website is also an important factor for conversion. Effort to increase engagement and interest in the website may yield higher conversion rates.
# Very few leads see any of the print or digital media advertisements, and these do not appear to yield many conversions. The company could consider saving resources by cutting down on these advertisements. ExtraaLearn could also consider investigating by what method most leads do discover the company, if not through these advertisements. For example, the company could ask whether users discovered ExtraaLearn through Google/browser search, as this may clarify lead entry points.




































