# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 10:40:59 2024

@author: Micaelan Valesky
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from scipy.stats import mannwhitneyu as mwu
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler

from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo

import statsmodels.api as sm
from statsmodels.formula.api import mixedlm



# read in data
isbedf = pd.read_csv(r"REDACTED", low_memory=False)
isbe5e = pd.read_csv(r"REDACTED")
yrbss = pd.read_csv(r"REDACTED")
isbe5e_24 = pd.read_csv(r"REDACTED")

# CPS crosswalk
cpsToRcdts = pd.read_csv(r"REDACTED")

# drop first column, it's an old index col
isbedf = isbedf.iloc[:,1:]

# set up some chi-specific dataframes
# district
chiRCDTS = '150162990250000'
chiDisdf = isbedf[isbedf['rcdts'] == chiRCDTS]

# schools
chidf = isbedf[(isbedf['district'] == 'City of Chicago SD 299') | (isbedf['district'] == 'Chicago Public Schools District 299')]

# MERGE 5E DATA
smallCross = cpsToRcdts.copy()
smallCross = smallCross.drop_duplicates(['rcdts'])
smallCross = smallCross.loc[:,'schoolID':'rcdts']
smallCrossDictRtS = dict(zip(smallCross['rcdts'],smallCross['schoolID']))
smallCrossDictStR = dict(zip(smallCross['schoolID'],smallCross['rcdts']))

isbe5e23 = isbe5e[isbe5e['year']== 2023]
isbe5eno23 = isbe5e[isbe5e['year'] != 2023]

isbe5e23['school_id'] = isbe5e23['rcdts'].replace(smallCrossDictRtS)
isbe5eno23['rcdts'] = isbe5eno23['school_id'].replace(smallCrossDictStR)

isbe5e = pd.concat([isbe5eno23,isbe5e23], ignore_index=False)
isbe5e['school_id'] = isbe5e['school_id'].astype(int)

# make column names something understandable
fiveEcross = pd.read_csv(r"REDACTED")
fiveEcross = dict(zip(fiveEcross['ThereYouGo'], fiveEcross['new_names']))
isbe5e = isbe5e.rename(columns=fiveEcross)

# add 2024 5e data to other isbe5e data (just one column of data for now)
merge5e24 = isbe5e_24.copy()
merge5e24 = merge5e24.reindex(columns=isbe5e.columns)
isbe5e = pd.concat([isbe5e, merge5e24], axis=0, ignore_index=True)

# merge 5e data with IRC data
chidf = pd.merge(left=chidf, right=isbe5e, left_on=['rcdts','year'],
                 right_on=['rcdts','year'])


# get pre/post-covid datasets
postcovidDF = chidf[(chidf['year'] == 2023) |
                  (chidf['year'] == 2022) |
                  (chidf['year'] == 2021)]

precovidDF = chidf[(chidf['year'] == 2020) |
                  (chidf['year'] == 2019) |
                  (chidf['year'] == 2018)]

# chidf_sm = chidf.copy()
# chidf_sm = chidf_sm.loc[:,]
chidf.replace('*', np.nan, inplace=True)




# TESTS FOR SIGNIFICANT EFFECTS BETWEEN VARS (not time or COVID)
# i.e., regressions
# vars of interest
# high_school_4year_graduation_rate_total
# chronic_absenteeism
# perc_8th_grade_passing_algebra_1
# perc_ela_proficiency
# perc_math_proficiency
# perc_science_proficiency
# pupil_teacher_ratio_elementary
# pupil_teacher_ratio_high_school
# sat_math_average_score
# sat_reading_average_score
# student_attendance_rate
# teacher_attendance_rate
# teacher_retention_rate
# perc_9thgradeontrack_total
# parent_supportiveness
# parent_involvement_in_school
# student_peer_relationships

# need to add postsecondary enrollment
#   need to look closer at what graduating classes are represented by each year
#   of isbe data, maybe on a two year lag?
#   data comes from national clearinghouse?
# be wary of iccb data, maybe reach out to iccb to see how different this is
# Jenn will send us ISBE disaggregated teacher evaluation data
# look at definitions of teacher retention

voiHS = ['high_school_4year_graduation_rate_total',
       'chronic_absenteeism',
       'perc_8th_grade_passing_algebra_1',
       'perc_ela_proficiency',
       'perc_math_proficiency',
       'perc_science_proficiency',
       # 'pupil_teacher_ratio_elementary',
       # 'pupil_teacher_ratio_high_school',
       'sat_math_average_score',
       'sat_reading_average_score',
       'student_attendance_rate',
       # 'teacher_attendace_rate',
       'teacher_retention_rate',
       'perc_9thgradeontrack_total',
       'parent_supportiveness',
       'parent_involvement_in_school',
       'student_peer_relationships',
       'emotional_health',
       'teacher-parent_trust',
       'student-teacher_trust']

voiMS = ['chronic_absenteeism',
         'perc_8th_grade_passing_algebra_1',
         'perc_ela_proficiency',
         'perc_math_proficiency',
         'perc_science_proficiency',
         'parent_supportiveness',
         'parent_involvement_in_school',
         'student_peer_relationships',
         'emotional_health',
         'teacher-parent_trust',
         'student-teacher_trust']


chidf_sm = chidf.copy()
chidf_sm = chidf_sm.loc[:,voiHS]
chidf_sm['rcdts'] = chidf['rcdts']
chidf_sm['school_type'] = chidf['school_type']
chidf_sm['year'] = chidf['year']


# TODO: run pairplots and corr for each school type in addition to
#   aggregate, then look at doing a factor analysis
chiHS = chidf[chidf['school_type'] == 'HIGH SCHOOL']
chiCH = chidf[chidf['school_type'] == 'CHARTER SCH']
chiES = chidf[chidf['school_type'] == 'ELEMENTARY']

#normalize HS variables for correlations
# did this as a sanity check, was correct, didn't need it
# chiHS_sm = chiHS.loc[:,voiHS]
# chiHS_norm = chiHS_sm.copy()
# for i in chiHS_sm.columns:
#     chiHS_norm[i] = (chiHS_norm[i] - chiHS_norm[i].min()) / (chiHS_norm[i].max() - chiHS_norm[i].min())


# for i in voiHS:
#     chiHS[i] = normalize(np.array(chiHS[i]))




voi = [voiHS, voiHS, voiHS, voiMS]
chiList = [chiHS, chiCH, chidf, chiES]
titles = ['High School', 'Charter School', 'All CPS', 'Elementary School']
def makePlots(dataList, voiList, cor_limit):
    for idx,elem in enumerate(dataList):

        smalldf = elem.loc[:,voiList[idx]]
        
        matrix = smalldf.corr()
        mask = (matrix >= cor_limit) | (matrix <= -abs(cor_limit))
        filt_matrix = matrix.where(mask, 0)
        
        sns.heatmap(matrix, annot=True)
        plt.title(titles[idx])
        plt.show()
        
        sns.heatmap(filt_matrix, annot=True)
        plt.title(titles[idx])
        plt.show()
        
        smatrix = stats.spearmanr(smalldf, nan_policy='omit')
        sns.heatmap(smatrix[0], annot=True)
        plt.title(titles[idx])
        plt.show()
        
        filtsmatrix = smatrix[0]
        filtsmatrix[~mask] = 0
        sns.heatmap(filtsmatrix, annot=True)
        plt.title(titles[idx])
        plt.show()
        
        pairplot = sns.pairplot(smalldf)
        for i in range(len(pairplot.axes)):
            for j in range(len(pairplot.axes)):
                if j > i:
                    pairplot.axes[i, j].set_visible(False)
        plt.title(titles[idx])
        plt.show()


# # SOME SORT OF EXPLORATORY FACTOR ANALYSIS
# # bartlett's test for sphericity
# chiHS_sm = chiHS.copy()
# chiHS_sm = chiHS_sm.loc[:,voiHS]
# #fill na with mean for tests
# chiHS_sm_filna = chiHS_sm.fillna(chiHS_sm.median())
# chi_sq_val, pval = calculate_bartlett_sphericity(chiHS_sm_filna)

# # kmo test
# kmo_all, kmo_model = calculate_kmo(chiHS_sm_filna)

# # tests look good, move forward with factor analysis
# fa = FactorAnalyzer()
# fa.fit(chiHS_sm)
# ev, va = fa.get_eigenvalues()

# plt.scatter(range(1,chiHS_sm.shape[1]+1),ev)
# plt.plot(range(1,chiHS_sm.shape[1]+1),ev)
# plt.title('Scree Plot')
# plt.xlabel('Factors')
# plt.ylabel('Eigenvalue')
# plt.grid()
# plt.show()

# # TODO: set analysis to an oblique rotation
# fa = FactorAnalyzer(n_factors=4)
# fa.fit(chiHS_sm)
# loadings = fa.loadings_
# loadings_df = pd.DataFrame(loadings, index=chiHS_sm.columns, columns=[f'Factor {i+1}' for i in range(loadings.shape[1])])


# let's do some regressions on the linear variables
# maybe do some multivariate, if I'm feelin' fancy

#plot chronic absenteeism over time, by school type
# chidf['year'] = chidf['year'].astype(str)
# chidf['school_type'] = chidf['school_type'].astype('category')

# avg_absenteeism = chidf.groupby(['year', 'school_type'])['chronic_absenteeism'].mean().reset_index()

# plt.figure(figsize=(12,6))
# sns.lineplot(data=avg_absenteeism, x='year', y='chronic_absenteeism', hue='school_type', marker='o')

# plt.title('Chronic Absenteeism Over Time, by School Type')
# plt.xlabel('Year')
# plt.ylabel('Mean Chronic Absenteeism (%)')
# plt.legend(title='School Type')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# plt.scatter(data=chidf[chidf['school_type'] == 'HIGH SCHOOL'], x='year', y='chronic_absenteeism')
# plt.show()

#define LME model formula
chiHS_lme = chiHS.copy()
chiHS_lme = chiHS_lme.loc[:,voiHS]
chiHS_lme['school_name'] = chiHS['school_name']
chiHS_lme['school_type'] = chiHS['school_type']
chiHS_lme['year'] = chiHS['year'].astype(str)
chiHS_lme = chiHS_lme.rename(columns={'teacher-parent_trust': 'teacher_parent_trust'})

voiHS = ['high_school_4year_graduation_rate_total',
       'chronic_absenteeism',
       'perc_8th_grade_passing_algebra_1',
       'perc_ela_proficiency',
       'perc_math_proficiency',
       'perc_science_proficiency',
       # 'pupil_teacher_ratio_elementary',
       # 'pupil_teacher_ratio_high_school',
       'sat_math_average_score',
       'sat_reading_average_score',
       'student_attendance_rate',
       # 'teacher_attendace_rate',
       'teacher_retention_rate',
       'perc_9thgradeontrack_total',
       'parent_supportiveness',
       'parent_involvement_in_school',
       'student_peer_relationships',
       'emotional_health',
       'teacher_parent_trust',
       'student-teacher_trust']

for i in voiHS:
    chiHS_lme[i] = chiHS_lme[i].astype(float)

chiHS_fl = chiHS_lme.copy()

# standardize the variables
object = StandardScaler()
CAmean = chiHS_lme['chronic_absenteeism'].mean()
CAstd = chiHS_lme['chronic_absenteeism'].std()
chiHS_lme['chronic_absenteeism'] = (chiHS_lme['chronic_absenteeism'] - CAmean) / CAstd
#chiHS_lme['chronic_absenteeism'] = (chiHS_lme['chronic_absenteeism'] - chiHS_lme['chronic_absenteeism'].mean()) / chiHS_lme['chronic_absenteeism'].std()#object.fit_transform(chiHS_lme['chronic_absenteeism'])
chiHS_lme['parent_involvement_in_school'] = (chiHS_lme['parent_involvement_in_school'] - chiHS_lme['parent_involvement_in_school'].mean()) / chiHS_lme['parent_involvement_in_school'].std() # object.fit_transform(chiHS_lme['parent_involvement_in_school'])


#formula = "sat_math_average_score ~ chronic_absenteeism + parent_supportiveness + teacher_parent_trust"
formula = "sat_math_average_score ~ chronic_absenteeism + parent_involvement_in_school"
# fit the model with random intercepts for the school and year
model = mixedlm(formula, chiHS_lme, groups=chiHS_lme['school_name'], re_formula='~year', missing='drop')
result = model.fit(reml=False, method='powell')
summary = result.summary()
print(summary)



residuals = result.resid
sns.histplot(residuals, kde=True)
plt.title('Residuals Distribution')
plt.show()

sm.qqplot(residuals, line='s')
plt.title('QQ Plot of Residuals')
plt.show()

# plot school/year trends
# two graphs per year, showing relationship between SAT and absenteeism
# and SAT and parental involvement, one point per high school
def makeScatters(years, df, xVar, yVar):
    for i in years:
        plt.figure(figsize=(12,12))
        df['year'] = df['year'].astype(int)
        filtdf = df[df['year'] == i]
        plt.scatter(data=filtdf, x=xVar, y=yVar)
        plt.title(f'{xVar} x {yVar}: SY{i}')
        plt.show()
        
yearList = [2024, 2023, 2022, 2021, 2020, 2019, 2018]
makeScatters(yearList, chiHS_lme, xVar='chronic_absenteeism', yVar='sat_math_average_score')

makeScatters(yearList, chiHS_lme, xVar='parent_involvement_in_school', yVar='sat_math_average_score')

# scatter plot with model fit
# Add the fitted values (predictions) to the original dataset
chiHS_lme['y_pred'] = result.fittedvalues

# Create a scatter plot of the data
plt.figure(figsize=(10, 6))
sns.scatterplot(x='chronic_absenteeism', y='sat_math_average_score', data=chiHS_lme, color='blue', alpha=0.6, label='Data')

# Plot the fitted line (the model's predicted values)
#sns.lineplot(x=chiHS_lme['chronic_absenteeism'], y=chiHS_lme['y_pred'], color='red', label='Fitted Line')
sns.scatterplot(x='chronic_absenteeism', y='y_pred', data=chiHS_lme, color='red', label='Predicted Data')
# Set labels and title
plt.xlabel('Chronic Absenteeism (Standardized)')
plt.ylabel('SAT Math Average Score')
plt.title('Scatter Plot with Linear Mixed Effects Model Fit')
plt.legend()

# Show the plot
plt.show()


# Plot predictions for fixed effects ######################################

# Chronic Absenteeism
# 1. Get the fixed effect coefficients
fixed_effects = result.fe_params

# 2. Construct the formula for the fixed effects
# Formula: sat_math_average_score = Intercept + (beta1 * chronic_absenteeism) + (beta2 * parent_involvement_in_school)
# Extracting the coefficients
intercept = fixed_effects['Intercept']
beta_chronic_absenteeism = fixed_effects['chronic_absenteeism']
beta_parent_involvement_in_school = fixed_effects['parent_involvement_in_school']

# 3. Create a range of values for chronic_absenteeism and parent_involvement_in_school to plot the formula
# Let's take some representative values of these predictors
x_range = np.linspace(chiHS_lme['chronic_absenteeism'].min(), chiHS_lme['chronic_absenteeism'].max(), 100)

# For simplicity, let's keep 'parent_involvement_in_school' constant at its mean value for the plot
mean_parent_involvement = chiHS_lme['parent_involvement_in_school'].mean()

# Calculate predicted SAT math scores based on the fixed effects formula
y_pred = intercept + beta_chronic_absenteeism * x_range + beta_parent_involvement_in_school * mean_parent_involvement

# 4. Plot the observed data and the fitted line
plt.figure(figsize=(10, 6))

# Scatter plot of the data
sns.scatterplot(x='chronic_absenteeism', y='sat_math_average_score', data=chiHS_lme, color='blue', alpha=0.6, label='Observed Data')
sns.scatterplot(x='chronic_absenteeism', y='y_pred', data=chiHS_lme, color='green', label='Predicted Data')

# Plot the predicted line from the formula
plt.plot(x_range, y_pred, color='red', linewidth=2, label='Fitted Line (Fixed Effects Only)')

# Add labels and title
plt.xlabel('Chronic Absenteeism (Standardized)')
plt.ylabel('SAT Math Average Score')
plt.title('Scatter Plot with Linear Model Fit (Fixed Effects Only)')

# Add a legend
plt.legend()

# Show the plot
plt.show()




# Parent Involvement
# 1. Get the fixed effect coefficients
fixed_effects = result.fe_params

# 2. Construct the formula for the fixed effects
# Formula: sat_math_average_score = Intercept + (beta1 * chronic_absenteeism) + (beta2 * parent_involvement_in_school)
# Extracting the coefficients
intercept = fixed_effects['Intercept']
beta_chronic_absenteeism = fixed_effects['chronic_absenteeism']
beta_parent_involvement_in_school = fixed_effects['parent_involvement_in_school']

# 3. Create a range of values for chronic_absenteeism and parent_involvement_in_school to plot the formula
# Let's take some representative values of these predictors
x_range = np.linspace(chiHS_lme['parent_involvement_in_school'].min(), chiHS_lme['parent_involvement_in_school'].max(), 100)

# For simplicity, let's keep 'chronic_absenteeism' constant at its mean value for the plot
mean_chronic = chiHS_lme['chronic_absenteeism'].mean()

# Calculate predicted SAT math scores based on the fixed effects formula
y_pred = intercept + beta_parent_involvement_in_school * x_range + beta_chronic_absenteeism * mean_chronic

# 4. Plot the observed data and the fitted line
plt.figure(figsize=(10, 6))

# Scatter plot of the data
sns.scatterplot(x='parent_involvement_in_school', y='sat_math_average_score', data=chiHS_lme, color='blue', alpha=0.6, label='Observed Data')
sns.scatterplot(x='parent_involvement_in_school', y='y_pred', data=chiHS_lme, color='green', label='Predicted Data')

# Plot the predicted line from the formula
plt.plot(x_range, y_pred, color='red', linewidth=2, label='Fitted Line (Fixed Effects Only)')

# Add labels and title
plt.xlabel('Parent Involvement (Standardized)')
plt.ylabel('SAT Math Average Score')
plt.title('Scatter Plot with Linear Model Fit (Fixed Effects Only)')

# Add a legend
plt.legend()

# Show the plot
plt.show()

# for mscore vars, recommended to subtract fifty and divide by 20

#############################################################################

# Model for chronic absenteeism, parent involvement, outcome = SAT reading avg score
formula = "sat_reading_average_score ~ chronic_absenteeism + parent_involvement_in_school"
chiHS_lme['year'] = chiHS_lme['year'].astype(str)

# fit the model with random intercepts for the school and year
model = mixedlm(formula, chiHS_lme, groups=chiHS_lme['school_name'], re_formula='~year', missing='drop')
result = model.fit(reml=False, method='powell')
summary = result.summary()
print(summary)
chiHS_lme['y_pred'] = result.fittedvalues


residuals = result.resid
sns.histplot(residuals, kde=True)
plt.title('Residuals Distribution')
plt.show()

sm.qqplot(residuals, line='s')
plt.title('QQ Plot of Residuals')
plt.show()

# makeScatters(yearList, chiHS, xVar='chronic_absenteeism', yVar='sat_reading_average_score')
# makeScatters(yearList, chiHS, xVar='parent_involvement_in_school', yVar='sat_reading_average_score')


# # plotting results


# # FIXED EFFECTS
# # 1. Get the fixed effect coefficients
# fixed_effects = result.fe_params
# fixed_effects_se = result.bse

# # Create a plot of the fixed effects with confidence intervals
# plt.figure(figsize=(8, 6))
# sns.barplot(x=fixed_effects[1:3].index, y=fixed_effects[1:3].values, 
#             yerr=1.96 * fixed_effects_se[1:3].values, 
#             capsize=5, color='skyblue')

# plt.title("Fixed Effects with 95% Confidence Intervals")
# plt.xlabel("Predictor Variables")
# plt.ylabel("Coefficient Value")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()



# # RANDOM EFFECTS FOR EACH SCHOOL GROUP AS A HIST

# random_effects = result.random_effects  # This will give you the random effects for each group

# # Convert random effects into a list for plotting
# random_effects_values = [effects[0] for effects in random_effects.values()]

# # Create a distribution plot of random intercepts
# plt.figure(figsize=(8, 6))
# sns.histplot(random_effects_values, kde=True, color='lightgreen')

# plt.title("Distribution of Random Intercepts")
# plt.xlabel("Random Intercept Value")
# plt.ylabel("Frequency")
# plt.tight_layout()
# plt.show()



# # SHOW RANDOM EFFECTS OF GROUP AND YEAR WITH HANDFUL OF SCHOOLS
# # Extract the random effects (intercepts and slopes) for each group
# random_effects = result.random_effects

# # Identify the year terms based on the fixed effects
# years = ['2021', '2022', '2023']  # Adjust according to the fixed effects you see

# # For illustration, let's take the first few groups
# import random
# groups_idx = random.sample(range(len(list(random_effects.keys()))), 5)
# selected_groups = list(random_effects.keys())  # Selecting top 5 groups for illustration
# selected_groups = [selected_groups[i] for i in groups_idx]


# # Assign a unique color to each group
# colors = plt.cm.get_cmap('tab10', len(selected_groups))

# # Plot predicted scores for selected groups over the years
# plt.figure(figsize=(8, 6))

# for idx, group in enumerate(selected_groups):
#     group_intercept = random_effects[group][0]  # The random intercept for this group
#     group_color = colors(idx)  # Get a unique color for this group
    
#     # Prepare the predicted scores for the group over the years
#     predicted_scores = []
#     for year in years:
#         # Fixed effect for the intercept + year effect (ensure you adjust based on the output)
#         year_effect = random_effects[group][f'year[T.{year}]'] #fixed_effects.get(f'year[T.{year}]', 0)  # Default to 0 if no effect for that year
#         predicted_score = fixed_effects['Intercept'] + group_intercept + year_effect
#         predicted_scores.append(predicted_score)
    
#     # Plot the group’s scores over time with lines connecting the points
#     plt.plot(years, predicted_scores, marker='o', label=f"Group {group}", color=group_color)

# # Customize the plot
# plt.title("Random Slopes for Groups over Time")
# plt.xlabel("Year")
# plt.ylabel("Predicted SAT Reading Score")

# # Move the legend to the bottom of the plot
# plt.legend(title="Groups", loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=3)

# # Adjust layout to avoid clipping the legend
# plt.tight_layout()
# plt.show()






# FIXED EFFECTS OVER PREDICTED RESULTS
# 2. Construct the formula for the fixed effects
# Formula: sat_math_average_score = Intercept + (beta1 * chronic_absenteeism) + (beta2 * parent_involvement_in_school)
# Extracting the coefficients
intercept = fixed_effects['Intercept']
beta_chronic_absenteeism = fixed_effects['chronic_absenteeism']
beta_parent_involvement_in_school = fixed_effects['parent_involvement_in_school']

# 3. Create a range of values for chronic_absenteeism and parent_involvement_in_school to plot the formula
# Let's take some representative values of these predictors
x_range = np.linspace(chiHS_lme['chronic_absenteeism'].min(), chiHS_lme['chronic_absenteeism'].max(), 100)

# For simplicity, let's keep 'parent_involvement_in_school' constant at its mean value for the plot
mean_parent_involvement = chiHS_lme['parent_involvement_in_school'].mean()

# Calculate predicted SAT math scores based on the fixed effects formula
y_pred = intercept + beta_chronic_absenteeism * x_range + beta_parent_involvement_in_school * mean_parent_involvement

# 4. Plot the observed data and the fitted line
plt.figure(figsize=(10, 6))

# Scatter plot of the data
sns.scatterplot(x='chronic_absenteeism', y='sat_reading_average_score', data=chiHS_lme, color='blue', alpha=0.6, label='Observed Data')
sns.scatterplot(x='chronic_absenteeism', y='y_pred', data=chiHS_lme, color='green', label='Predicted Data')

# Plot the predicted line from the formula
plt.plot(x_range, y_pred, color='red', linewidth=2, label='Fitted Line (Fixed Effects Only)')

# Add labels and title
plt.xlabel('Chronic Absenteeism (Standardized)')
plt.ylabel('SAT Reading Average Score')
plt.title('Scatter Plot with Linear Model Fit (Fixed Effects Only)')

# Add a legend
plt.legend()

# Show the plot
plt.show()


#############################################################################

# model for chronic absenteeism, parent involvement, outcome = graduation rate
formula = "high_school_4year_graduation_rate_total ~ chronic_absenteeism + parent_involvement_in_school"
chiHS_lme['year'] = chiHS_lme['year'].astype(str)

# fit the model with random intercepts for the school and year
model = mixedlm(formula, chiHS_lme, groups=chiHS_lme['school_name'], re_formula='~year', missing='drop')
result = model.fit(reml=False, method='powell')
summary = result.summary()
print(summary)
chiHS_lme['y_pred'] = result.fittedvalues

# TODO: apply model to grad rate, ela + math+ science proficiency
##############################################################################

# model for chronic absenteeism, parent involvement, outcome = perc ela proficiency
formula = "perc_ela_proficiency ~ chronic_absenteeism + parent_involvement_in_school"
chiHS_lme['year'] = chiHS_lme['year'].astype(str)

# fit the model with random intercepts for the school and year
model = mixedlm(formula, chiHS_lme, groups=chiHS_lme['school_name'], re_formula='~year', missing='drop')
result = model.fit(reml=False, method='powell')
summary = result.summary()
print(summary)

chiHS_lme['y_pred'] = result.fittedvalues

#############################################################################

# model for chronic absenteeism, parent involvement, outcome = perc math proficiency
formula = "perc_math_proficiency ~ chronic_absenteeism + parent_involvement_in_school"
chiHS_lme['year'] = chiHS_lme['year'].astype(str)

# fit the model with random intercepts for the school and year
model = mixedlm(formula, chiHS_lme, groups=chiHS_lme['school_name'], re_formula='~year', missing='drop')
result = model.fit(reml=False, method='powell')
summary = result.summary()
print(summary)
chiHS_lme['y_pred'] = result.fittedvalues


#############################################################################

# model for chronic absenteeism, parent involvement, outcome = perc science proficiency
formula = "perc_science_proficiency ~ chronic_absenteeism + parent_involvement_in_school"
chiHS_lme['year'] = chiHS_lme['year'].astype(str)

# fit the model with random intercepts for the school and year
model = mixedlm(formula, chiHS_lme, groups=chiHS_lme['school_name'], re_formula='~year', missing='drop')
result = model.fit(reml=False, method='powell')
summary = result.summary()
print(summary)
chiHS_lme['y_pred'] = result.fittedvalues




#####################################################################################

# ELEMENTARY SCHOOL PROFICIENCY MODELS
#define LME model formula
chiES_lme = chiES.copy()
chiES_lme = chiES_lme.loc[:,voiMS]
chiES_lme['school_name'] = chiES['school_name']
chiES_lme['school_type'] = chiES['school_type']
chiES_lme['year'] = chiES['year'].astype(str)

chiES_fl = chiES_lme.copy()

for i in voiMS:
    chiES_lme[i] = chiES_lme[i].astype(float)

chiES_lme = chiES_lme.rename(columns={'teacher-parent_trust': 'teacher_parent_trust'})

# standardize the variables
CAmean = chiES_lme['chronic_absenteeism'].mean()
CAstd = chiES_lme['chronic_absenteeism'].std()
chiES_lme['chronic_absenteeism'] = (chiES_lme['chronic_absenteeism'] - CAmean) / CAstd
chiES_lme['parent_involvement_in_school'] = (chiES_lme['parent_involvement_in_school'] - chiES_lme['parent_involvement_in_school'].mean()) / chiES_lme['parent_involvement_in_school'].std() # object.fit_transform(chiHS_lme['parent_involvement_in_school'])

# model with CA and parent involvement as predictors, ela prof as outcome
formula = "perc_ela_proficiency ~ chronic_absenteeism + parent_involvement_in_school"
# fit the model with random intercepts for the school and year
model = mixedlm(formula, chiES_lme, groups=chiES_lme['school_name'], re_formula='~year', missing='drop')
result = model.fit(reml=False, method='powell')
summary = result.summary()
print(summary)

residuals = result.resid
sns.histplot(residuals, kde=True)
plt.title('Residuals Distribution')
plt.show()

sm.qqplot(residuals, line='s')
plt.title('QQ Plot of Residuals')
plt.show()


# model with CA and parent involvement as predictors, math prof as outcome
formula = "perc_math_proficiency ~ chronic_absenteeism + parent_involvement_in_school"
# fit the model with random intercepts for the school and year
model = mixedlm(formula, chiES_lme, groups=chiES_lme['school_name'], re_formula='~year', missing='drop')
result = model.fit(reml=False, method='powell')
summary = result.summary()
print(summary)


# model with CA and parent involvement as predictors, science prof as outcome
formula = "perc_science_proficiency ~ chronic_absenteeism + parent_involvement_in_school"
# fit the model with random intercepts for the school and year
model = mixedlm(formula, chiES_lme, groups=chiES_lme['school_name'], re_formula='~year', missing='drop')
result = model.fit(reml=False, method='powell')
summary = result.summary()
print(summary)




















