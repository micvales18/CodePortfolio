# -*- coding: utf-8 -*-
"""
Created on Sat May  9 16:10:19 2026

@author: mvale
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# for building recommendation systems
from surprise.prediction_algorithms.matrix_factorization import SVD
from collections import defaultdict
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

# cross validation
from surprise.model_selection import KFold

import warnings
warnings.filterwarnings('ignore')

# read in data
df = pd.read_csv(r"REDACTED")
# check what the dataframe looks like
df.head()

# There are no column headers, so we need to add them
df.columns = ['userId', 'prodId', 'rating', 'timestamp']

# drop 'timestamp', as we will not be using this column
df = df.drop('timestamp', axis=1)

# make a copy of the df
dfCopy = df.copy(deep=True)

# Get the column containing the users
users = df.userId

# Create a dictionary from users to their number of ratings
ratings_count = dict()

for user in users:

    # If we already have the user, just add 1 to their rating count
    if user in ratings_count:        
        ratings_count[user] += 1
  
    # Otherwise, set their rating count to 1
    else:
        ratings_count[user] = 1
        
# We want our users to have at least 50 ratings to be considered
RATINGS_CUTOFF = 50

remove_users = []

for user, num_ratings in ratings_count.items():
    if num_ratings < RATINGS_CUTOFF:
        remove_users.append(user)

df = df.loc[ ~ df.userId.isin(remove_users)]

# Get the column containing the products
prods = df.prodId

# Create a dictionary from products to their number of ratings
ratings_count = dict()

for prod in prods:
    
    # If we already have the product, just add 1 to its rating count
    if prod in ratings_count:
        ratings_count[prod] += 1
    
    # Otherwise, set their rating count to 1
    else:
        ratings_count[prod] = 1    
# We want our item to have at least 5 ratings to be considered
RATINGS_CUTOFF = 5

remove_users = []

for user, num_ratings in ratings_count.items():
    if num_ratings < RATINGS_CUTOFF:
        remove_users.append(user)

df_final = df.loc[~ df.prodId.isin(remove_users)]
# Print a few rows of the imported dataset
df_final.head()


# Check the number of rows and columns and provide observations
df_final.shape

# Check Data types and provide observations
df_final.info()

# Check for missing values present and provide observations
df_final.isnull().values.any()

# Summary statistics of 'rating' variable and provide observations
df_final['rating'].describe()

# Create the bar plot and provide observations
plt.figure(figsize = (12,12))

df_final['rating'].value_counts().plot(kind='bar')
plt.title("Distribution of Ratings", fontsize=10)
plt.xlabel('Ratings', fontsize=10)
plt.ylabel('Number of Ratings', fontsize=10)
plt.show()

print('Number of entries in the final dataset: ', len(df_final))
print('Number of unique users in the final dataset: ', df_final['userId'].nunique())
print('Number of unique items in the final dataset: ', df_final['prodId'].nunique())

# Top 10 users based on the number of ratings
mostRatings = df_final.groupby('userId').size().sort_values(ascending=False)
mostRatings[:10]

# Calculate the average rating for each product
avgRating = df_final.groupby('prodId')['rating'].mean()

# Calculate the count of ratings for each product
ctRating = df_final.groupby('prodId')['rating'].count()

# Create a dataframe with calculated average and count of ratings
ratingDf = pd.DataFrame({'avgRating':avgRating, 'ctRating':ctRating})

# Sort the dataframe by average of ratings in the descending order
ratingDf = ratingDf.sort_values(by=['avgRating', 'ctRating'], ascending=False)

# See the first five records of the "final_rating" dataset
ratingDf.head()

# Defining a function to get the top n products based on the highest average rating and minimum interactions
def top_n_prods(data, n, min_interactions):
    
    # get products with minimum number of interactions
    recs = data[data['ctRating'] > min_interactions]
    
    # Sorting values with respect to average rating
    recs = recs.sort_values(by='avgRating', ascending=False)

    return recs[:n]


# Finding products with minimum number of interactions
prodMin10 = top_n_prods(ratingDf, 10, 10)
    
# Sorting values with respect to average rating 
# this has already been written into the top_n_prods function
prodMin10

# Find top 5 products with 50 minimum interactions
top5_min50 = top_n_prods(ratingDf, 5, 50)

# Find top 5 products with 50 minimum interactions
top5_min100 = top_n_prods(ratingDf, 5, 100)
top5_min100

# To compute the accuracy of models
from surprise import accuracy

# Class is used to parse a file containing ratings, data should be in structure - user ; item ; rating
from surprise.reader import Reader

# Class for loading datasets
from surprise.dataset import Dataset

# For tuning model hyperparameters
from surprise.model_selection import GridSearchCV

# For splitting the rating data in train and test datasets
from surprise.model_selection import train_test_split

# For implementing similarity-based recommendation system
from surprise.prediction_algorithms.knns import KNNBasic

# For implementing matrix factorization based recommendation system
from surprise.prediction_algorithms.matrix_factorization import SVD

# for implementing K-Fold cross-validation
from surprise.model_selection import KFold

# For implementing clustering-based recommendation system
from surprise import CoClustering

def precision_recall_at_k(model, k = 10, threshold = 3.5):
    """Return precision and recall at k metrics for each user"""

    # First map the predictions to each user
    user_est_true = defaultdict(list)
    
    # Making predictions on the test data
    predictions = model.test(testset)
    
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key = lambda x: x[0], reverse = True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        # Precision@K: Proportion of recommended items that are relevant
        # When n_rec_k is 0, Precision is undefined. Therefore, we are setting Precision to 0 when n_rec_k is 0

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

        # Recall@K: Proportion of relevant items that are recommended
        # When n_rel is 0, Recall is undefined. Therefore, we are setting Recall to 0 when n_rel is 0

        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0
    
    # Mean of all the predicted precisions are calculated.
    precision = round((sum(prec for prec in precisions.values()) / len(precisions)), 3)
    
    # Mean of all the predicted recalls are calculated.
    recall = round((sum(rec for rec in recalls.values()) / len(recalls)), 3)
    
    accuracy.rmse(predictions)
    
    print('Precision: ', precision) # Command to print the overall precision
    
    print('Recall: ', recall) # Command to print the overall recall
    
    print('F_1 score: ', round((2*precision*recall)/(precision+recall), 3))
    
# Instantiating Reader scale with expected rating scale
reader = Reader(rating_scale=(1,5))

# Loading the rating dataset
data = Dataset.load_from_df(df_final[['userId','prodId','rating']], reader)

# Splitting the data into train and test datasets
trainset, testset = train_test_split(data, test_size=0.3, random_state=1)

# Declaring the similarity options
sim_options = {'name':'cosine',
              'user_based':True}

# Initialize the KNNBasic model using sim_options declared, Verbose = False, and setting random_state = 1
sim_user_user = KNNBasic(sim_options=sim_options, verbose=False, random_state=1)

# Fit the model on the training data
sim_user_user.fit(trainset)

# Let us compute precision@k, recall@k, and f_1 score using the precision_recall_at_k function defined above
precision_recall_at_k(sim_user_user)

# Predicting rating for a sample user with an interacted product
sim_user_user.predict('A3LDPF5FMB782Z', '1400501466', r_ui=5, verbose=True)

# Find unique user_id where prod_id is not equal to "1400501466"
df_final[df_final.prodId != '1400501466'].userId.unique()

# Predicting rating for a sample user with a non interacted product
sim_user_user.predict('A34BZM6S9L7QI4','1400501466', verbose=True)

# Setting up parameter grid to tune the hyperparameters
param_grid = {'k':[10,20,30], 'min_k': [3,6,9],
             'sim_options': {'name': ['cosine','pearson','pearson_baseline'],
                            'user_based': [True], 'min_support':[2,4]}}

# Performing 3-fold cross-validation to tune the hyperparameters
gs = GridSearchCV(KNNBasic, param_grid, measures=['rmse'], cv=3, n_jobs=-1)

# Fitting the data
gs.fit(data)

# Best RMSE score
print(gs.best_score['rmse'])

# Combination of parameters that gave the best RMSE score
print(gs.best_params['rmse'])

# Using the optimal similarity measure for user-user based collaborative filtering
sim_options = {'name':'cosine',
              'user_based':True, 'min_support':2}

# Creating an instance of KNNBasic with optimal hyperparameter values
sim_user_user_optimized = KNNBasic(sim_options=sim_options, k=20, min_k=3, random_state=1, verbose=False)

# Training the algorithm on the trainset
sim_user_user_optimized.fit(trainset)

# Let us compute precision@k and recall@k also with k =10
# using k=20, as this is the value identified with grid search
precision_recall_at_k(sim_user_user_optimized)

# Use sim_user_user_optimized model to recommend for userId "A3LDPF5FMB782Z" and productId 1400501466
sim_user_user_optimized.predict('A3LDPF5FMB782Z','1400501466', verbose=True)

# Use sim_user_user_optimized model to recommend for userId "A34BZM6S9L7QI4" and productId "1400501466"
sim_user_user_optimized.predict('A34BZM6S9L7QI4','1400501466', verbose=True)

# 0 is the inner id of the above user
sim_user_user_optimized.get_neighbors(0,5)

def get_recommendations(data, user_id, top_n, algo):
    
    # Creating an empty list to store the recommended product ids
    recommendations = []
    
    # Creating an user item interactions matrix 
    user_item_interactions_matrix = data.pivot(index = 'userId', columns = 'prodId', values = 'rating')
    
    # Extracting those product ids which the user_id has not interacted yet
    non_interacted_products = user_item_interactions_matrix.loc[user_id][user_item_interactions_matrix.loc[user_id].isnull()].index.tolist()
    
    # Looping through each of the product ids which user_id has not interacted yet
    for item_id in non_interacted_products:
        
        # Predicting the ratings for those non interacted product ids by this user
        est = algo.predict(user_id, item_id).est
        
        # Appending the predicted ratings
        recommendations.append((item_id, est))

    # Sorting the predicted ratings in descending order
    recommendations.sort(key = lambda x: x[1], reverse = True)

    return recommendations[:top_n] # Returing top n highest predicted rating

# Making top 5 recommendations for user_id "A3LDPF5FMB782Z" with a similarity-based recommendation engine
recs = get_recommendations(df_final, 'A3LDPF5FMB782Z', 5, sim_user_user_optimized)
# Building the dataframe for above recommendations with columns "prod_id" and "predicted_ratings"
pd.DataFrame(recs, columns=['prodID', 'predictedRatings'])

# Declaring the similarity options
sim_options = {'name':'cosine',
              'user_based':False}

# KNN algorithm is used to find desired similar items. Use random_state=1
sim_item_item = KNNBasic(sim_options=sim_options, random_state=1, verbose=False)

# Train the algorithm on the trainset, and predict ratings for the test set
sim_item_item.fit(trainset)

# Let us compute precision@k, recall@k, and f_1 score with k = 10
precision_recall_at_k(sim_item_item)

# Predicting rating for a sample user with an interacted product
sim_item_item.predict('A3LDPF5FMB782Z', '1400501466', verbose=True)

# Predicting rating for a sample user with a non interacted product
sim_item_item.predict('A34BZM6S9L7QI4', '1400501466', verbose=True)


# Setting up parameter grid to tune the hyperparameters
param_grid = {'k': [10,20,30], 'min_k':[3,6,9],
             'sim_options': {'name': ['cosine','msd'],
                            'user_based':[False]}}

# Performing 3-fold cross validation to tune the hyperparameters
gs = GridSearchCV(KNNBasic, param_grid, measures=['rmse'], cv=3, n_jobs=-1)

# Fitting the data
gs.fit(data)

# Find the best RMSE score
print(gs.best_score['rmse'])

# Find the combination of parameters that gave the best RMSE score
print(gs.best_params['rmse'])

# Using the optimal similarity measure for item-item based collaborative filtering
sim_options = {'name':'msd',
              'user_based':False}

# Creating an instance of KNNBasic with optimal hyperparameter values
sim_item_item_optimized = KNNBasic(sim_options=sim_options, k=20, min_k=6, random_state=1, verbose=False)

# Training the algorithm on the trainset
sim_item_item_optimized.fit(trainset)

# Let us compute precision@k and recall@k, f1_score and RMSE
precision_recall_at_k(sim_item_item_optimized)

# Use sim_item_item_optimized model to recommend for userId "A3LDPF5FMB782Z" and productId "1400501466"
sim_item_item_optimized.predict('A3LDPF5FMB782Z', '1400501466', r_ui=5, verbose=True)

# Use sim_item_item_optimized model to recommend for userId "A34BZM6S9L7QI4" and productId "1400501466"
sim_item_item_optimized.predict('A34BZM6S9L7QI4', '1400501466', verbose=True)

sim_item_item_optimized.get_neighbors(0,k=5)

# Making top 5 recommendations for user_id A1A5KUIIIHFF4U with similarity-based recommendation engine.
recs = get_recommendations(df_final, 'A1A5KUIIIHFF4U', 5, sim_item_item_optimized)
# Building the dataframe for above recommendations with columns "prod_id" and "predicted_ratings"
pd.DataFrame(recs, columns=['prodID', 'predictedRatings'])

# Using SVD matrix factorization. Use random_state = 1
svd = SVD(random_state=1)

# Training the algorithm on the trainset
svd.fit(trainset)

# Use the function precision_recall_at_k to compute precision@k, recall@k, F1-Score, and RMSE
precision_recall_at_k(svd)

# Making prediction
svd.predict('A3LDPF5FMB782Z', '1400501466', r_ui=5, verbose=True)

# Making prediction
svd.predict('A34BZM6S9L7QI4', '1400501466', verbose=True)

# Set the parameter space to tune
param_grid = {'n_epochs':[10,20,30], 'lr_all':[0.001, 0.005, 0.01],
             'reg_all': [0.2, 0.4, 0.6]}

# Performing 3-fold gridsearch cross-validation
gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3, n_jobs=-1)

# Fitting data
gs.fit(data)

# Best RMSE score
print(gs.best_score['rmse'])

# Combination of parameters that gave the best RMSE score
print(gs.best_params['rmse'])

# Build the optimized SVD model using optimal hyperparameter search. Use random_state=1
svd_optimized = SVD(n_epochs=20, lr_all=0.01, reg_all=0.2, random_state=1)

# Train the algorithm on the trainset
svd_optimized.fit(trainset)

# Use the function precision_recall_at_k to compute precision@k, recall@k, F1-Score, and RMSE
precision_recall_at_k(svd_optimized)

# Use svd_algo_optimized model to recommend for userId "A3LDPF5FMB782Z" and productId "1400501466"
svd_optimized.predict('A3LDPF5FMB782Z', '1400501466', r_ui=5, verbose=True)

# Use svd_algo_optimized model to recommend for userId "A34BZM6S9L7QI4" and productId "1400501466"
svd_optimized.predict('A34BZM6S9L7QI4', '1400501466', verbose=True)

# CONCLUSIONS AND RECOMMENDATIONS

# I have built recommendation systems using four different algorithms:

# rank-based, utilizing averages
# user-user similarity-based collaborative filtering
# item-item similarity-based collaborative filtering
# model-based matrix factorization collaborative filtering
# The collaborative filtering approaches were all tuned in an effort to optimized hyperperameters. Tuned models for user-user and item-item models performed better than their respective baseline models, and the tuned matrix factorization model performed similarly between tuned and baseline.

# Proposal for final solution design:

# Both the tuned user-user and item-item models perform well, with RMSE values of 0.9881 and 0.9804, respectively, and F1 scores of 0.825 and 0.816, respectively.
# Given that the item-item model more closely predicted user A3Ldx's rating of product 1400501466, we can preliminarily select the item-item model for our design. However, further spot-checking of more users and ratings, along with more attempts at tuning hyperperameters, may provide greater evidence for either the user-user or item-item model.










