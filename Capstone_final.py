#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Alpha = 0.005 = 5e-3

import numpy as np # For most of my data analysis I'll use numpy 
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, roc_curve, auc

from sklearn.utils import resample
from scipy.stats import chi2_contingency
from scipy.stats import mode
import random 
from scipy.stats import pearsonr
#-----RNG SEEDING-----#
random.seed(11447755) # just excluding "N"

data = np.genfromtxt('rmpCapstoneNum.csv',delimiter=',') 
# The cols are delimited by commas

#-----Preliminary Nan Removal-----

# To execute nan removal, I first try simple row-wise nan removal, as follows:
row_nan_ct = 0
for var in data: # Iterates over first axis
    if np.isnan(var).sum() > 0:
        row_nan_ct += 1

# I may now count how many observations I must remove on this basis. From the console:
print('Number of observations to remove in row-wise nan removal: ', row_nan_ct, '\n')

# This seems very high. In fact, I can compute the percentage of remaining data. 
# From the console:
print('Percentage of data remaining after row-wise nan removal: ', 
      round((1-row_nan_ct/len(data))*100, 1), '\n')
# I retain only about 13.5% of the observations! Surely I can do better than this

thresholds = range(1, 51) # 1 through 50
# 50 is an otherwise arbitrary choice, intended to illustrate the 
# long-run change in percentage share

num_ratings = data[np.isfinite(data[:,2]),2] # Col-wise removal
# Logic: the first (row) index is a boolean mask by which are selected only the
# finite, so non-nan, observations (rows) from col 2, then the second (col) index 
# indicates to slice out col 2 for these selected rows

perc_ratings_left = [num_ratings[num_ratings >= t].size / np.size(num_ratings) * 100 
                     for t in thresholds]

# Now I plot thresholds vs. percentage of ratings left:
plt.plot(thresholds, perc_ratings_left, marker='o')
plt.title('Percentage of ratings left vs. Threshold values')
plt.xlabel('Threshold values')
plt.ylabel('Percentage of ratings left')
plt.grid(True)
plt.show()
    
data = pd.DataFrame(data) # I'll turn it back into numpy right after this analysis

nan_cts = {} # Initialize a dictionary, to then turn into a dataframe
for col_index in range(data.shape[1]): 
# data.shape returns the tuple (number of data's rows, number of data's cols)
    if col_index == 2:
        continue
    col_is_nan = data.iloc[:, col_index].isna()
    # Check if nan for every value in the col of col_index, then put in array
    num_ratings_valid = data.iloc[:, 2] >= 10 # Review
    nan_ct = (col_is_nan & num_ratings_valid).sum() # Review
    # "&" does element-wise boolean AND comparison
    nan_cts[col_index] = nan_ct

nan_cts_df = pd.DataFrame.from_dict(nan_cts, orient='index', 
                                    columns=['Nan count with non-nan number of ratings'])
nan_cts_df.index.name = 'Column number (0-based indexing)'
# Turn nan_cts into a dataframe: orient='index' arranges the keys by row
print(nan_cts_df, '\n')

data = data.to_numpy()

# Redefine data based on my threshold:
data = data[data[:,2]>=10]

# Print the number of observations left:
print('Number of observations left: ',len(data),'\n')

#-----1-----

print('Sum of number of nans in male? and female? cols: ', 
      np.isnan(data[:,6]).sum() + np.isnan(data[:,7]).sum(), '\n')
# Sum is 0 for each, so no nans to remove

both_male_and_female = (data[:,6]==1) & (data[:,7]==1)
print('Count of both male and female: ', 
      np.sum(both_male_and_female), '\n')

neither_male_nor_female = (data[:,6]==0) & (data[:,7]==0)
print('Count of neither male nor female: ', 
      np.sum(neither_male_nor_female), '\n')

male_xor_female_data = data[~both_male_and_female & ~neither_male_nor_female] 

male_ratings = male_xor_female_data[male_xor_female_data[:, 6] == 1, 0]
female_ratings = male_xor_female_data[male_xor_female_data[:, 7] == 1, 0]

print('Number of male ratings left: ', len(male_ratings), '\n')
print('Number of female ratings left: ', len(female_ratings), '\n')

u_stat, p_val = stats.mannwhitneyu(male_ratings, female_ratings, alternative='greater') # First value in this function is the statistic, other is the corresponding p-value
print('Question 1 p-value: ', p_val,'\n')

plt.hist(male_ratings, bins=20, alpha=0.5, label='Male Ratings', density=True, color='blue')
plt.hist(female_ratings, bins=20, alpha=0.5, label='Female Ratings', density=True, color='red')
plt.title('Distribution of Male and Female Ratings')
plt.xlabel('Rating')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()

#-----2-----
avg_rating = data[:, 0]
num_ratings = data[:, 2]
valid_mask = np.isfinite(avg_rating) & np.isfinite(num_ratings)
avg_rating = avg_rating[valid_mask]
num_ratings = num_ratings[valid_mask]
exp_cutoff = np.median(num_ratings)
print('exp_cutoff', exp_cutoff, '\n')
low_exp = avg_rating[num_ratings <= exp_cutoff]
high_exp = avg_rating[num_ratings > exp_cutoff]
print('low_exp size: ',np.size(low_exp),'\n')
print('high size: ',np.size(high_exp),'\n')

u_stat, p_val = stats.mannwhitneyu(low_exp, high_exp,alternative='less')
print("Q2 p-value: ", p_val, '\n')
#-----3-----
avg_difficulty = data[:, 1]
difficulty_cutoff = np.median(avg_difficulty)
low_difficulty = avg_rating[avg_difficulty <= difficulty_cutoff]
high_difficulty = avg_rating[avg_difficulty > difficulty_cutoff]
print('low_difficulty: ',np.size(low_difficulty))
plt.hist(low_difficulty, bins=20, alpha=0.5, label='Low Difficulty', density=True, color='blue')
plt.hist(high_difficulty, bins=20, alpha=0.5, label='High Difficulty', density=True, color='green')
plt.title('Distribution of Average Ratings by Difficulty Level')
plt.xlabel('Average Rating')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()
u_stat, p_val = stats.mannwhitneyu(low_difficulty, high_difficulty, alternative='greater') # Want this
print("Q3: Mann-Whitney U Test: U-Statistic", u_stat, "P-Value=", p_val)

#-----4-----
num_online_ratings = data[:,5]
print('Median of num_online_ratings: ', np.median(num_online_ratings)) # 0!
print('Count of num_online_ratings > 0: ', np.sum(num_online_ratings > 0))
print('95th percentile of num_online_ratings: ',np.percentile(num_online_ratings, 95))
print('Count of num_online_ratings > 5 (95th percentile): ',np.sum((num_online_ratings > 5)))
exp_online_cutoff = 5
plt.hist(num_online_ratings, bins=30, color='blue', edgecolor='black') # Make the bin centers whole #'s
plt.xlabel("Number of online ratings")
plt.ylabel("Frequency")
plt.title("Number of online ratings vs. Frequency")
plt.grid(True)
plt.show()
low_exp_online = avg_rating[num_online_ratings<=exp_online_cutoff]
high_exp_online = avg_rating[num_online_ratings>exp_online_cutoff]
n_bootstrap = 1000 # Number of bootstrap iterations
bootstrap_results = []  # To store p-values from all the bootstrap iterations
for _ in range(n_bootstrap):
    # Sample with replacement from the high_exp_online to match size of low_exp_online
    bootstrapped_high_exp_online = np.random.choice(high_exp_online, size=np.size(low_exp_online), replace=True)
    u_stat, p_val = stats.mannwhitneyu(low_exp_online, bootstrapped_high_exp_online, alternative='two-sided') # Include IN for-loop, and want 2 sided
    bootstrap_results.append(p_val)
mean_p_value = np.mean(bootstrap_results) # Get the mean
print(f"Q4: Bootstrap Mean P-Value: {mean_p_value}")
plt.hist(low_exp_online, bins=20, alpha=0.5, label='Low Experience in Online Teaching', density=True, color='blue')
plt.hist(high_exp_online, bins=20, alpha=0.5, label='High Experience in Online Teaching', density=True, color='green')
plt.title('Distribution of Average Ratings by Experience in Online Teaching')
plt.xlabel('Average Rating')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()

#-----5-----
prop_retake_class = data[np.isfinite(data[:,4]),4]
num_ratings = data[np.isfinite(data[:,4]), 2]
avg_rating = data[np.isfinite(data[:,4]), 0]
#CORRELATION
pearson_corr, _= pearsonr(prop_retake_class, avg_rating)
print(f"Pearson Correlation (r): {pearson_corr:.3f}")
#PLOTTING
plt.figure(figsize=(8, 6))
plt.scatter(avg_rating, prop_retake_class, alpha=0.6, edgecolor='k')
plt.title("Scatter Plot of Proportion Retake Class vs Average Rating", fontsize=14)
plt.xlabel("Average Rating", fontsize=12)
plt.ylabel("Proportion Retake Class", fontsize=12)

plt.grid(alpha=0.4)
plt.show()
#-----6-----
hot = data[data[:,3]==1, 0]
not_hot = data[data[:,3]==0, 0]
u_stat, p_value = stats.mannwhitneyu(hot, not_hot, alternative='greater')
print(f"Q6: Mann-Whitney U Statistic: {u_stat}")
print(f"Q6: p-value: {p_value}")
plt.figure(figsize=(10, 6))
plt.hist(hot, bins=30, alpha=0.6, label="Hot Professors", density=True, edgecolor='k')
plt.hist(not_hot, bins=30, alpha=0.6, label="Not Hot Professors", density=True, edgecolor='k')
plt.title("Distribution of Ratings: Hot vs. Not Hot Professors", fontsize=14)
plt.xlabel("Ratings", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.legend(title="Group")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()

#-----7-----
# Explain why column-wise removal could fail (though here specifically it wouldn't), and so do row-wise removal
# Col-wise removal would fail if you had a different # nans among predictors
# and outcome cols or if you had the same # but the nans were in different places, 
# so row-wise removal is best for this reason.

# Plot average rating vs average difficulty

avg_rating = data[:,0]

#data_7_bool = np.isfinite(data[:,0]) & np.isfinite(data[:,1])

difficulties = avg_difficulty.reshape(-1,1)

ratings = avg_rating.reshape(-1,1)

model_7 = LinearRegression()

model_7.fit(difficulties, ratings) # Difficulty prediciting average rating

y_pred = model_7.predict(difficulties)

print("Arguments for ratings vs. difficulties linear regression: ","Slope: ",
      model_7.coef_[0][0], "Intercept: ",model_7.intercept_[0])

r2 = r2_score(ratings, y_pred)
print("R^2: ",r2)

rmse = root_mean_squared_error(ratings, y_pred)
print("RMSE: ",rmse)

# Plot ratings vs. difficulties - put this before?

plt.scatter(difficulties, ratings, alpha=0.2, label='Data Points')
plt.plot(difficulties, y_pred, color='red', label='Regression line')
plt.title('Ratings vs. Difficulties Linear Regression')
plt.xlabel('Average Difficulty')
plt.ylabel('Average Rating')
plt.legend()
plt.grid(True)
plt.show()

#-----8-----
# Just want to accurately *predict* rating, don't care for the reasons why
# Vars to use: 2, 5, 4 (but 4 is boolean), maybe 6 (# online ratings)
# Try doing a train-test split


# "Comment on how this model compares to the “difficulty only” model and on individual betas." 
# Built-in fcts should provide for individual betas.
# The R^2 
# NOTE: the individual betas in a multiple regression are calculated on the basis
# of their independent contribution to the outcome - how, exactly?

#Row-wise nan removal
columns_to_check = [1, 2, 3, 4, 6, 0]
data_cleaned = data[~np.isnan(data[:, columns_to_check]).any(axis=1)]
#TRAIN-TEST SPLIT FOR 5 PREDICTORS:
X = data_cleaned[:, [1, 2, 3, 4, 6]]
y = data_cleaned[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)
model = LinearRegression()
model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_rmse = root_mean_squared_error(y_train, y_train_pred)
test_rmse = root_mean_squared_error(y_test, y_test_pred)

print('\n\n')
print('5 PREDICTORS')
# Print results
print(f"Train R^2: {train_r2}")
print(f"Train RMSE: {train_rmse}")
print(f"Test R^2: {test_r2}")
print(f"Test RMSE: {test_rmse}")


cols = ['Average Difficulty', 'Num Ratings', 'Received a Pepper', 'Proportion Retake', 'Male']
col_indices = [1, 2, 3, 4, 6] # Review
data_8_bool = np.isfinite(data[:,col_indices]).all(axis=1) # Review
data_8_cleaned = data[data_8_bool][:,col_indices] # Review
corr_matrix = np.corrcoef(data_8_cleaned, rowvar=False) # Review
print('Correlation matrix:')
print(corr_matrix)
plt.imshow(corr_matrix, cmap='coolwarm', interpolation='nearest') # Review
plt.colorbar(label='Correlation Coefficient')
plt.xticks(ticks=np.arange(len(cols)), labels=cols, rotation=45, ha='right') # Review
plt.yticks(ticks=np.arange(len(cols)), labels=cols) # Review
plt.title("Correlation Matrix Heatmap (Cleaned Data with isfinite)")
plt.show()

columns_to_check = [2, 4, 6, 0]
data_cleaned = data[~np.isnan(data[:, columns_to_check]).any(axis=1)]
#TRAIN-TEST SPLIT FOR 3 PREDICTORS:
X = data_cleaned[:, [2, 4, 6]]
y = data_cleaned[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)
model = LinearRegression()
model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_rmse = root_mean_squared_error(y_train, y_train_pred)
test_rmse = root_mean_squared_error(y_test, y_test_pred)

print('\n\n')
print('3 PREDICTORS')
# Print results
print(f"Train R^2: {train_r2}")
print(f"Train RMSE: {train_rmse}")
print(f"Test R^2: {test_r2}")
print(f"Test RMSE: {test_rmse}")

print('\n''\n')

#-----9-----
X = data[:, 0].reshape(-1, 1)
y = data[:, 3]
combined = np.hstack((X, y.reshape(-1, 1)))
cleaned_data = combined[~np.isnan(combined).any(axis=1)]
X = cleaned_data[:, 0].reshape(-1, 1)
y = cleaned_data[:, 1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)
model = LogisticRegression(random_state=69, max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilities for th positive clas

# Calculate quality metrics
auc = roc_auc_score(y_test, y_pred_proba)
classification_metrics = classification_report(y_test, y_pred)

print(f"AU(ROC): {auc:.3f}")
print("Classification Report:")
print(classification_metrics)

print('\n')
#-----10-----
X = data[:, [0, 1, 2, 4, 5, 6, 7]]
y = data[:, 3]  
# Handle missing values (row-wise removal)
combined = np.hstack((X, y.reshape(-1, 1)))
data_cleaned = combined[~pd.isnull(combined).any(axis=1)]
X = data_cleaned[:, :-1]
y = data_cleaned[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=455)
logistic_model = LogisticRegression(max_iter=1000, random_state=42)
logistic_model.fit(X_train, y_train)
y_pred = logistic_model.predict(X_test)
y_pred_proba = logistic_model.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_pred_proba)
classification_metrics = classification_report(y_test, y_pred)
#NOW: roc curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {auc_score:.3f})', lw=2)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2, label='Random Classifier')
plt.title('ROC Curve for Logistic Regression', fontsize=14)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.legend(loc='lower right', fontsize=10)
plt.grid(alpha=0.4)
plt.show()
# Output metrics
print("Classification Report:\n", classification_metrics)
print("AUC:", auc_score)

# Based on this, 
