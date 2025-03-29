# RateMyProfessor Capstone

Statistical analysis of RateMyProfessor.com data using hypothesis testing, regression models, and bootstrapping. This project was completed as part of the final capstone for NYU’s PODS II course (Fall 2024), under the instruction of Dr. Pascal Wallisch.

## Project Overview

This project investigates bias and behavioral patterns in over 10,000 student-submitted professor evaluations. Through exploratory analysis and statistical testing, the project addresses multiple questions around perceived quality, difficulty, experience, and instructor characteristics.

## Key Research Questions

1. Do male professors receive higher ratings than female professors?
2. Does instructor experience (number of ratings) relate to quality?
3. How does perceived difficulty impact average rating?
4. Is online teaching experience associated with better ratings?
5. What is the relationship between retake likelihood and average rating?
6. Is there evidence of bias based on physical appearance ("hotness")?
7. Can difficulty predict rating using linear regression?
8. Which predictors are most useful for multiple regression models of quality?
9. Can we classify whether a professor gets a "hot" rating based on their features?

## Methods Used

- Mann-Whitney U Tests (non-parametric significance testing)
- Linear and logistic regression
- Bootstrapping for statistical power and resampling
- Data cleaning (NaN handling, threshold filtering)
- Visualization (scatter plots, histograms, heatmaps, ROC curves)
- Model evaluation (R², RMSE, AUROC, classification metrics)

## Technologies

- Python
- NumPy, pandas, matplotlib, scikit-learn
- Jupyter Notebook / Spyder
- Git & GitHub

## File Structure

- `Capstone_final.py` – Main code file with all analyses and visualizations
- `README.md` – This file

## Full Project Report

For a detailed write-up of methodology, decision-making, and statistical results, see the [capstone_writeup.md](./capstone_writeup.pdf) file.

## Author

Michael Kazarian  
New York University, College of Arts and Sciences  
B.A. Mathematics & Data Science, Class of 2026
