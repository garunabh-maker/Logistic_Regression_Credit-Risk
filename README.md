# Logistic_Regression_Credit-Risk
Credit Risk Modeling and Portfolio Risk Analysis
Overview

This project builds an end-to-end retail credit risk framework using LendingClub loan data. A logistic regression model is developed to estimate borrower Probability of Default (PD) using feature engineering, preprocessing, and WOE/IV analysis.

The project also estimates Expected Loss (EL), Credit VaR, and Expected Shortfall (ES) using Monte Carlo simulation and stress testing techniques.

Features
Data cleaning and preprocessing
Feature engineering
WOE and IV analysis
Logistic Regression PD model
ROC-AUC and KS statistic evaluation
Calibration analysis
Risk bucket segmentation
Expected Loss calculation
Monte Carlo portfolio loss simulation
Credit VaR and Expected Shortfall estimation
Stress testing framework
Technologies Used
Python
Pandas
NumPy
Scikit-learn
Matplotlib
Model Workflow
Data preprocessing and cleaning
Feature engineering
WOE and IV calculation
Logistic regression training
Model evaluation (ROC-AUC, KS, calibration)
PD estimation
Expected Loss calculation
Monte Carlo portfolio simulation
Credit VaR and Expected Shortfall estimation
Stress testing analysis
Key Risk Metrics
Expected Loss

EL = PD × LGD × EAD

Credit VaR

Measures portfolio loss at a specified confidence level.

Expected Shortfall

Measures the average loss beyond the VaR threshold.

Stress Testing

Stress scenarios are created by:

Increasing PD assumptions
Increasing LGD assumptions

The impact on portfolio losses and VaR is then analyzed.

Assumptions
Defaults are independent
LGD is assumed constant
Loan amount is used as EAD
Logistic regression is used as baseline PD model
Limitations
Default correlation is not modeled
Macroeconomic variables are excluded
LGD is assumed rather than estimated
Portfolio may not fully represent real bank exposures
Results

The model successfully:

Estimated borrower PDs
Evaluated model discrimination and calibration
Simulated portfolio loss distributions
Calculated Credit VaR and Expected Shortfall
Assessed portfolio sensitivity under stress scenarios
Author

Arunabh Ghosh
