import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt


#import os
#print(os.getcwd())


df=pd.read_csv("accepted_2007_to_2018Q4.csv")
df.columns = df.columns.str.strip()
print(df.head())
print(df.describe())
print(df.columns)
print(df.info())

#Cleaning the data

columns=['loan_amnt', 'term', 'int_rate', 'installment' , 'emp_length', 'home_ownership', 'annual_inc', 'purpose', 'dti', 'delinq_2yrs', 'revol_util', 'total_acc','emp_title']
df=df[columns + ['loan_status']]
df=df.dropna()
df['income_band'] = pd.cut(
    df['annual_inc'],
    bins=[0, 40000, 80000, 120000, 1000000],
    labels=['Low', 'Medium', 'High', 'Very High']

)


df['loan/income'] = np.where(
    df['annual_inc'] > 0,
    df['loan_amnt'] / df['annual_inc'],
    0
)

df['installment/income'] = np.where(
    df['annual_inc'] > 0,
    df['installment'] / df['annual_inc'],
    0
)


categorical_cols=['term', 'emp_length', 'home_ownership', 'purpose', 'emp_title', 'income_band']

numeric_cols=['loan_amnt', 'int_rate', 'installment', 'dti', 'delinq_2yrs', 'revol_util', 'total_acc', 'annual_inc', 'loan/income', 'installment/income']

preprocessor=ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),  # Standardize numeric features
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols) # One-hot encode categorical features
    ]
)


#Logistic Regression
X=df[['loan_amnt', 'term', 'int_rate', 'installment', 'emp_length', 'home_ownership', 'annual_inc', 'purpose', 'dti', 'delinq_2yrs', 'revol_util', 'total_acc', 'emp_title', 'income_band', 'loan/income', 'installment/income']]
y=df['loan_status'].apply(lambda x: 0 if x in ['Fully Paid'] else 1)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

def calculate_woe_iv(df, feature, target):

    # Create a contingency table
    contingency_table = pd.crosstab(df[feature], target)
    

    contingency_table.columns = ['Good', 'Bad']
    # Calculate the total number of good and bad outcomes
    contingency_table=contingency_table.replace(0, 0.0001)  # Avoid division by zero
    
    total_good=contingency_table['Good'].sum()
    total_bad=contingency_table['Bad'].sum()

    # Calculate the proportion of good and bad outcomes for each category
    contingency_table['bad_prop'] = contingency_table['Bad'] / total_bad
    contingency_table['good_prop'] = contingency_table['Good'] / total_good
    
    # Calculate WOE and IV
    contingency_table['woe'] = np.log(contingency_table['good_prop'] / contingency_table['bad_prop'])
    contingency_table['iv'] = (contingency_table['good_prop'] - contingency_table['bad_prop']) * contingency_table['woe']
    
    return contingency_table[['woe', 'iv']]

print("WOE and IV for 'home_ownership':")
print(calculate_woe_iv(df, 'home_ownership',y))



print(df['home_ownership'].value_counts())

print("WOE and IV for 'income_band':")
print(calculate_woe_iv(df, 'income_band',y))


# Predicted probabilities
y_prob = model.predict_proba(X_test)[:,1]

# ROC-AUC Score
auc = roc_auc_score(y_test, y_prob)

print("ROC-AUC Score:", auc)

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# Plot
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
plt.plot([0,1], [0,1], linestyle='--')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()

plt.show()

#KS Statistic
ks_statistic = max(tpr - fpr)
print("KS Statistic:", ks_statistic)

y_pred_prob = model.predict_proba(X_test)[:,1]

#Calibration

calibration_df = pd.DataFrame({
    'actual': y_test,
    'predicted_pd': y_pred_prob
})

calibration_df['decile'] = pd.qcut(calibration_df['predicted_pd'], 10, labels=False)

calibration_summary = calibration_df.groupby('decile').agg(
    avg_predicted_pd=('predicted_pd', 'mean'),
    actual_default_rate=('actual', 'mean'),
    observations=('actual', 'count')
)

print(calibration_summary)


plt.figure(figsize=(8,5))

plt.plot(
    calibration_summary['avg_predicted_pd'],
    calibration_summary['actual_default_rate'],
    marker='o'
)

# Perfect calibration line
plt.plot(
    [0,1],
    [0,1],
    linestyle='--'
)

plt.xlabel('Predicted PD')
plt.ylabel('Actual Default Rate')
plt.title('Calibration Plot')

plt.show()

#Assign PD to each loan

results=X_test.copy()
results['Actual_Default'] = y_test.values

results['PD']=model.predict_proba(X_test)[:,1]

#Create Risk Buckets
results['Risk_Bucket'] = pd.qcut(results['PD'], 5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
print("Risk Buckets Distribution:")
#print(results['Risk_Bucket'].value_counts())


bucket_summary = results.groupby('Risk_Bucket').agg(
    avg_pd=('PD', 'mean'),
    actual_default_rate=('Actual_Default', 'mean'),
    count=('PD', 'count')
)

print(bucket_summary)


#Calculating EL
LGD = 0.4
results['EAD']=df.loc[results.index, 'loan_amnt']
results['EL'] = results['PD'] * LGD * results['EAD']
portfolio_expected_loss = results['EL'].sum()
print(f"Total Expected Loss for the Portfolio: ${portfolio_expected_loss:,.2f}")

print(results.head())

#Bernoulli Distribution
n_simulations = 10000
portfolio_losses = []

for i in range(n_simulations):
    simulated_defaults = np.random.binomial(1, results['PD'])
    simulated_loss = (simulated_defaults * results['EAD'] * LGD).sum()
    
    portfolio_loss=simulated_loss.sum()
    portfolio_losses.append(portfolio_loss)

portfolio_losses = np.array(portfolio_losses)


import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))

plt.hist(
    portfolio_losses,
    bins=50
)

plt.xlabel('Portfolio Loss')
plt.ylabel('Frequency')
plt.title('Monte Carlo Portfolio Loss Distribution')

plt.show()

Credit_risk_var_95 = np.percentile(portfolio_losses, 95)
print(f"Credit Risk VaR at 95% confidence: ${Credit_risk_var_95:,.2f}")

Expected_shortfall_95 = portfolio_losses[portfolio_losses >= Credit_risk_var_95].mean()
print(f"Expected Shortfall at 95% confidence: ${Expected_shortfall_95:,.2f}")

plt.figure(figsize=(10,6))
plt.hist(
    portfolio_losses,
    bins=50
)
plt.axvline(Credit_risk_var_95, color='r', linestyle='--', label=f'VaR 95%: ${Credit_risk_var_95:,.2f}')
plt.xlabel('Portfolio Loss')
plt.ylabel('Frequency')
plt.title('Monte Carlo Portfolio Loss Distribution with VaR')
plt.legend()
plt.show()


#Stress Testing
stress_factor = 1.5
results['Stressed_PD'] = np.minimum(results['PD'] * stress_factor, 1.0)
results['Stressed_LGD'] = 0.6

#Stressed Mpnte Carlo Simulation
n_simulations = 10000
stressed_portfolio_losses = []

for i in range(n_simulations):
    simulated_defaults = np.random.binomial(1, results['Stressed_PD'])
    simulated_loss = (simulated_defaults * results['EAD'] * results['Stressed_LGD']).sum()
    
    stressed_portfolio_losses.append(simulated_loss)

stressed_portfolio_losses = np.array(stressed_portfolio_losses)
stressed_var_95 = np.percentile(stressed_portfolio_losses, 95)
print(f"Stressed Credit Risk VaR at 95% confidence: ${stressed_var_95:,.2f}")
print(f"Increase in VaR due to stress: ${stressed_var_95 - Credit_risk_var_95:,.2f}")

plt.figure(figsize=(10,6))
plt.hist(
    stressed_portfolio_losses,
    bins=50
)
plt.axvline(stressed_var_95, color='r', linestyle='--', label=f'Stressed VaR 95%: ${stressed_var_95:,.2f}')
plt.xlabel('Stressed Portfolio Loss')
plt.ylabel('Frequency')
plt.title('Monte Carlo Stressed Portfolio Loss Distribution with VaR')
plt.legend()




