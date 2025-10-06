import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#from xgboost import XGBClassifier

#from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

scaler = StandardScaler()

drop_columns = ['id','DateTime','meal']

# Load train dataset
train_df = pd.read_csv('https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv')
train_df['Total'] = scaler.fit_transform(train_df[['Total']]) # Standardize total column

# Load test dataset
test_df = pd.read_csv('https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv')
test_df['Total'] = scaler.fit_transform(test_df[['Total']]) # Standardize total column

Y = train_df['meal']
X = train_df.drop(drop_columns, axis=1) 


#model = XGBClassifier(n_estimators=100, max_depth=10, objective='binary:logistic') # Declare xgb classication model
#modelFit = model.fit(X, Y) # Fit the model 

model = RandomForestClassifier(n_estimators=100, n_jobs = -1) # Generate the random forest model
modelFit = model.fit(X, Y) # Fit the model 

pred = modelFit.predict(test_df.drop(drop_columns, axis=1)) # Generate prediction

