import pandas as pd 
import numpy as np
import seaborn as sb
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from sklearn import metrics
from sklearn.model_selection import train_test_split
data = pd.read_csv('heart.csv')

# The first thing I am going to do is to see whether our features are correlated or no, beacause we want our explanatory variables
# To be indepenedent

sb.heatmap(data.corr())
# Now we are going to calculate the number of zeros and ones in our target variable and see if we have a skewed data
data['target'].value_counts()
# Now we will build a logistic regression model and see which of our feartures are explaning our target variable and which aren't
X = train.loc[:,X.columns!= 'target']
y = data['target']
model = sm.Logit(y,X)
result = model.fit()
print(result.summary2())
# The summary shows that the p-values of the variables 'slope', 'chol', 'fbs', 'age', 'restecg' and 'trestbps' are greater than 0.05
# Therefore we should remove them 
X = X.loc[:,X.columns!= 'slope']
X = X.loc[:,X.columns!= 'chol']
X = X.loc[:,X.columns!= 'fbs']
X = X.loc[:,X.columns!= 'age']
X = X.loc[:,X.columns!= 'restecg']
X = X.loc[:,X.columns!= 'trestbps']
# Logistic regression model fitting 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

# Accuracy of our classifier on test set 

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

# Confusion Matrix

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)




