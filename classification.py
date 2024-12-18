import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import requests

BOLD_START = '\033[1m'
END = '\033[0m'
UNDERLINE = '\033[4m'
PURPLE = '\033[95m'
CYAN = '\033[96m'
DARKCYAN = '\033[36m'
BLUE = '\033[94m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'

url =
local_filename = 'fruits.txt'

with requests.get(url, stream=True) as response:
    response.raise_for_status()
    with open(local_filename, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

fruits_data = pd.read_table('fruits.txt')
sns.countplot(x=fruits_data['fruit_name'])
print('%s%s%s%s'%(BOLD_START,UNDERLINE,'Count Plot',END))
plt.show()

print('%s%s%s%s'%(BOLD_START,UNDERLINE,'\nBox Plot of various numerical features\n',END))
fruits_data.drop('fruit_label', axis=1).plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False, figsize=(9,9),
                                        title='Box Plot for each input variable')
plt.show()

print('%s%s%s%s'%(BOLD_START,UNDERLINE,'\nHistogram of various numerical features\n',END))
fruits_data.drop('fruit_label' , axis = 1).hist(bins=30,figsize = (9,9))
plt.suptitle("Histogram for each numeric input variable")
plt.show()

print('%s%s%s%s'%(BOLD_START,UNDERLINE,'\nPair Plot\n',END))
numeric_fruits_data = fruits_data[['mass','width','height','color_score']]
sns.pairplot(numeric_fruits_data)
plt.show()

stats_data = fruits_data.describe()
print('%s%s%s%s'%(BOLD_START,UNDERLINE,'\nStatistics of various numerical features\n',END))
print(stats_data)

feature_names = ['mass', 'width', 'height', 'color_score']
X = fruits_data[feature_names]
y = fruits_data['fruit_label'] # target

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.25, random_state=100)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print('%s%s%s%s'%(BOLD_START,UNDERLINE,'\nLogistic Regression\n',END))
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train,y_train)
y_predicted = logistic_regression.predict(X_test)
acc_score_train_data = logistic_regression.score(X_train,y_train)
acc_score_test_data = logistic_regression.score(X_test,y_test)
print('Accuracy score on training data = %.3f'%acc_score_train_data)
print('Accuracy score on test/predicted data = %.3f'%acc_score_test_data)

print('%s%s%s%s'%(BOLD_START,UNDERLINE,'\nDecision Tree\n',END))
d_tree = DecisionTreeClassifier()
d_tree.fit(X_train,y_train)
# accuracy score on test data
acc_score_train = d_tree.score(X_train,y_train)
# accuracy score on test data or predicted dataa
acc_score_test = d_tree.score(X_test,y_test)
print('Accuracy score on training data = %.3f'%acc_score_train)
print('Accuracy score on test/predicted data = %.3f'%acc_score_test)

print('%s%s%s%s'%(BOLD_START,UNDERLINE,'\nK Nearest Neighbour \n',END))
KN_classifier = KNeighborsClassifier()
KN_classifier.fit(X_test,y_test)
# accuracy score on training data
acc_score_train = KN_classifier.score(X_train,y_train)
# accuracy score on test data or predicted dataa
acc_score_test = KN_classifier.score(X_test,y_test)
print('Accuracy score on training data = %.3f'%acc_score_train)
print('Accuracy score on test/predicted data = %.3f'%acc_score_test)

print('%s%s%s%s'%(BOLD_START,UNDERLINE,'\nNaive Bayes \n',END))
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
acc_train_data = gnb.score(X_train, y_train)
acc_test_data = gnb.score(X_test, y_test)
print('Accuracy score on training data = %.3f'%acc_train_data)
print('Accuracy score on test/predicted data = %.3f'%acc_test_data)

print('%s%s%s%s'%(BOLD_START,UNDERLINE,'\nSupport Vector Machine \n',END))
svm = SVC()
svm.fit(X_train, y_train)
acc_train_data =  svm.score(X_train, y_train)
acc_test_data = svm.score(X_test, y_test)
print('Accuracy score on training data = %.3f'%acc_train_data)
print('Accuracy score on test/predicted data = %.3f'%acc_test_data)