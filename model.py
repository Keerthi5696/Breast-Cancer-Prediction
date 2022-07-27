# import packages
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import pickle
# filter warnings that can be ignored
import warnings
warnings.filterwarnings('ignore')

#read dataset using read_csv() - data.csv
df = pd.read_csv("datapr.csv")
# removing the duplicate values
df = df.drop_duplicates(subset='id', inplace=False)
# # Outlier Detection and Handling
Q1 = df.quantile(.25)
Q3 = df.quantile(.75)
IQR = Q3-Q1

for features in df.columns[2:]:
    OF_Q1 = df[features].quantile(0.25)
    OF_Q2 = df[features].quantile(0.50)
    OF_Q3 = df[features].quantile(0.75)
    OF_IQR = OF_Q3-OF_Q1
    OF_low_limit = OF_Q1-1.5*OF_IQR
    OF_up_limit = OF_Q3+1.5*OF_IQR
    OF_outlier = df[(df[features] < OF_low_limit) |
                    (df[features] > OF_low_limit)]
    df[features] = df[features].clip(OF_up_limit, OF_low_limit)

# All outliers were removed.

# # Encoding


labelencoder_Y = LabelEncoder()
df.diagnosis = labelencoder_Y.fit_transform(df.diagnosis)


d_list = ['id']
df = df.drop(d_list, axis=1)


# # Feature selection with correlation

# Drop high correlated columns in a dataset


drop_list_cor = ['perimeter_mean', 'radius_mean', 'compactness_mean', 'concave points_mean', 'radius_se', 'perimeter_se', 'radius_worst', 'perimeter_worst', 'compactness_worst', 'concave points_worst',
                 'compactness_se', 'concave points_se', 'texture_worst', 'area_worst', 'smoothness_mean', 'symmetry_mean', 'fractal_dimension_mean', 'texture_se', 'smoothness_se', 'symmetry_se']
# do not modify df, we will use it later
df1 = df.drop(drop_list_cor, axis=1)


# split the dataset into dependent(X) and Independent(Y) datasets
X = df1.drop(['diagnosis'], axis=1)
y = df1['diagnosis']


# spliting the data into trainning and test dateset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)


# # Model Building

# # Logistic Regression


rmodel = LogisticRegression()

# training the Logistic Regression model using Training data
rmodel.fit(X_train, y_train)


# accuracy on training data
X_train_prediction = rmodel.predict(X_train)
training_data_accuracy = accuracy_score(y_train, X_train_prediction)


# accuracy on test data
X_test_prediction = rmodel.predict(X_test)
test_data_accuracy = accuracy_score(y_test, X_test_prediction)
print('Accuracy on test data = ', test_data_accuracy)


# # GradientBoostingClassifier


gb_clf = GradientBoostingClassifier(
    n_estimators=20, learning_rate=0.5, max_features=2, max_depth=2, random_state=0)
gb_clf.fit(X_train, y_train)
predictions = gb_clf.predict(X_test)
print("Accuracy score (training): {0:.3f}".format(
    gb_clf.score(X_train, y_train)))
print("Accuracy score (test): {0:.3f}".format(gb_clf.score(X_test, y_test)))


pickle.dump(gb_clf, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))
print(model)

# With the Gradien tBoosting Classifier, we achieve the highest precision, recall, and f1-score. As a result, we chose the random forest classifier.
