import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

train_data_path = "dataset/train.csv"
test_data_path = "dataset/test.csv"

train_dataframe = pd.read_csv(train_data_path)
test_dataframe = pd.read_csv(test_data_path)


label_encoder = LabelEncoder()

train_dataframe['Gender'] = label_encoder.fit_transform(train_dataframe['Gender'])
train_dataframe = pd.get_dummies(train_dataframe, columns=['Geography'])
x_train = train_dataframe.drop(['Exited','Surname','CustomerId'], axis=1)
y_train = train_dataframe['Exited']

#x_train.head(10)
#y_train.head(10)

test_dataframe['Gender'] = label_encoder.fit_transform(test_dataframe['Gender'])
test_dataframe = pd.get_dummies(test_dataframe, columns=['Geography'])
x_test = test_dataframe.drop(['Surname','CustomerId'], axis=1)
x_test.head(10)

model = XGBClassifier(objective='binary:logistic')

model.fit(x_train, y_train)

y_pred = model.predict(x_test)
y_pred_proba = model.predict_proba(x_test)[:,1]

output = x_test
output['Exited'] = y_pred_proba

columns_to_keep = ['id', 'Exited']

# Find columns to drop
columns_to_drop = [col for col in output.columns if col not in columns_to_keep]
# Drop the columns
output.drop(columns=columns_to_drop, inplace=True)
output.head(5)

output.to_csv('output/submission.csv', index=False)
