import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn



# =========== Model Libraries ===============
# Package needed --> pip install scikit-learn
from sklearn.preprocessing import MinMaxScaler  
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from mpl_toolkits.mplot3d import Axes3D



# =========== AI Models ===============
from sklearn.svm import SVR 
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge



# ======= Import the dataset ======
# Reading given dataset
#df = pd.read_excel('Car_Purchasing_Data.xlsx')
df = pd.read_csv("Car_Purchasing_Data.csv")





# ======= Display first 5 rows of the dataset ======
df.head()

print("\n\n")





# ======= Display last 5 rows of the dataset ======
df.tail()

print("\n\n")





# ======= Determine shape of the dataset (shape - total numbers of rows and columns) ======
shape = df.shape
print(f"\nShape of given dataframe {shape}.TO be precise, this dataset consists of {shape[0]} rows and {shape[1]} columns\n")

print("\n\n")






# ======= Display concise summary of the dataset (info) ======
print(f"\nConcise summary of given data set is {df.info()}\n")

print("\n\n")





# ======= Check the null values in dataset (isnull) ======
# isnull will show nulls
# sum will show count of the null values
print(f"\nTotal ammount of null values in data set is: \n{df.isnull().sum()}\n")

print("\n\n")





# ====== Identify library to plot graph to understand relations among various columns =======
# installing seaborn library pip install seaborn
# imported matplotlib
# shows relationship between data
# no need to print as show does it

# Pair plot
sbn.pairplot(df)
plt.show()

# individual columns
sbn.scatterplot(x ='Age', y = 'Customer Name', data = df)
plt.show()


print("\n\n")




# ====== Create input dataset from original dataset by dropping irrelevant features ======
# dropped columns will be stored in variable X
# axis=1: This parameter specifies the axis along which to drop the columns. 
#In this case, axis=1 means that you want to drop columns (as opposed to rows, which would be axis=0).

# columns stated are irrelevant to this dataset hence being dropped
X = df.drop(['Customer Name','Customer e-mail','Country', 'Car Purchase Amount'], axis=1)
print(f"\nDropped columns from dataset is {X}\n")

print("\n\n")





# ====== Create output dataset from original dataset =======
# output column will be stored in variable Y
Y = df['Car Purchase Amount']
print(f"\nOutput column from dataset is {Y}\n")

print("\n\n")





# ====== Transform input dataset into percentage based weighted between 0 and 1 =======
# input columns are the output colouts to which we need to use Gender, Age, Annual Salary, Credit Card Debt, Net Worth, Car Purchase Amount

sc = MinMaxScaler()
x_scaler = sc.fit_transform(X)
print(x_scaler)

print("\n\n")




# ====== Transform output dataset into percentage based weighted between 0 and 1 =======

sc = MinMaxScaler()
y_reshape = Y.values.reshape(-1,1)
y_scaler = sc.fit_transform(y_reshape)
print(y_scaler)

print("\n\n")



# ====== Print first few rows of scaled input dataset =======
# y_scaler.head() nto working - used below mixmaxScaler function
# :5 number is based on what is written in array so if it says :7, 7 rows of data will show
print(f"\nthe follow is the data from dropped columns{x_scaler[:5]}\n")

print("\n\n")





# ====== Print first few rows of scaled output dataset =======
# y_scaler.head() nto working - used below mixmaxScaler function
# :5 number is based on what is written in array so if it says :7, 7 rows of data will show
print(f"\nthe follow is the data from output columns{y_scaler[:7]}\n")

print("\n\n")




# ====== Split data into training and testing sets =======
# split the dataset
# if the size has been dictated, shuffle wouldnt take affect
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 0.2, train_size = 0.8, random_state = 42, shuffle = True)

print("X_train results:\n", X_train)
print("X_test results:", X_test)
print("Y_train results:", Y_train)
print("Y_test results:\n", Y_test)

print("\n\n")




# ====== Print shape of test and training data =======
print("Shape of training given dateset:\n", X_train.shape)
print("Shape of test given dateset", X_test.shape)

print("Shape of training given dateset:", Y_train.shape)
print("Shape of test given dateset\n", Y_test.shape)

print("\n\n")




# ====== Print first few rows of test and training data =======
print("First few rows of training given dateset:\n", X_train[:5])
print("First few rows of test given dateset", X_test[:6])

print("First few rows of training given dateset:", Y_train[:7])
print("First few rows of test given dateset\n", Y_test[:8])

print("\n\n")




# ====== Import and initialize AI models (need 10) =======
# used models --> [ SVR | XGBRegressor | RandomForestRegressor | BayesianRidge |  statsmodels.api --> QuantReg(endog, exog)]

# assigned variables
svm = SVR()
xgr = XGBRegressor()
rdf = RandomForestRegressor()
bsr = BayesianRidge()


print("\n\n")




# ====== Train models using training data =======
# .fit functions allows models to ake accurate predictions
svm.fit(X_train, Y_train)
xgr.fit(X_train, Y_train)
rdf.fit(X_train, Y_train)
bsr.fit(X_train, Y_train)

print("\n\n")




# ====== Prediction on test data =======
# training x models

# variables assigned
svm_pred = svm.predict(X_test)
xgr_pred = xgr.predict(X_test)
rdf_pred = rdf.predict(X_test)
bsr_pred = bsr.predict(X_test)

print("\n\n")





# ====== Evaluate model performance =======
# use RSME --> root mean squared error
# mean_squared_error alias = mse

# variables
svm_rmse = mse(Y_test, svm_pred, squared = False)
xgr_rmse = mse(Y_test, xgr_pred, squared = False)
rdf_rmse = mse(Y_test, rdf_pred, squared = False)
bsr_rmse = mse(Y_test, bsr_pred, squared = False)

print("\n\n")


 

# ====== Display evaluation results =======
print(f"Support Vector Regression RMSE results are: {svm_rmse}")
print(f"Xtreme Gradient Boosting Regression RMSE results are: {xgr_rmse}")
print(f"Bayesian Ridge Regression RMSE results are: {bsr_rmse}")
print(f"Random Forest Regression RMSE results are: {rdf_rmse}")





# ====== Choose best model =======
entity_models = [svm, xgr, rdf, bsr]
rmse_sum = [svm_rmse, xgr_rmse, rdf_rmse, bsr_rmse]

optimal_model_index = rmse_sum.index(min(rmse_sum))
optimal_model_entity = entity_models[optimal_model_index]


# graph presentation
modelling_entities = ["Support Vector Regression", "XGB Regression", "BayesianRidge Regression", "Random Forest Regression"]
plt.figure(figsize = (10, 7))


# Colour Pallet
# Lime Green = #32CD32 | Tomato Red = #FF6347 | Orange = #FFA500 | Medium Orchid = #BA55D3
colors = ["#32CD32", "#FF6347", "#FFA500", "#BA55D3"]

# plot pie chart
wedges, texts, autotexts = plt.pie(rmse_sum, labels = modelling_entities, colors = colors, autopct = "%1.2f%%", startangle = 180)


# text colour in pie chart
for autotext in autotexts:
    autotext.set_color("black")


#Adding key legend
plt.legend(wedges, modelling_entities, loc = "lower right")

# Graph Title
plt.title("RMSE Regression Model Results")
plt.axis("equal")


# graph output
plt.show()


