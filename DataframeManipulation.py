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
from joblib import dump
from joblib import load



# =========== SCIKIT AI MODELS ===============
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression # --> used for classification
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor


# =========== USER INTERFACE ===============
import tkinter as tk
from tkinter import messagebox



# ============= Import the dataset ============
# Reading given dataset
#df = pd.read_excel('Car_Purchasing_Data.xlsx')
df = pd.read_csv("Car_Purchasing_Data.csv")





# ============= Display first 5 rows of the dataset ===========
df.head()

# creating space from content to content
print("\n\n")





# ============= Display last 5 rows of the dataset ============
df.tail()

# creating space from content to content
print("\n\n")





# ============= Determine shape of the dataset (shape - total numbers of rows and columns) ============
shape = df.shape
print(f"\nShape of given dataframe {shape}.TO be precise, this dataset consists of {shape[0]} rows and {shape[1]} columns\n")

# creating space from content to content
print("\n\n")






# ============= Display concise summary of the dataset (info) ============
print(f"\nConcise summary of given data set is {df.info()}\n")

# creating space from content to content
print("\n\n")





# ============= Check the null values in dataset (isnull) ============
# isnull will show nulls
# sum will show count of the null values
print(f"\nTotal ammount of null values in data set is: \n{df.isnull().sum()}\n")

# creating space from content to content
print("\n\n")





# ============ Identify library to plot graph to understand relations among various columns =============
# installing seaborn library pip install seaborn
# imported matplotlib
# shows relationship between data
# no need to print as show does it

# Pair plot
sbn.pairplot(df)
plt.show()

# creating space from content to content
print("\n\n")

# individual columns
sbn.scatterplot(x ='Age', y = 'Customer Name', data = df)
plt.show()

# creating space from content to content
print("\n\n")





# ============ Create input dataset from original dataset by dropping irrelevant features ============
# dropped columns will be stored in variable X
# axis=1: This parameter specifies the axis along which to drop the columns. 
#In this case, axis=1 means that you want to drop columns (as opposed to rows, which would be axis=0).

# columns stated are irrelevant to this dataset hence being dropped
X = df.drop(['Customer Name','Customer e-mail','Country', 'Car Purchase Amount'], axis=1)
print(f"\nDropped columns from dataset is {X}\n")

# creating space from content to content
print("\n\n")





# ============ Create output dataset from original dataset =============
# output column will be stored in variable Y
Y = df['Car Purchase Amount']
print(f"\nOutput column from dataset is {Y}\n")

# creating space from content to content
print("\n\n")





# ============ Transform input dataset into percentage based weighted between 0 and 1 =============
# input columns are the output colouts to which we need to use Gender, Age, Annual Salary, Credit Card Debt, Net Worth, Car Purchase Amount

sc = MinMaxScaler()
x_scaler = sc.fit_transform(X)
print(x_scaler)

# creating space from content to content
print("\n\n")




# ============ Transform output dataset into percentage based weighted between 0 and 1 =============

sc1 = MinMaxScaler()
y_reshape = Y.values.reshape(-1,1)
y_scaler = sc1.fit_transform(y_reshape)
print(y_scaler)

# creating space from content to content
print("\n\n")



# ============ Print first few rows of scaled input dataset =============
# :5 number is based on what is written in array so if it says :7, 7 rows of data will show
print(f"\nthe follow is the data from dropped columns{x_scaler[:5]}\n")

# creating space from content to content
print("\n\n")





# ============ Print first few rows of scaled output dataset =============
# y_scaler.head() nto working - used below mixmaxScaler function
# :5 number is based on what is written in array so if it says :7, 7 rows of data will show
print(f"\nthe follow is the data from output columns{y_scaler[:7]}\n")

# creating space from content to content
print("\n\n")




# ============ Split data into training and testing sets =============
# split the dataset
# if the size has been dictated, shuffle wouldnt take affect
X_train, X_test, Y_train, Y_test = train_test_split( x_scaler, y_scaler, test_size = 0.2, train_size = 0.8, random_state = 42)

print("X_train results:\n", X_train)
print("X_test results:", X_test)
print("Y_train results:", Y_train)
print("Y_test results:\n", Y_test)

# creating space from content to content
print("\n\n")




# ============ Print shape of test and training data =============
print("Shape of training given dateset:\n", X_train.shape)
print("Shape of test given dateset", X_test.shape)

# creating space from content to content
print("\n")

print("Shape of training given dateset:", Y_train.shape)
print("Shape of test given dateset\n", Y_test.shape)


# creating space from content to content
print("\n\n")





# ============ Print first few rows of test and training data =============
print("First few rows of training given dateset:\n", X_train[:5])
print("First few rows of test given dateset", X_test[:6])

# creating space from content to content
print("\n")

print("First few rows of training given dateset:", Y_train[:7])
print("First few rows of test given dateset\n", Y_test[:8])

# creating space from content to content
print("\n\n")




# ============ Import and initialize AI models (need 10) =============
# models used [LinearRegression | Lasso | LogisticRegression | Ridge | ElasticNet | DecisionTreeRegressor]

# assign variables
lnr = LinearRegression()
lso = Lasso()
# lgs = LogisticRegression() --> used for classification
rdg = Ridge()
eln = ElasticNet(alpha = 1.0, l1_ratio = 0.5)
dtr = DecisionTreeRegressor()




# ============ Train models using training data =============
# use .fit -->  method is how a machine learning model learns from the training data. It adjusts the model's parameters so that it can make accurate predictions.
lnr.fit(X_train, Y_train)
lso.fit(X_train, Y_train)
# lgs.fit(X_train, Y_train) --> used for classification
rdg.fit(X_train, Y_train)
eln.fit(X_train, Y_train)
dtr.fit(X_train, Y_train)





# ============ Prediction on test data =============
# predicting the x_test  to the models variables from earlier

# assigned variables
lnr_pred = lnr.predict(X_test)
lso_pred = lso.predict(X_test)
# lgs_pred = lgs.predict(X_test) --> used for classification
rdg_pred = rdg.predict(X_test)
eln_pred = eln.predict(X_test)
dtr_pred = dtr.predict(X_test)



# ============ Evaluate model performance =============
# use RMSE --> Root Mean Squared Error
# mean_squared_error alias = mse

# variable assigned
lnr_rmse = mse(Y_test, lnr_pred, squared = False)
lso_rmse = mse(Y_test, lso_pred, squared = False)
# lgs_rmse = mse(Y_test, lgs_pred, squared = False) --> used for classification
rdg_rmse = mse(Y_test, rdg_pred, squared = False)
eln_rmse = mse(Y_test, eln_pred, squared = False)
dtr_rmse = mse(Y_test, dtr_pred, squared = False)

# creating space from content to content
print("\n\n")




# ============ Display evaluation results =============
# displaying the RMSE and result from last exercise
print(f"Linear Regression RMSE results are: {lnr_rmse}")
print(f"Lasso Regression RMSE results are: {lso_rmse}")
# print(f"Logistic Regression RMSE results are: {lgs_rmse}") --> used for classification
print(f"Ridge Regression RMSE results are: {rdg_rmse}")
print(f"Elastic Net Regression RMSE results are: {eln_rmse}")
print(f"Desicion Tree Regression RMSE results are: {dtr_rmse}")


# creating space from content to content
print("\n\n")




# # ============ Choose best model ============
model_entity = [lnr, lso, rdg, eln, dtr]
rmse_results = [lnr_rmse, lso_rmse, rdg_rmse, eln_rmse, dtr_rmse]

optimum_model_index = rmse_results.index(min(rmse_results))
optimum_model_entity = model_entity[optimum_model_index]


#  pictorial representaion of data
modelling_object = ["Linear Regression", "Lasso Regression", "Ridge Regression", "Elastic Net Regression", "Decision Tree Regression"]
rmse_results = [lnr_rmse, lso_rmse, rdg_rmse, eln_rmse, dtr_rmse]

# Creating figure
plt.figure(figsize = (10, 6))

# Plotting the bars and colours
# Royal Blue = #4169E1 | Emerald Green = #50C878 | Crimson Red = #DC143C | Golden Yellow = #FFD700 | Coral Pink = #FF6F61
colors = ["#4169E1", "#50C878", "#DC143C", "#FFD700", "#FF6F61"]  
bars = plt.bar(modelling_object, rmse_results, color = colors)

# Adding labels to the bars and its value
# displaying value in center of vertical and horizontal alignment
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.00001, round(yval, 2), va = "bottom", ha = "center")


# Labels and title
plt.xlabel("Regression Models")
plt.ylabel("RMSE")
plt.title("RMSE Regression Model Comparison")
plt.xticks(rotation = 45)

# Showing the plot
plt.show()


# ===============================================================================================================================================================================
# ========================================================================= RETRAINING ==========================================================================================
# ===============================================================================================================================================================================

# Retraining the model for given dataset
linReg_retrain = LinearRegression()
linReg_retrain.fit(x_scaler, y_scaler)


# Saving the models
# dump & load = serialising (saving) and deserialising (loading) || # referes to the optimum_model_entity created earlier || # give saved model a name with .joblin extension
dump(optimum_model_entity, "Car_Purchasing.joblib")

# loading the saved file from earlier
load_model = load("Car_Purchasing.joblib")




# # ============ User Inputs ============
# gender = int(input("Please enter your gender 'Enter 0 for FEMALE and 1 for MALE': "))
# age = int(input("Please enter your age: "))
# annual_income = float(input("Please enter your yearly income: "))
# credit_card_debt = float(input("Please enter your credit card debt: "))
# net_worth = float(input("Please enter your net worth: "))

# print("\n\n")





# # ============ Predicting using given inputs ============
# # sc = alias for mixmaxascaler
# input_details = sc.transform([[gender, age, annual_income, credit_card_debt, net_worth]])
# input_pred = load_model.predict(input_details)
# # input_tf = input_pred.reshape(-1,1)

# print(input_pred)
# # print("\n")
# print(f"value of car purchase amount prediction based on your given information is: {sc1.inverse_transform(input_pred)}")


# print("\n\n")



    # ===============================================================================================================================================================================
    # ========================================================================= USER INTERFACE ======================================================================================
    # ===============================================================================================================================================================================
def prediction():
    try:
        # ============ User input field ============
        # getting users input from UI entry fields
        gender = int(gender_input.get())
        age = int(age_input.get())
        annual_income = float(annual_income_input.get())
        credit_card_debt = float(credit_card_debt_input.get())
        net_worth = float(net_worth_input.get())


        # ============ Predicting using given inputs ============
        # sc = alias for mixmaxascaler
        input_details = sc.transform([[gender, age, annual_income, credit_card_debt, net_worth]])
        input_pred = load_model.predict(input_details)
        predicted_amount = sc1.inverse_transform(input_pred)


        # ============ Predicting Output ============
        result_label.config(text = f"Prediction based on your given information is: {predicted_amount[0][0]:.5f}")
    
    except ValueError:
        messagebox.showerror("OOPS! Invalid input... Please enter numerical values into valid fields!")



# ============================================
# ============ CREATING UI WINDOW ============
root = tk.Tk()
# title for the UI
root.title("Car Purchase Amount Prediction")


# =============================================
# ============ ADDING INPUT FIELDS ============
#padx and pady = paddying x and y

# Gender Input Field
tk.Label(root, text = "Please enter your gender 'Enter 0 for FEMALE and 1 for MALE': ").grid(row = 0, column = 0, padx = 10, pady = 5)
gender_input = tk.Entry(root)
gender_input.grid(row = 0, column = 1, padx = 10, pady = 5)


# Age Input Field
tk.Label(root, text = "Please enter your age: ").grid(row = 1, column = 0, padx = 10, pady = 5)
age_input = tk.Entry(root)
age_input.grid(row = 1, column = 1, padx = 10, pady = 5)


# Annual Income Input Field
tk.Label(root, text = "Please enter your yearly income: ").grid(row = 2, column = 0, padx = 10, pady = 5)
annual_income_input = tk.Entry(root)
annual_income_input.grid(row = 2, column = 1, padx = 10, pady = 5)


# Credit Card Debt Input Field
tk.Label(root, text = "Please enter your credit card debt: ").grid(row = 3, column = 0, padx = 10, pady = 5)
credit_card_debt_input = tk.Entry(root)
credit_card_debt_input.grid(row = 3, column = 1, padx = 10, pady = 5)


# Credit Card Debt Input Field
tk.Label(root, text = "Please enter your net worth: ").grid(row = 4, column = 0, padx = 10, pady = 5)
net_worth_input = tk.Entry(root)
net_worth_input.grid(row = 4, column = 1, padx = 10, pady = 5)


# ===========================================================
# ============ ADDING PREDICT BUTTON & PREDCTION ============

# Button for click event --> command = function name
predict_button = tk.Button(root, text = "Predict", command = prediction)
predict_button.grid(row = 7, column = 0, columnspan = 2, pady = 10)

result_label = tk.Label(root, text = "")
result_label.grid(row = 9, column = 0, columnspan = 2, pady = 10)



# ===============================================
# ============ RUNNING TKINTER EVENT ============
root.mainloop()

#script to run file
# python DataframeManipulation.py