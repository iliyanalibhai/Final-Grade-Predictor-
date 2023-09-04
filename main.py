import pandas as pd
import numpy as np
import sklearn # for linear regression model
from sklearn import linear_model
from sklearn.utils import shuffle

# read my data in
data = pd.read_csv("student-mat.csv", sep = ";") # each attribite seperated by ;, not a ,
# print(data.head())

data =  data[["G1", "G2", "G3", "studytime", "failures", "absences", "freetime"]] # attributes

print(data.head())

predict = "G3" # this is what we are trying to predict, this is our label

x = np.array(data.drop([predict], axis=1))  # Remove one '1' argument, this is all of our attributes
y = np.array(data[predict]) # all of our labels

#split our data into 4 arrays, so when we test we can test without computer knowing all info

x_train , x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y, test_size = 0.1) # x and y represents attributes and labels

linear = linear_model.LinearRegression() 

linear.fit(x_train, y_train) # using this data we are going to create a best fit line
accuracy = linear.score(x_test,y_test) #returns us a value an accuracy of our model, the score
# print(accuracy)

print("Co: \n" ,linear.coef_) #Coefficient for each of our 6 attributes
print("Intercept: \n" , linear.intercept_) 

predictions = linear.predict(x_test) # takes an array of arrays and the model will make predictions on our test_data, that we didn't train our model on
print(f"The accuracy for this model is {accuracy} \n")
for x in range(len(predictions)):
    rounded_prediction = round(predictions[x], 2) # round the prediction nearest 100 place
    print(f"Prediction: {rounded_prediction}, Attributes {x_test[x]}, Actual: {y_test[x]} \n")  # outputs the prediction, each of the used attributes, and the Actual grade
