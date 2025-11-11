#!/usr/bin/env python
# coding: utf-8

# # Importing libraries

# In[2]:


import numpy as np #to create numeric arrays
import pandas as pd #to create data panes
from sklearn.preprocessing import StandardScaler #to standardize the data in one single range
from sklearn.model_selection import train_test_split #split our data into training and test 
from sklearn import svm #support vector machine
from sklearn.metrics import accuracy_score


# # Data Collection and Analysis
# Diabetes Dataset (Dataset is for Females)

# In[3]:


#Loading the dataset
file_path = r'C:\Users\ilika\Downloads\diabetes.csv'

# Read the CSV file into a Pandas DataFrame
diabetes_dataset = pd.read_csv(file_path)


# In[4]:


#print first 5 rows for the dataset

diabetes_dataset.head()


# In[5]:


#number of rows and columns in our dataset
diabetes_dataset.shape


# Data is taken from 768 people with 9 attributes

# # EDA and Statistical Measure Analysis of the Dataset

# In[8]:


#Exploratary and statistical measure analysis of the data
diabetes_dataset.describe()


# we obtain the count, average, standard deviation, minimum value, 25% (example:25% values are less than 99 in glucose,and so on), and maximum value

# In[9]:


#check how many cases for diabetic and non diaebtic
diabetes_dataset['Outcome'].value_counts()


# 0---> Non Diabetic
# 1---> Diabetic

# In[10]:


#to calucalte mean value for both the cases (diabetic or no diabetic)

diabetes_dataset.groupby('Outcome').mean()


# So we obtain mean value of glucise, for example for non diabetic (0) is 109.9 and for diabetic is 141, and for example the mean age of people without diabetes is 32 and with diabetes is 38

# # Data Preprocessing

# In[12]:


#to seperate the data from the labels
X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
Y = diabetes_dataset['Outcome']


# In[13]:


print(X)


# In[14]:


print(Y)


# In python indexing begins from 0 , so we see '767' but length as 768

# Data Standardisation

# In[15]:


#standardizing the dataset will enable in making better decisions
scaler= StandardScaler()


# In[21]:


scaler.fit(X) #we are fitting all inconsistent data with our standard scaler function and based on the standardization we are transforming it to a standard range


# In[22]:


standardized_data= scaler.transform(X)


# In[23]:


print(standardized_data)


# so all values are in similar range form 0 to 1

# In[26]:


X= standardized_data
Y= diabetes_dataset['Outcome']

#feeding standardized data to variable X and in Y we have the outcomes, whcic we had already seperated
#we will use this to train our model


# In[27]:


print(X)
print(Y)


# # Train and Test Split

# In[29]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, stratify=Y, random_state=2)

#we are taking four variables so like X (which was standardized data) will be split into two arrays test and train and similar for Y and once the model is trained we will evaluate it using test data
#0.2 representa how much data you want for test data , so we want 20% data as test data; and now Y has outcomes either 0 or 1 which we want to split in same proprtion otherwise all the cases will go to only a single data frame for example all the diabetic cases will got to X train
#random state is basically the indexing or serial number for splitting the data


# In[30]:


print(X.shape, X_train.shape, X_test.shape)


# this shows that totallythere were 768 responses for your dataset, out of which, 614 go to training data and 154 go to test

# # Training the Model : usage of svm

# In[32]:


classifier= svm.SVC(kernel='linear')
#this will load the svm model in variable classifier


# In[34]:


#feed training data to classifier
#training the classifier model
classifier.fit(X_train, Y_train)


# # Model Evaluation
# 
# Accuracy Score

# In[35]:


#model evaluation i.e to check how many times our model is predicting correctly



X_train_predicition = classifier.predict(X_train)

#this will predict all the labels/outcomes for X train and we are storing it in X_train_predicition

training_data_accuracy= accuracy_score(X_train_predicition, Y_train)

#compare the prediction of our model with the original data;accuracy score


# In[36]:


print('Accuracy score on the training data : ', training_data_accuracy)


# this means that out of 100 predictions model is predicting approximately 79 times correctly

# In[37]:


#we also need to check the accuracy of model based on unknown data i.e based on test data

X_test_predicition = classifier.predict(X_test)
testing_data_accuracy= accuracy_score(X_test_predicition, Y_test)


# In[38]:


print('Accuracy score on the testing data : ', testing_data_accuracy)


# # Building a Predicitve system Model

# In[45]:


input_data = (0,118,84,47,230,45.8,0.551,31)
#change the list above to numpy array
input_data_as_numpy_array= np.asarray(input_data)

#reshape as we are predicitng for one instance as we need to tell model that we need predicitipn only for only one data point
input_data_reshaped= input_data_as_numpy_array.reshape(1,-1)

#standardize the input_data
std_data= scaler.transform(input_data_reshaped)
print(std_data)

prediction=classifier.predict(std_data)
print(prediction)

if prediction[0] == 0:
    print('The Person is not diabetic')
else:
    print('The Person is Diabetic ')


# In[46]:


attributes = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

# Initialize an empty input_data array
input_data = np.zeros(len(attributes))


# In[49]:


for i in range(len(attributes)):
    while True:
        try:
            input_data[i] = float(input(f"Enter {attributes[i]}: "))
            break  # Break the loop if the user enters a valid number
        except ValueError:
            print("Please enter a valid number.")
            
            # Change the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the data for prediction
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Standardize the input data
std_data = scaler.transform(input_data_reshaped)

# Make a prediction
prediction = classifier.predict(std_data)

if prediction[0] == 0:
    print('The Person is not diabetic')
else:
    print('The Person is Diabetic ')



# # Trying Other models 
# # 1. Naive Bayes 

# In[50]:


from sklearn.naive_bayes import GaussianNB

# Create a Gaussian Naive Bayes classifier
classifier = GaussianNB()

# Feed training data to the classifier and train the model
classifier.fit(X_train, Y_train)


# Model Evaluation

# In[51]:


# Model evaluation on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score on the training data: ', training_data_accuracy)


# In[52]:


# Model evaluation on the testing data
X_test_prediction = classifier.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score on the testing data: ', testing_data_accuracy)


# # 2. Random Forest

# In[53]:


from sklearn.ensemble import RandomForestClassifier

# Create a Random Forest Classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=0)

# Feed training data to the classifier and train the model
classifier.fit(X_train, Y_train)


# Model Evaluation
# 

# In[54]:


# Model evaluation on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score on the training data: ', training_data_accuracy)


# In[55]:


# Model evaluation on the testing data
X_test_prediction = classifier.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score on the testing data: ', testing_data_accuracy)


# # 3. Gradient Boosting Model

# In[56]:


from sklearn.ensemble import GradientBoostingClassifier

# Create a Gradient Boosting Classifier
classifier = GradientBoostingClassifier(n_estimators=100, random_state=0)

# Feed training data to the classifier and train the model
classifier.fit(X_train, Y_train)


# Model Evaluaion

# In[57]:


# Model evaluation on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score on the training data: ', training_data_accuracy)


# In[58]:


# Model evaluation on the testing data
X_test_prediction = classifier.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score on the testing data: ', testing_data_accuracy)


# # 4. XGBOOST

# In[61]:


pip install xgboost


# In[62]:


import xgboost as xgb

# Create a Gradient Boosting Classifier
classifier = xgb.XGBClassifier(n_estimators=100, random_state=0)

# Feed training data to the classifier and train the model
classifier.fit(X_train, Y_train)


# Model Evaluation

# In[64]:


# Model evaluation on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score on the training data: ', training_data_accuracy)


# In[65]:


# Model evaluation on the testing data
X_test_prediction = classifier.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score on the testing data: ', testing_data_accuracy)


# In[ ]:




