# salary-years-simple-linear-regression
Simple linear regression model implementation in Python and R for to showing the relationship between salary and years of experience.

############################################################################################
# Simple linear Regression in Python
############################################################################################

# Importing libraries
import numpy as np    
import matplotlib.pyplot as plt   
import pandas as pd

# Importing dataset
dataset = pd.read_csv('Salary_Data.csv')  
X = dataset.iloc[:, :-1].values   
y = dataset.iloc[:, 1].values   

# Splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)    

# Applying simple linear regression

from sklearn.linear_model import LinearRegression   
regressor = LinearRegression()    
regressor.fit(X_train, y_train)   

y_pred = regressor.predict(X_test)    

# Visualizing Training set results
plt.scatter(X_train, y_train, color = 'red')    
plt.plot(X_train, regressor.predict(X_train), color = 'blue')   
plt.title('Salary vs Experience {Training set}')    
plt.xlabel('Years of experience')   
plt.ylabel('Salary')
plt.show()

# Visualizing Test set results
plt.scatter(X_test, y_test, color = 'red')    
plt.plot(X_train, regressor.predict(X_train), color = 'blue')   
plt.title('Salary vs Experience {Test set}')    
plt.xlabel('Years of experience')   
plt.ylabel('Salary')    
plt.show()    

############################################################################################
# Simple linear Regression in R
############################################################################################

# Importing dataset
dataset = read.csv('Salary_Data.csv')   

# Splitting the dataset into the training set and test set
#install.packages('caTools')    
library(caTools)    
set.seed(123)   
split = sample.split(dataset$Salary, SplitRatio = 2/3)    
training_set = subset(dataset, split == TRUE)   
test_set = subset(dataset, split == FALSE)    

# Applying simple linear regression
regressor = lm(formula = Salary ~ YearsExperience,    
               data = training_set)   
 y_pred = predict(regressor, newdata = test_set)    
 
# Visualising the Training set results
#install.packages('ggplot2')    
library(ggplot2)    

ggplot() +    
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),    
             colour = 'red') +    
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),    
            colour = 'blue') +    
  ggtitle('Salary vs Experience (Training set)') +    
  xlab('Years of experience') +   
  ylab('Salary')    

# Visualizing Test set results
ggplot() +
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary),    
             colour = 'red') +    
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),    
            colour = 'blue') +    
  geom_smooth(method = 'lm') +      
  ggtitle('Salary vs Experience (Test set)') +    
  xlab('Years of experience') +   
  ylab('Salary')    
