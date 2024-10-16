# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start the program
2. Import the necessary python packages
3. Read the dataset.
4. Define X and Y array.
5. Define a function for costFunction,cost and gradient.
6. Define a function to plot the decision boundary and predict the Regression value
7. End the program.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by:    SURESH S
RegisterNumber:  212223040215
*/
```

```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
```

```
dataset = pd.read_csv("/content/Placement_Data.csv")
dataset.head()
dataset.tail()
```
<img width="1250" alt="image" src="https://github.com/user-attachments/assets/be9aa1d4-7aad-44cf-b1b1-781248f7491c">

```
dataset.info()
```
<img width="650" alt="image" src="https://github.com/user-attachments/assets/8050d443-28db-40d6-a28f-dbead9cd589f">

```
dataset=dataset.drop(['sl_no'],axis=1)
```

```
dataset.info()
```
<img width="650" alt="image" src="https://github.com/user-attachments/assets/c926ae8f-5565-4bc8-ae9e-18009e2ff431">

```
dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
```

```
dataset.info()
```
<img width="650" alt="image" src="https://github.com/user-attachments/assets/059be377-510b-425a-a981-7f819f353342">

```
dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
```

```
dataset.info()
```
<img width="650" alt="image" src="https://github.com/user-attachments/assets/12cc71d2-3d2a-4690-acff-096ef58bf519">

```
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
```

```
theta=np.random.randn(x.shape[1])
y=Y
```

```
def sigmoid(z):
  return 1/(1+np.exp(-z))
def loss(theta,X,Y):
  n=sigmoid(x.dot(theta))
  return -np.sum(y*np.log(h) + (1-y) * np.log(1-h))
def gradient_descent(theta,X,y,alpha,num_iterations):
  m=len(y)
  for i in range(num_iterations):
    h=sigmoid(X.dot(theta))
    gradient=X.T.dot(h-y)/m
    theta -=alpha*gradient
  return theta
```

```
theta = gradient_descent(theta,x,y,alpha=0.01,num_iterations=1000)
```

```
def predict(theta, X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h >= 0.5, 1, 0)
    return y_pred
y_pred = predict(theta, X)
accuracy = np.mean(y_pred.flatten() == y)
print("Accuracy:", accuracy)
print(y_pred)
print(Y)
xnew = np.array([[0, 87, 0, 95, 0, 2, 78, 2, 0, 0, 1, 0]])
y_prednew = predict(theta, xnew)
print(y_prednew)
xnew = np.array([[0, 0, 0, 0, 0, 2, 8, 2, 0, 0, 1, 0]])
y_prednew = predict(theta, xnew)
print(y_prednew)
```
<img width="950" alt="image" src="https://github.com/user-attachments/assets/3c5d0ec6-cac5-4bf5-890b-7759652930c9">

```
print(theta)
```
<img width="950" alt="image" src="https://github.com/user-attachments/assets/4f42f527-0576-400f-a082-7ca0602c0664">



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.
