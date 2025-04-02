# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd 
import numpy  as np
import os 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report



wine_names = [
    'Class',
    'Alcohol',
 	'Malic acid',
 	'Ash',
	'Alcalinity of ash',  
 	'Magnesium',
	'Total phenols',
 	'Flavanoids',
 	'Nonflavanoid phenols',
 	'Proanthocyanins',
	'Color intensity',
 	'Hue',
 	'OD280/OD315 of diluted wines',
 	'Proline'

]
            


os.chdir('C:\\Users\\semih\\.spyder-py3\\winedata')

winedf= pd.read_csv('wine.data',header=None)
winedf.columns= wine_names

print(winedf.head())


print('***********---Checking for missing values---***********')
print(winedf.isnull().sum())  # Check for missing values
print('********---Checking for missing values is over---********')

# Step 2: Split into features and target
X = winedf.drop('Class', axis=1)  # Features
y = winedf['Class']               # Target

#standarizing the features 
scaler = StandardScaler() #creating the instance of StandartScaler. St features by removing the meand and scaling to unit var
X_scaled = scaler.fit_transform(X) # Compute mean and standart dev. And subtract th mean an scale to unit variance

# Step 4: Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Optional: check the shape of the splits
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
""""
plt.rcParams['figure.figsize']=(30,25)
winedf.plot(kind='hist',bins=200,subplots=True,layout=(7,2),sharex=False,sharey=False)
plt.show()

correlation=winedf.corr()
correlation['Class'].sort_values(ascending=False)

plt.figure(figsize=(10,8))
plt.title('Correlation of Attributes with Class variable')
a = sns.heatmap(correlation, square=True, annot=True, fmt='.2f', linecolor='white')
a.set_xticklabels(a.get_xticklabels(), rotation=90)
a.set_yticklabels(a.get_yticklabels(), rotation=30)           
plt.show()
"""

def euclidiandist(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))
    
def manhattandist(test,train):
    return np.sum(np.abs(test-train))


def kNearestNeighborsfunc(X_train,X_test,Y_train,k):
    
    
    predictions = []

    for testsample in X_test:

        distancevals = [manhattandist(testsample,traindatas) for traindatas in X_train]    #find the distances of all neighbors between every test data with train data
        kNneighbors_indices=np.argsort(distancevals)[:k]    # hold the indices of all nearest neighbors 
        kNlabels = Y_train.iloc[kNneighbors_indices]  # get the class of k nearest neighbors 
        
    
        predicted_label = Counter(kNlabels).most_common(1)[0][0]    
        predictions.append(predicted_label)
    
    return predictions

acc=[]
kvals=range(1,50,2)

for i in kvals:
    pred=[]
    pred=kNearestNeighborsfunc(X_train,X_test,y_train,i)
    acc.append(accuracy_score(pred, y_test))

    cm=confusion_matrix(pred,y_test)
    cr=classification_report(pred,y_test)
    
    print(f"Classification Report for k = {i}:\n")
    print(cr)

    # Plot confusion matrix using seaborn heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.title(f'Confusion Matrix for k = {i}')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.show()



# Plot Accuracy vs. k
plt.figure(figsize=(8, 6))
plt.plot(kvals, acc, marker='o', linestyle='-')
plt.title('Accuracy vs. k in k-NN')
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()




