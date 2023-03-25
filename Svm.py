from sklearn.neighbors import KNeighborsClassifier 
from sklearn import svm, datasets
import pandas as pd
import csv
import numpy as np
import time,datetime
#read data from train.csv
df = pd.read_csv("train.csv")
print(df)
#input contain all col except last i.e types
inputs = df.drop(['ID Number','Type of Glass'],axis='columns')
#print(inputs)
target = df.drop(['ID Number','Refractive Index',	'Sodium',	'Magnesium','Aluminium','Silicon','Potassium','Calcium','Barium','Iron'],axis='columns')
#Importing Libraries to be used in this program#in order to measure time , signalizing the starting point to note time
start = datetime.datetime.now()
#Train model through builtin and the purpose of ravel is to convert 2D array in to 1D
linear = svm.SVC(kernel='linear', C=1, decision_function_shape='ovo')
linear.fit(inputs.values, target.values.ravel())
#in order to measure time , signalizing the ending point to note time
end = datetime.datetime.now()
#taking difference
difference = (end - start)
#calculating time in miliseconds
milisecond = int(difference.total_seconds() * 10000)
print('Time to train: {} Milliseconds'.format(milisecond))
#initializing array for test
validation= []
d = []
error = []
prediction_list =[]
with open('test.csv', 'r') as csvfile:
        next(csvfile)
        for row in csv.reader(csvfile):
            d.append(row)
        validation = (np.array(d)[:,1:-1]).tolist()
        error = (np.array(d)[:,-1]).tolist()
count=0
total = len(error)
#in order to measure time , signalizing the starting point to note time
start2 = datetime.datetime.now()
for i in validation:
    prediction = linear.predict([i])
    prediction_list.append(prediction)
#in order to measure time , signalizing the ending point to note time
end2 = datetime.datetime.now()
#taking difference
difference2 = (end2 - start2)
#calculating time in miliseconds
m_seconds = int(difference2.total_seconds() * 10000)
print('Time to test: {} Milliseconds'.format(m_seconds ))
dff = pd.DataFrame(prediction_list, columns = ['output'])
pdtolist = list(dff['output'])
for i ,j in zip(pdtolist,error):
    if int(i) == int(j):
       count = count+1
print('-------------RESULTS(FA19-BCS-027)-----------------')
percentage = (count/total)*100
error = 100 - (count/total)*100
print('Correct Predictions: {} out of {}'.format(count,total))
print('Accuracy: ',round(percentage,2,),'%')
print('Error: ',round(error,2),'%')