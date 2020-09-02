# -*- coding: utf-8 -*-
#240201043 ÖYKÜ ANDAÇ
import pandas as pd
import math
import operator
import matplotlib.pyplot as plt


covid = pd.read_csv("covid.csv")
df = pd.DataFrame(covid)


test1 = [[5,39.0,1], [4,35.0,0], [3,38.0,0], [2,39.0,1], [1,35.0,0],
[0,36.2,0], [5,39.0,1], [2,35.0,0], [3,38.9,1], [0,35.6,0]]

cough_list = df['cough_level'].tolist() #cough attribute to list
fever_list = df['fever'].tolist() #fever attribute to list
def min_max_norm(x,y): #normalization of train data
    
    for i in range(len(x)):
        normalized_x = (x[i]-min(x))/(max(x)-min(x))
        df['cough_level'].replace(x[i], normalized_x, inplace=True) #replace df values with normalized values
        
    for i in range(len(y)):
        normalized_y = (y[i]-min(y))/(max(y)-min(y))
        df['fever'].replace(y[i], normalized_y, inplace=True) #replace df values with normalized values
        
min_max_norm(cough_list, fever_list)

def normalize_test(test): #normalization of test data
    cough = [] 
    fever = [] 
    for i in range(len(test)):
        cough.append(test[i][0]) #cough attribute to list
        fever.append(test[i][1]) #fever attribute to list
    
    for i in range(len(cough)):
        normalized_x = (cough[i]-min(cough))/(max(cough)-min(cough))
        test[i][0] = normalized_x
        
    for i in range(len(fever)):
       normalized_y = (fever[i]-min(fever))/(max(fever)-min(fever))
       test[i][1] = normalized_y  
normalize_test(test1)    
 
def create_trainset(): 
    train = []
    for i in range(len(df)):
        temp = []
        temp.append(df['cough_level'][i])
        temp.append(df['fever'][i])
        temp.append(df['target'][i])
        train.append(temp) #create train set according to each row
    temp.clear()
    train.remove(train[len(df)-1])
    return train

   
def euclidean_distance(list1,list2):
    
    distance = 0
    for i in range(len(list1)-1):
        distance = distance + (list1[i] - list2[i])**2 #euclidean distance formula
    
    return math.sqrt(distance)

def manhattan_distance(list1,list2):
    
    distance = 0
    for i in range(len(list1)):
        distance = distance + (list1[i] - list2[i]) #manhattan distance formula
    
    return distance

   
def knn(train_data, test_data, distance_type, k):
    dist_list = []
    neighbors = []
    targets = []
    predict = []
    dist = 0
    for i in train_data: #in this for loop, I checked distance type as string and calculate the distance
        if (distance_type == "euclidean"):
            dist = euclidean_distance(test_data,i)
        elif (distance_type == "manhattan"):
            dist = manhattan_distance(test_data,i)
            
        dist_list.append((i,dist)) #I added distances with related train data as a tuple
    dist_list.sort(key=operator.itemgetter(1)) #to sort distances of tuple
    for i in range(k): #takes first k element of sorted dist list
        neighbors.append(dist_list[i])
       
    for i in range(len(neighbors)):
        
        targets.append(neighbors[i][0][2]) #take target values of neighbors
        
    count1 = targets.count(1) #number of 1's in targets
    count0 = targets.count(0) #number of 0's in targets
    if (count1 > count0): 
        predict.append(1)      
    else:
        predict.append(0)
    return predict
     

train = create_trainset()

def call_knn(test, k): #to call knn with different k values
    predicts = []
    for i in range(len(test)):
    
        p = knn(train,test[i],"euclidean",k)
        predicts.append(p)
    return predicts #return each k value's prediction list

def accuracy(test,prediction): 
    true = 0
    for i in range(len(test)):
        if (test[i][2] == prediction[i][0]):        
            true += 1
    return (true/float(len(test))) * 100.0   #calculates accuracy 
    
def plot(test):
    k_values = [0, 5 ,10 ,15, 20, 25]  
    accuracies = []
    for k in k_values:
        a = accuracy(test, call_knn(test, k))
        accuracies.append(a) #accuracy values of each k value
    plt.plot(k_values, accuracies)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Testing Accuracy')

plot(test1)
#According to the accuracy graph, the best value of k is 5,10,15,20,25 
