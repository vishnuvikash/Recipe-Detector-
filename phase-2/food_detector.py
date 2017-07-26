from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from scipy.sparse import csr_matrix

import json
import codecs
import pandas as pd
import numpy as np
import time

start_time = time.time()
meal_id,cuisine,ingredients,ing,main_set,train_set,test_set =[],[],[],[],[],[],[]
predicted_cuisine = ''

#creates different lists needed in the program
def lists_creater(filename):
    with codecs.open( filename,encoding = 'utf-8') as f:
        data = json.load(f)
        
    for i in range(0,len(data)):
        meal_id.append(data[i]["id"])
        cuisine.append(data[i]["cuisine"])
        ingredients.append(data[i]["ingredients"])
        
    for i in ingredients:
        temp =u''
        for f in range(len(i)):
            temp = temp+u" "+i[f]
        ing.append(temp.encode('utf-8'))
    #print(ing)
        
    return meal_id
    return cuisine
    return ingredients
    return ing

#vectorizes the document and converts them into features   
def ing_vectorizer(exis_ing,user_ing):
    exis_ing.append(user_ing)
    vectorizer = TfidfVectorizer(use_idf = True, stop_words = 'english',max_features = 4000)
    #X=np.array(exis_ing)
    #X=X.reshape(1,-1)
    ing_vect = vectorizer.fit_transform(exis_ing)
    return (ing_vect.todense())

#creates train set(existing data from json file) and test set (user entered ingredient)
def set_creator(main_set):
    train_set = main_set[:len(main_set)-1]
    test_set = main_set[len(main_set)-1]
    return (train_set,test_set)

#trains the model with the json data for future prediction
def KNN_trainer(train_set,cuisine,n):  
    n = int(n)
    #train_set=train_set(1,-1)
    #X=np.array(train_set)
    #X=X.shape
    close_n = KNeighborsClassifier(n_neighbors=n)
    return close_n.fit(train_set,cuisine)

#user entered ingredient is given to the model to predict cuisine and return n nearest neighbors
def KNN_predictor(test_set,close_n,no_of_neigh):
    no_of_neigh = int(no_of_neigh)
    print ("")
    #test_set=test_set(1,-1)
    #list of probabilities of Top N Closest Foods
    predicted_cuisine = close_n.predict_proba(test_set)[0]
    #Top most matched Cuisine among Top N CLosest Foods
    predicted_single_cuisine = close_n.predict(test_set)
    #List of predicted Cuisines of Top N Closest Foods
    predicted_class = close_n.classes_
    print ("The model predicts that the ingredients resembles %s" %(predicted_single_cuisine[0]))
    print ("")
    for i in range(len(predicted_cuisine)):
        if not(predicted_cuisine[i] == 0.0):
            print ("The ingredients resemble %s with %f percentage" %(predicted_class[i],predicted_cuisine[i]*100))
    
    print ("")
    print ("The %d closest meals are listed below : " % no_of_neigh)
    match_perc,match_id = close_n.kneighbors(test_set)
    for i in range(len(match_id[0])):
        print (meal_id[match_id[0][i]])
        #print (ingredients[match_id[0][i]])
    print ("")
    
    print("--- It took %s seconds ---" %(time.time() - start_time))
    print ("")
    print ("")
    return predicted_single_cuisine

#handles the sequential execution of the program
def seq_exec():
    user_ing = input("Enter the ingredients that you want to compare : ")
    main_set = ing_vectorizer(ing,user_ing)
    train_set,test_set = set_creator(main_set)
    no_of_neigh = input("Enter the number of closest items you want to find : ")
    close_n = KNN_trainer(train_set,cuisine,no_of_neigh)
    print ("Model has been successfully trained..")
    print ("Trying to predict the cuisine and n closest meal items...")
    KNN_predictor(test_set,close_n,no_of_neigh)
    ing.pop()
    try:
        nextStep = int(input("Enter 1 if you want to search again or 2 if you want to quit.."))
        if not(nextStep == 1 or nextStep == 2):
            raise ValueError()
        elif (nextStep == 1):
            seq_exec()
        elif (nextStep == 2):
            quit()
    except ValueError:
        print ("Invalid Option. Enter correctly")
        seq_exec()
        
if __name__ == '__main__':
    print ("Reading all the data files and creating lists....")
    lists_creater(filename = "data/yummly.json")
    seq_exec()
    
    

