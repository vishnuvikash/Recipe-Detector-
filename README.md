**PROJECT-2 IN CLASS PHASE 1-- PART A**

As part of the project we are provided with the https://www.dropbox.com/s/f0tduqyvgfuin3l/yummly.json?dl=0 YUMMLY DATA SET.

We are given the master list of all possible dishes, their ingredients, an identifier, and the cuisine for thousands of different dishes.

We have to present a display of clustered ingredients and train a classifier to predict the cuisine type of a new food.

The phase1 of the project involves 2 tasks:
--------------------------------------------------------------------------
TASK1: CLUSTERING THE CUISINES BASED ON THEIR INGREDIENTS

In the dataset, we include the recipe id, the type of cuisine, and the list of ingredients of each recipe (of variable length). The data is stored in JSON format.

An example of a recipe node in yummly.json:

 {
 "id": 24717,
 "cuisine": "indian",
 "ingredients": [
     "tumeric",
     "vegetable stock",
     "tomatoes",
     "garam masala",
     "naan",
     "red lentils",
     "red chili peppers",
     "onions",
     "spinach",
     "sweet potatoes"
 ]
 }

STEP1:
I converted the JSON to a countsMatrix[i,j] where each row in countsmatrix represent a recipe and each column denotes a unique ingredient in the entire list of ingredients.

countsmatrix[i,j] denotes the count of occurence of ingredient j in cuisine i in the data set.

STEP2:
Generate Td-idf matrix from the count matrix and reduce the components to 2 using PCA analysis

For Tf-idf Vectorizer I have used the sklearn package:
from sklearn.feature_extraction.text import TfidfTransformer

For PCA i have used the Sklearn package:
from sklearn.decomposition import PCA

The reduced data after fitting through PCA is used in CLustering the data.

STEP 3:
For Clustering I have taken the KMeans Clustering Algorithm and clustered the data into three clusters.

I have Considered five clusters for the project.



---------------------------------------------------------------------------
TASK2: VISUALIZING THE CLUSTERS

For Visualizing the clusters I have used the PYLAB and MATPLOTLIB packages

from pylab import *
from scipy import *
import matplotlib.pyplot as plt

Effect on the cluster has been used in caluculating the size of the clusters. The sixe of the clusters is based on the effect of the clusters,JACCARD SIMILARITY (one vs the other cuisines in its cluster) We intersect all ingredients of cuisine i with the union of ingredients in all OTHER cuisines in its CLUSTER! (e.g.: intersect(filipino_Ingredients, other_asian_Ingredients) and divide by the union of all ingredients in the cluster.

I have created spheres of size relative to the times that it occurs within a cluster The size of the bubble is determined by the within-cluster similarity which is obtained by the Jaccard similarity of each cuisine with other members in it its cluster (computed above).

**PROJECT-2 IN CLASS PHASE 2---PART B**

--------------------------------------------------------------------------

Creates the list of given different data types such as list for meal_ids, cuisines, ingredients for the each of the meal and unicode format of ingredients of each of the meal

lists_created:

meal_id
cuisine
ingredients
ing

Takes input of the Ingredients from the user and appends to the "ing" list

Vectorizes the ing list and converts them into features, used Tfidf Vectorizer.

The Transformed data from the data set is split into train set and test set.
train_Set=ing[:len(ing)-1]
test_Set-ing[len(ing)-1] which is the user input of ingredients

---Train the Model

Take the input of n closest foods from the user and train the model using KNeighbours Classifier.

close_n = KNeighborsClassifier(n_neighbors=n)
return close_n.fit(train_set,cuisine)

---Predict the Cuisine and Top N closest foods
For the Top N Closest Foods used the advantage of KNeighborsClassifier probability predictor function
predicted_cuisine = close_n.predict_proba(test_set)[0] which gives the probabilities in Decreasing Order

For Identifying the Most Common Cuisine among the N closest foods
predicted_single_cuisine = close_n.predict(test_set)
most_common_Cuisine=predicted_single_cuisine[0]

For Cuisine Types of the Top N Closest foods
#List of predicted Cuisines of Top N Closest Foods
predicted_class = close_n.classes_
#list of probabilities of Top N Closest Foods
predicted_cuisine = close_n.predict_proba(test_set)[0]



For meal ids of the top N Closest Foods with matched ingredients
match_perc,match_id = close_n.kneighbors(test_set)
   for i in range(len(match_id[0])):
       print (meal_id[match_id[0][i]])

--Accuracy Checker

I have also included the accuracy checker where I have trained on 95% train data and checked on 5% test data the accuracy found to be 72%.


ASSUMPTIONS :
1) Accepts user input as space strings. Input should not be given in any other formats
2) User specifies n closest neighbours every time they run the model.
3) Since this is a lazy ML approach every time the user runs the model, based on n size it trains again, leading to a high wait time to see the results

DESIGN DECISIONS FOR THE MODEL:
1) Supervised Machine Learning is used instead of a unsupervised ML like clustering. Since the yummly data set already had meal information along with
the cuisine tag.
2) A K Nearest Neighbour classifier is used and the model is trained with the entire dataset from yummly
3) Due to memory issues encountered during testing, the model is made as a lazy ML. The model is trained only after the user gives input.
4) The data set has 20 cuisines. Testing (Accuracy_checker.py) showed that the model had an accuracy of 72 % when the yummly data set is split
as 95% train set and 5% test set with 5 nearest neighbours. So the model now shows the predicted cuisine as well as the probabilities of all non - zero cuisines
to account for alpha and beta errors

FILES NEEDED :
1) yummly.json is in the data directory(specified in the script-no need to give externally)
2) food_detector.py that predicts the cuisine and n nearest meals
3) Accuracy_checker.py (runs with 5 nearest neighbors and 95% - train data and 5 % - test data selected randomnly from the yummly data set)
that uses functions in the food_detector package and checks it's performance. (Warning : Running time is very long for this script)


EXECUTABLE METHOD: python3 AnalyzingFood.py -PHASE ONE  CLUSTERING AND VISUALIZATION
pyhton3 food_detector.py -PHASE 2







---------------------------------------------------------------------------------------
**OUTPUT**
vishnu@vishnu-Inspiron-5537:~/Desktop/AnalyzingFood/AnalyzingFood$ python3 AnalyzingFood.py 
vishnu@vishnu-Inspiron-5537:~/Desktop/AnalyzingFood/AnalyzingFood$ python3 food_detector.py
Reading all the data files and creating lists....
Enter the ingredients that you want to compare : eggs sugar
Enter the number of closest items you want to find : 6
Model has been successfully trained..
Trying to predict the cuisine and n closest meal items...

The model predicts that the ingredients resembles french

The ingredients resemble brazilian with 16.666667 percentage
The ingredients resemble french with 33.333333 percentage
The ingredients resemble italian with 16.666667 percentage
The ingredients resemble mexican with 16.666667 percentage
The ingredients resemble russian with 16.666667 percentage

The 6 closest meals are listed below :
48995
38948
2399
42427
11897
34440

--- It took 37.185532569885254 seconds ---


Enter 1 if you want to search again or 2 if you want to quit..1
Enter the ingredients that you want to compare : wheat sugar eggs
Enter the number of closest items you want to find : 5
Model has been successfully trained..
Trying to predict the cuisine and n closest meal items...

The model predicts that the ingredients resembles indian

The ingredients resemble french with 20.000000 percentage
The ingredients resemble indian with 60.000000 percentage
The ingredients resemble italian with 20.000000 percentage

The 5 closest meals are listed below :
5366
22213
22463
30385
11251

--- It took 92.15953874588013 seconds ---


Enter 1 if you want to search again or 2 if you want to quit..2
vishnu@vishnu-Inspiron-5537:~/Desktop/AnalyzingFood/AnalyzingFood$


>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
REQUIRED:

DEPENDENCIES:
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from scipy.sparse import csr_matrix

import json
import codecs
import pandas as pd
import numpy as np
import time

VERSION:
PYTHON VERSION 3.5+ preferred ,any version above PY(3.0) have to work.


<REFERENCES:>
YUMMLY DATA SET-     https://www.dropbox.com/s/f0tduqyvgfuin3l/yummly.json?dl=0

Sorting List using Itemgetter- http://stackoverflow.com/questions/10695139/sort-a-list-of-tuples-by-2nd-item-integer-value
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
