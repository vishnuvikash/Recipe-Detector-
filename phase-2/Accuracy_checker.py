import food_detector
import random

per_test = []
per_test_id = []
per_train = []
per_train_id = []
per_train_cuisine = []
per_test_cuisine = []

print ("started")
food_detector.lists_creater(filename = "data/yummly.json")
full_list = food_detector.ing_vectorizer(food_detector.ing,"wheat sugar")
print ("list completed")
for i in range(0,len(full_list)-1):
    if random.random() < 0.95:
        per_train.append(full_list[i])
        per_train_id.append(food_detector.meal_id[i])
        per_train_cuisine.append(food_detector.cuisine[i])
    else:
        per_test.append(full_list[i])
        per_test_id.append(food_detector.meal_id[i])
        per_test_cuisine.append(food_detector.cuisine[i])

print("Training started..")
per_close_n = food_detector.KNN_trainer(per_train,per_train_cuisine,5)

print("training done")
no_of_tests = len(per_test)
no_of_passes = 0

print("testing started..")
print (no_of_tests)
for i in range(0,len(per_test)):
    print (i)
    per_cuisine = food_detector.KNN_predictor(per_test[i],per_close_n,5)
    print(per_cuisine,per_test_cuisine[i])
    if per_cuisine == per_test_cuisine[i]:
        no_of_passes += 1
        
print("Accuracy percentage %d" %((no_of_passes/no_of_tests)*100))
    


