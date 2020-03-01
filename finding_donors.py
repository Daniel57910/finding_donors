# Import libraries necessary for this project
import numpy as np
import pandas as pd
from time import time
import os
import category_encoders as ce
import pdb
data = pd.read_csv(os.getcwd() + "/census.csv")

# Success - Display the first record
print(data.head(n=1))

over_50k = data.loc[data['income'] == '>50K']
under_50k = data.loc[data['income'] != '>50K']

n_records = len(data)
n_over_50k = len(over_50k)
n_under_50k = len(under_50k)
percentage_50k = n_over_50k / (n_over_50k + n_under_50k)

print(f'Number of record => {n_records}')
print(f'Number of earners over 50k => {n_over_50k}')
print(f'Number of earners under 50k => {n_under_50k}')
print(f'Ratio => {percentage_50k}')


income_raw = data['income']
features_raw = data.drop('income', axis = 1)

skewed = ['capital-gain', 'capital-loss']
features_log_transformed = pd.DataFrame(data = features_raw)
features_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))

# Visualize the new log distributions


from sklearn.preprocessing import MinMaxScaler

# Initialize a scaler, then apply it to the features
scaler = MinMaxScaler() # default=(0, 1)
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

features_log_minmax_transform = pd.DataFrame(data = features_log_transformed)
features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])

print(features_log_minmax_transform.head(n = 5))



features_final = pd.get_dummies(features_log_minmax_transform)
print(features_final)

income = income_raw.apply(lambda col: col == '<=50K')
income = pd.DataFrame(income)
encoded = list(features_final.columns)
print("{} total features after one-hot encoding.".format(len(encoded)))

# Uncomment the following line to see the encoded feature names
print('\n'.join(encoded))




from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features_final, 
                                                    income, 
                                                    test_size = 0.2, 
                                                    random_state = 0)

print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))


# ----


'''
TP = np.sum(income) # Counting the ones as this is the naive case. Note that 'income' is the 'income_raw' data 
encoded to numerical values done in the data preprocessing step.
FP = income.count() - TP # Specific to the naive case

TN = 0 # No predicted negatives in the naive case
FN = 0 # No predicted negatives in the naive case
'''
TP = len(income_raw.index)
FP = income_raw.count()
TN = 0
FN = 0

# TODO: Calculate accuracy, precision and recall
accuracy = TP + TN / (TP + TN + FP + FN)
recall = TP / (TP + FN)
precision = TP / (TP + FP)
beta = 0.5 * 0.5
beta_weight = 1 + beta
numerator = precision * recall
precision_weight = beta * precision

denominator = precision_weight + recall
fscore = (numerator / denominator) * beta_weight

# TODO: Calculate F-score using the formula above for beta = 0.5 and correct values for precision and recall.
print("Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore))

from sklearn.metrics import accuracy_score, fbeta_score

def train_predict(learner, sample_size, X_train, y_train, X_test, y_test): 

    PREDICTION_LIMIT = 300
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''
    
    results = {}
    
    # TODO: Fit the learner to the training data using slicing with 'sample_size' using .fit(training_features[:], training_labels[:])
    train_start = time() # Get start time
    print(y_train.columns)
    learner.fit(X_train[:sample_size], y_train[:sample_size])
    train_end = time() # Get end time
    
    # TODO: Calculate the training time
    results['train_time'] = int(train_end - train_start)
        
    # TODO: Get the predictions on the test set(X_test),
    #       then get predictions on the first 300 training samples(X_train) using .predict()
    pred_start = time() # Get start time
    predictions_test = learner.predict(X_test[:PREDICTION_LIMIT])
    predictions_train = learner.predict(X_train[:PREDICTION_LIMIT])
    pred_end = time() # Get end time
    
    # TODO: Calculate the total prediction time
    results['pred_time'] = int(pred_end - pred_start)
            
    # TODO: Compute accuracy on the first 300 training samples which is y_train[:300]
    results['acc_train'] = accuracy_score(y_train[:PREDICTION_LIMIT], predictions_train[:PREDICTION_LIMIT])
    # TODO: Compute accuracy on test set using accuracy_score()

    results['acc_test'] = accuracy_score(y_test[:PREDICTION_LIMIT], predictions_test[:PREDICTION_LIMIT])
    
    # TODO: Compute F-score on the the first 300 training samples using fbeta_score()
    results['f_train'] = fbeta_score(y_train[:PREDICTION_LIMIT], predictions_train[:PREDICTION_LIMIT], average='weighted', beta=0.3)
        
    # TODO: Compute F-score on the test set which is y_test
    results['f_test'] = fbeta_score(y_test[:PREDICTION_LIMIT], predictions_test[:PREDICTION_LIMIT], average='weighted', beta=0.3)
       
    # Success
    print("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))
        
    # Return the results
    return results


# TODO: Import the three supervised learning models from sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
# TODO: Initialize the three models
clf_A = DecisionTreeClassifier()

clf_B = RandomForestClassifier()

clf_C = AdaBoostClassifier()

samples_100 = len(y_train)
samples_10 = int(samples_100 / 10)
samples_1 = int(samples_10 / 10)

# Collect results on the learners
results = {}
# add in rest once sorted
for clf in [clf_A, clf_B, clf_C]:
    clf_name = clf.__class__.__name__
    print(clf_name)
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
      results[clf_name][i] =  train_predict(clf, samples, X_train, y_train, X_test, y_test)

# # # Run metrics visualization for the three supervised learning models chosen
# # vs.evaluate(results, accuracy, fscore)


# # ----
# # ## Improving Results
# # In this final section, you will choose from the three supervised learning models the *best* model to use on the student data. You will then perform a grid search optimization for the model over the entire training set (`X_train` and `y_train`) by tuning at least one parameter to improve upon the untuned model's F-score. 

# # ### Question 3 - Choosing the Best Model
# # 
# # * Based on the evaluation you performed earlier, in one to two paragraphs, explain to *CharityML* which of the three models you believe to be most appropriate for the task of identifying individuals that make more than \$50,000. 
# # 
# # ** HINT: ** 
# # Look at the graph at the bottom left from the cell above(the visualization created by `vs.evaluate(results, accuracy, fscore)`) and check the F score for the testing set when 100% of the training set is used. Which model has the highest score? Your answer should include discussion of the:
# # * metrics - F score on the testing when 100% of the training data is used, 
# # * prediction/training time
# # * the algorithm's suitability for the data.

# # **Answer: **

# # ### Question 4 - Describing the Model in Layman's Terms
# # 
# # * In one to two paragraphs, explain to *CharityML*, in layman's terms, how the final model chosen is supposed to work. Be sure that you are describing the major qualities of the model, such as how the model is trained and how the model makes a prediction. Avoid using advanced mathematical jargon, such as describing equations.
# # 
# # ** HINT: **
# # 
# # When explaining your model, if using external resources please include all citations.

# # **Answer: ** 

# # ### Implementation: Model Tuning
# # Fine tune the chosen model. Use grid search (`GridSearchCV`) with at least one important parameter tuned with at least 3 different values. You will need to use the entire training set for this. In the code cell below, you will need to implement the following:
# # - Import [`sklearn.grid_search.GridSearchCV`](http://scikit-learn.org/0.17/modules/generated/sklearn.grid_search.GridSearchCV.html) and [`sklearn.metrics.make_scorer`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html).
# # - Initialize the classifier you've chosen and store it in `clf`.
# #  - Set a `random_state` if one is available to the same state you set before.
# # - Create a dictionary of parameters you wish to tune for the chosen model.
# #  - Example: `parameters = {'parameter' : [list of values]}`.
# #  - **Note:** Avoid tuning the `max_features` parameter of your learner if that parameter is available!
# # - Use `make_scorer` to create an `fbeta_score` scoring object (with $\beta = 0.5$).
# # - Perform grid search on the classifier `clf` using the `'scorer'`, and store it in `grid_obj`.
# # - Fit the grid search object to the training data (`X_train`, `y_train`), and store it in `grid_fit`.
# # 
# # **Note:** Depending on the algorithm chosen and the parameter list, the following implementation may take some time to run!

# # In[ ]:


# # TODO: Import 'GridSearchCV', 'make_scorer', and any other necessary libraries

# # TODO: Initialize the classifier
# clf = None

# # TODO: Create the parameters list you wish to tune, using a dictionary if needed.
# # HINT: parameters = {'parameter_1': [value1, value2], 'parameter_2': [value1, value2]}
# parameters = None

# # TODO: Make an fbeta_score scoring object using make_scorer()
# scorer = None

# # TODO: Perform grid search on the classifier using 'scorer' as the scoring method using GridSearchCV()
# grid_obj = None

# # TODO: Fit the grid search object to the training data and find the optimal parameters using fit()
# grid_fit = None

# # Get the estimator
# best_clf = grid_fit.best_estimator_

# # Make predictions using the unoptimized and model
# predictions = (clf.fit(X_train, y_train)).predict(X_test)
# best_predictions = best_clf.predict(X_test)

# # Report the before-and-afterscores
# print("Unoptimized model\n------")
# print("Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions)))
# print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta = 0.5)))
# print("\nOptimized Model\n------")
# print("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
# print("Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5)))


# # ### Question 5 - Final Model Evaluation
# # 
# # * What is your optimized model's accuracy and F-score on the testing data? 
# # * Are these scores better or worse than the unoptimized model? 
# # * How do the results from your optimized model compare to the naive predictor benchmarks you found earlier in **Question 1**?_  
# # 
# # **Note:** Fill in the table below with your results, and then provide discussion in the **Answer** box.

# # #### Results:
# # 
# # |     Metric     | Unoptimized Model | Optimized Model |
# # | :------------: | :---------------: | :-------------: | 
# # | Accuracy Score |                   |                 |
# # | F-score        |                   |   EXAMPLE       |
# # 

# # **Answer: **

# # ----
# # ## Feature Importance
# # 
# # An important task when performing supervised learning on a dataset like the census data we study here is determining which features provide the most predictive power. By focusing on the relationship between only a few crucial features and the target label we simplify our understanding of the phenomenon, which is most always a useful thing to do. In the case of this project, that means we wish to identify a small number of features that most strongly predict whether an individual makes at most or more than \$50,000.
# # 
# # Choose a scikit-learn classifier (e.g., adaboost, random forests) that has a `feature_importance_` attribute, which is a function that ranks the importance of features according to the chosen classifier.  In the next python cell fit this classifier to training set and use this attribute to determine the top 5 most important features for the census dataset.

# # ### Question 6 - Feature Relevance Observation
# # When **Exploring the Data**, it was shown there are thirteen available features for each individual on record in the census data. Of these thirteen records, which five features do you believe to be most important for prediction, and in what order would you rank them and why?

# # **Answer:**

# # ### Implementation - Extracting Feature Importance
# # Choose a `scikit-learn` supervised learning algorithm that has a `feature_importance_` attribute availble for it. This attribute is a function that ranks the importance of each feature when making predictions based on the chosen algorithm.
# # 
# # In the code cell below, you will need to implement the following:
# #  - Import a supervised learning model from sklearn if it is different from the three used earlier.
# #  - Train the supervised model on the entire training set.
# #  - Extract the feature importances using `'.feature_importances_'`.

# # In[ ]:


# # TODO: Import a supervised learning model that has 'feature_importances_'


# # TODO: Train the supervised model on the training set using .fit(X_train, y_train)
# model = None

# # TODO: Extract the feature importances using .feature_importances_ 
# importances = None

# # Plot
# vs.feature_plot(importances, X_train, y_train)


# # ### Question 7 - Extracting Feature Importance
# # 
# # Observe the visualization created above which prints the five most relevant features for predicting if an individual makes at most or above \$50,000.  
# # * How do these five features compare to the five features you discussed in **Question 6**?
# # * If you were close to the same answer, how does this visualization confirm your thoughts? 
# # * If you were not close, why do you think these features are more relevant?

# # **Answer:**

# # ### Feature Selection
# # How does a model perform if we only use a subset of all the available features in the data? With less features required to train, the expectation is that training and prediction time is much lower — at the cost of performance metrics. From the visualization above, we see that the top five most important features contribute more than half of the importance of **all** features present in the data. This hints that we can attempt to *reduce the feature space* and simplify the information required for the model to learn. The code cell below will use the same optimized model you found earlier, and train it on the same training set *with only the top five important features*. 

# # In[ ]:


# # Import functionality for cloning a model
# from sklearn.base import clone

# # Reduce the feature space
# X_train_reduced = X_train[X_train.columns.values[(np.argsort(importances)[::-1])[:5]]]
# X_test_reduced = X_test[X_test.columns.values[(np.argsort(importances)[::-1])[:5]]]

# # Train on the "best" model found from grid search earlier
# clf = (clone(best_clf)).fit(X_train_reduced, y_train)

# # Make new predictions
# reduced_predictions = clf.predict(X_test_reduced)

# # Report scores from the final model using both versions of data
# print("Final Model trained on full data\n------")
# print("Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
# print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5)))
# print("\nFinal Model trained on reduced data\n------")
# print("Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, reduced_predictions)))
# print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, reduced_predictions, beta = 0.5)))


# # ### Question 8 - Effects of Feature Selection
# # 
# # * How does the final model's F-score and accuracy score on the reduced data using only five features compare to those same scores when all features are used?
# # * If training time was a factor, would you consider using the reduced data as your training set?

# # **Answer:**

# # > **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  
# # **File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.

# # ## Before You Submit
# # You will also need run the following in order to convert the Jupyter notebook into HTML, so that your submission will include both files.

# # In[ ]:


# get_ipython().getoutput('jupyter nbconvert *.ipynb')

