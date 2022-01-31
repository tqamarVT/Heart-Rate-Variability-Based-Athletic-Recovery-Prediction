"""
This python file reads in a data-base of cardiovascular and physical training metrics
specific to this project from a CSV file and applies various learning techniques to 
the data in order to predict two separate objectives related to the cardiovascular and training metrics,
the "workout rating" objective and the "strength/time improvement" objective. After splitting the data-set
read in into separate training and test data-sets, implementing the learning techniques on the training
data-set, and making predictions on the test data-set, results of the percentage accuracy of all
classifiers in predicting the two objectives for the test data-set are displayed both graphically and numerically. 

Created on Mon Nov 16 13:16:02 2020
@author: Taimoor Qamar
"""
import pandas as pd
import math
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
'''
PLEASE CHANGE THE "path_name" VARIABLE BELOW TO MATCH THE PATH NAME OF THE PATH LOCATION WHERE THE CSV FILE CONTAINING THE DATA
IS LOCATED ON YOUR COMPUTER
'''
path_name = r'C:\Users\Timmy Boy\OneDrive\Desktop\Virginia Tech, Computer Engineering\Adv ML\Project\Code\\'
##################################################################################################################################################################
'''
This method reads in a csv file/data-set containing cardio and training data
specific to this project, eliminates un-necessary data (data not used as features or labels), 
splits the data-set into training and test data via a 70/30 split respectively, scales the features 
in the newly split train and test data-sets, and returns the train and test data-sets as well as the 
respective labels for each data-set. It also returns an unaltered version of the original data-set.
@ parameters: 
        path_name: The name of the path where the csv file / data-set is located
        file_name: The name of the csv file / data-set
@ returns:
        master: An unaltered version of the original data-set as it was read in 
        train: The training data-set which is a 70% split of the original data-set
        test: The test data-set which is a 30% split of the original data-set   
        trainY1: The first set of labels, for workout rating, of the training data-set
        trainY2: The second st of labels, for strength/time improvement, of the training data-set
        testY1: The first set of labels, for workout rating, of the test data-set
        testY2: The second set of labels, for strength/time improvement, of the test data-set
'''
def format_data(path_name, file_name):
    df = pd.read_csv(path_name + file_name)
    master = df.copy()
    split_index = math.ceil(len(df)*0.7)
    end_index = len(df)    
    df = df.drop(['DATE', 'DAY', 'Avg Night HRV', 'Total Volume', 'Temp Deviation', 'Respiratory Rate', 'Notes'], axis = 1)
    train = df.iloc[1:split_index, :]
    trainY1 = train['Workout Rating (Bad - 1/ Moderate - 2 / Good - 3)']
    trainY2 = train['Strength or Time Improvement'] 
    train = train.drop(['Workout Rating (Bad - 1/ Moderate - 2 / Good - 3)', 'Strength or Time Improvement'], axis = 1) 
    test = df.iloc[split_index:end_index, :]
    testY1 = test['Workout Rating (Bad - 1/ Moderate - 2 / Good - 3)'] 
    testY2 = test['Strength or Time Improvement'] 
    test = test.drop(['Workout Rating (Bad - 1/ Moderate - 2 / Good - 3)', 'Strength or Time Improvement'], axis = 1)
    scaler = StandardScaler()
    scaler.fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)
    return master, train, trainY1, trainY2, test, testY1, testY2
###################################################################################################################################################################
'''
This method takes in two arrays containing the percentage accuracy values for all classifiers
for both the 'workout rating' and 'strength/time improvement' objectives and creates a horizontal bar
plot comparing the different classifiers' accuracies for both objectives. The two bar plots created are
saved as png files in the directory identified by the path_name variable at the top. This method also
prints a table of the different classifiers' accuracies for both of the classification objectives.
@ parameters:
    Y1_accuracy: List of percentage accuracy values for each classifier for the 'workout rating' objective
    Y2_accuracy: List of percentage accuracy values for each classifier for the 'strength/time improvement' objective
'''
def display_results(Y1_accuracy, Y2_accuracy):
    labels = ['Baseline', 'Random Forest Classifier', 'Multi-layer Perceptron', 'Support Vector Machine', 'Quadratic Discriminant Analysis', 'Stacked Gradient']
    Y1_acc = pd.DataFrame({'PERCENT ACCURACY' : Y1_accuracy})
    Y2_acc = pd.DataFrame({'PERCENT ACCURACY' : Y2_accuracy})
    Y1_acc['CLASSIFIER'] = labels
    Y2_acc['CLASSIFIER'] = labels
    Y1_acc = Y1_acc.sort_values(by = 'PERCENT ACCURACY')
    Y2_acc = Y2_acc.sort_values(by ='PERCENT ACCURACY')
    ax1 = Y1_acc.plot.barh(x = 'CLASSIFIER', y = 'PERCENT ACCURACY', color = 'red')
    ax2 = Y2_acc.plot.barh(x = 'CLASSIFIER', y = 'PERCENT ACCURACY', color = 'red')
    ax1.get_figure().savefig(path_name + 'Workout Rating Objective Accuracy Chart', bbox_inches='tight')    
    ax2.get_figure().savefig(path_name + 'Strength or Time Improvement Objective Accuracy Chart', bbox_inches='tight')    
    print('____________________________________________________')
    print('RESULTS FOR THE WORKOUT RATING OBJECTIVE')
    print(Y1_acc)
    print('____________________________________________________')
    print('RESULTS FOR THE STRENGTH/TIME IMPROVEMENT OBJECTIVE')
    print(Y2_acc)
###################################################################################################################################################################
'''
This class contains all the classifiers used for this project, as well as
custom methods to train, predict, and validate each classifier and its
respective accuracy. 
@ classifiers:
    Baseline : Majority Class Predictor
    RFC: Random Forest Classifier
    MLP: Multi-layer Perceptron
    SVM: Support Vector Machine
    QDA: Quadratic Discriminant Analysis
    SGLOG: Stacked Generalization with Logistic Regression as final classifier
'''
class multi_class_model:
    '''
    This method initializes the variables for the class.
    @ parameters: 
        X: Training data-set
        Y1: Training data-set labels for workout rating objective
        Y2: Training data-set labels for strength/time improvement objective
        X_val: Test data-set
        Y1_val: Test data-set labels for workout rating objective
        Y2_val: Test data-set labels for strength/time improvement objective
    @ returns:
        self.X: A stored copy of the training data-set
        self.Y1: A stored copy of the training data-set labels for 'workout rating' objective
        self.Y2: A stored copy of the training data-set labels for 'strength/time improvement' objective
        self.X_val: A stored copy of the test data-set
        self.Y1_val: A stored copy of the test data-set labels for 'workout rating' objective
        self.Y2_val: A stored copy of the test data-set labels for 'strength/time improvement' objective
        self.models_Y1: A stored list used to hold all classifiers used to predict 'workout rating' objective
        self.models_Y2: A stored list used to hold all classifiers used to predict 'strength/time improvement' objective
        self.results_Y1: A stored list used to hold all results from all classifiers for 'workout rating' objective predictions
        self.results_Y2: A stored list used to hold all results from all classifiers for 'strength/time improvement' objective predictions
        self.accuracy_Y1: A stored list used to hold all percentage accuracy values for all classifiers for 'workout rating' objective predictions
        self.accuracy_Y2: A stored list used to hold all percentage accuracy values for all classifiers for 'strength/time improvement' objective predictions
    '''    
    def __init__(self, X, Y1, Y2, X_val, Y1_val, Y2_val):
        self.X = X.copy()
        self.Y1 = Y1.copy()
        self.Y2 = Y2.copy()
        self.X_val = X_val.copy()
        self.Y1_val = Y1_val.copy()
        self.Y2_val = Y2_val.copy() 
        self.models_Y1 = []
        self.models_Y2 = []
        self.results_Y1 = []
        self.results_Y2 = []
        self.accuracy_Y1 = []
        self.accuracy_Y2 = []
    ########################################################################    
    '''
    This method initializes all the classifiers which will be used to predict
    the "workout rating" objective labels in the test data-set and stores the 
    classifiers in a list. The parameters passed inside of the classifier initilization 
    methods were found via 5-fold cross-validation. The cross-valdiation is omitted 
    from this code to save computation time. 
    @ returns:
        self.models_Y1: A list containing all of the classifiers initialized using parameters
        optimized for predicting the 'workout rating' objective.
    '''
    def set_models_Y1(self):    
        self.models_Y1.append(DummyClassifier(strategy = 'most_frequent'))
        self.models_Y1.append(RandomForestClassifier(n_estimators = 100, random_state = 3))
        self.models_Y1.append(MLPClassifier(solver = 'lbfgs', random_state = 2, hidden_layer_sizes = (70,), max_iter = 500))
        self.models_Y1.append(svm.SVC(kernel = 'rbf', gamma = 'auto', C = 2.8))
        self.models_Y1.append(QuadraticDiscriminantAnalysis(reg_param = 0.5))
        estimators = [('mlp', self.models_Y1[2]), ('svm', self.models_Y1[3]), ('qda', self.models_Y1[4])]
        self.models_Y1.append(StackingClassifier(estimators = estimators, final_estimator = LogisticRegression(C = 0.1), passthrough = False))
        self.num_models = len(self.models_Y1)
    ########################################################################
    '''
    This method initializes all the classifiers which will be used to predict
    the "strength/time improvement" objective labels in the test data-set and stores the 
    classifiers in a list. The parameters passed inside of the classifier initilization 
    methods were found via 5-fold cross-validation. The cross-valdiation is omitted 
    from this code to save computation time. 
    @ returns:
        self.models_Y2: A list containing all the classifiers initialized using parameters
        optimized for predicting the 'strength/time improvement' objective.
    '''            
    def set_models_Y2(self):
        self.models_Y2.append(DummyClassifier(strategy = 'most_frequent'))
        self.models_Y2.append(RandomForestClassifier(n_estimators = 100, random_state = 1))                              
        self.models_Y2.append(MLPClassifier(solver = 'lbfgs', random_state = 1, hidden_layer_sizes = (8,)))
        self.models_Y2.append(svm.SVC(kernel = 'rbf', gamma = 'auto', C = 0.8))
        self.models_Y2.append(QuadraticDiscriminantAnalysis(reg_param = 0.7))
        estimators = [('mlp', self.models_Y1[2]), ('svm', self.models_Y1[3]), ('qda', self.models_Y1[4])]
        self.models_Y2.append(StackingClassifier(estimators = estimators, final_estimator = LogisticRegression(C = 1.3), passthrough = False))
    ########################################################################
    '''
    This method trains all the classifiers which will be used to predict
    the "workout rating" objective labels on the training data-set. 
    '''
    def train_models_Y1(self):
        for model in  self.models_Y1:
            model.fit(self.X, self.Y1)
    ########################################################################
    '''
    This method trains all the classifiers which will be used to predict
    the "strength/time improvement" objective labels on the training data-set. 
    '''
    def train_models_Y2(self):
        for model in  self.models_Y2:
            model.fit(self.X, self.Y2)
    ########################################################################
    '''
    This method iterates through all the classifiers used to predict
    the "workout rating" objective labels and calls on the classifiers' methods 
    to predict the labels in the test data-set. The predicted labels
    are stored in a list.
    @ returns:
        self.results_Y1: A list containing all of the predictions from each classifier
        for the 'workout rating' objective
    '''
    def predict_Y1(self):
        for model in  self.models_Y1:
            self.results_Y1.append(model.predict(self.X_val))
    ########################################################################
    '''
    This method iterates through all the classifiers used to predict
    the "strength/time improvement" objective labels and calls on the classifiers' 
    methods to predict the labels in the test data-set. The predicted labels
    are stored in a list.
    @ returns:
        self.results_Y2: A list containing all of the predictions from each classifier
        for the 'strength/time improvement' objective
    '''
    def predict_Y2(self):
        for model in  self.models_Y2:
            self.results_Y2.append(model.predict(self.X_val))
    ########################################################################
    '''
    This method iterates through all of the predicted results for both the
    "workout rating" objective and the "strength/time improvement" objective,
    comparing the predicted results from each classifier for each objective to 
    the actual label values, using sklearn's accuracy_score method. The scores
    are mulitplied by 100 to obtain accuracy percentage valuea, and the values
    are stored in a list.
    @ returns:
        self.accuracy_Y1: A list containing all of the percentage accuracy values for
        each classifier for the 'workout rating' objective
        self.acchracy_Y2: A list containing all of the percentage accuracy values for
        each classifier for the 'strength/time improvement' objective
    '''
    def calculate_accuracy(self):
        for i in range (self.num_models):
            self.accuracy_Y1.append(accuracy_score(self.Y1_val, self.results_Y1[i]) * 100)
            self.accuracy_Y2.append(accuracy_score(self.Y2_val, self.results_Y2[i]) * 100)
    ########################################################################
    '''
    This method executes all of the methods listed above in this class to initialize all 
    the classifiers for this project, fit the classifiers to the training data-set, invoke
    the classifiers to predict the two different objectives' label values for the test data-set,
    and finally compute a percentage accuracy value for all classifiers and both objectives by
    comparing predicted labels to actual labels.
    '''
    def run(self):
        self.set_models_Y1()
        self.set_models_Y2()
        self.train_models_Y1()
        self.train_models_Y2()
        self.predict_Y1()
        self.predict_Y2()
        self.calculate_accuracy()
###################################################################################################################################################################
master, X, Y1, Y2, X_val, Y1_val, Y2_val = format_data(path_name, 'Project Data.csv')
model = multi_class_model(X, Y1, Y2, X_val, Y1_val, Y2_val)
model.run()
display_results(model.accuracy_Y1, model.accuracy_Y2)


labels = ['Volume Delta', 'Avg Night Heart Rate', 'Avg Day Heart Rate', 'Prev Max Day Heart Rate', 'Ln Avg HRV', 'Ln Avg HRV Delta', 'Lowest Night Heart Rate']
rfc1 = model.models_Y1[1].feature_importances_
rfc2 = model.models_Y2[1].feature_importances_
rfc1 = pd.DataFrame(rfc1)
rfc2 = pd.DataFrame(rfc2)
rfc1 = rfc1.apply(lambda x: x/x.max(), axis=0)
rfc2 = rfc2.apply(lambda x: x/x.max(), axis=0)
rfc1['Feature'] = labels
rfc2['Feature'] = labels
rfc1 = rfc1.sort_values(by = [0])
rfc2 = rfc2.sort_values(by = [0])
