This python file reads in a data-base of cardiovascular and physical training metrics
specific to this project from a CSV file and applies various learning techniques to 
the data in order to predict two separate objectives related to the cardiovascular and training metrics,
the "workout rating" objective and the "strength/time improvement" objective. After splitting the data-set
read in into separate training and test data-sets, implementing the learning techniques on the training
data-set, and making predictions on the test data-set, results of the percentage accuracy of all
classifiers in predicting the two objectives for the test data-set are displayed both graphically and numerically. 

In order to run this code, please follow the instructions below

1) Please ensure that you have python / a python interpreter downloaded on your computer. This is a python script
file and as such can only be executed if python is installed on your computer.

2) Please ensure that you have the following packages downloaded on your computer. These packages are used in 
the script and if they are not downloaded on your computer, you will get a "module not found" error.
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


3) Please ensure that you have the data-base file, "Project Data.csv" located somewhere on your computer. This
file has been provided in this zip file. It would be preferable to keep all files in the zip-file provided.

4) Please open the python code file either as a text file or with an IDE and change the "path_name" variable to match 
the path name of the location where the data-base file is located on your computer. The path_name variable is prefaced
with a comment in the script indicating to change its string value. Simply changing "path_name" to the address of this
zip-file on your computer would work.

5) If you want to run this script from the terminal, please open your command terminal, cd into the directory where this
python script is located via the cd command as such "cd {PATHNAME}" without the "{}" characters, and type 
"python TAIMOOR_QAMAR_PROJECT_CODE.py". The script should execute in the terminal, printing a table of classifier accuracy 
values, and two new png images should appear in the directory you defined in the "path_name" variable in the script.

6) If you want to run this script in an IDE, simply open the py file "TAIMOOR_QAMAR_PROJECT_CODE.py" in your IDE and run
the script file. Again, please ensure you have all the packages listed above downloaded on your computer. 

