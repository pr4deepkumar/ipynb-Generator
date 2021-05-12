import nbformat as nbf

nbname = input("Enter the name of the notebook: ")
dataset = input("Enter dataset Name/Path: ")

nb = nbf.v4.new_notebook()

code = """\
# importing necessary packages

# for numerical/scientific calculations
import numpy as np

# for data manipulation
import pandas as pd

# to ignore/not display warnings
import warnings
warnings.filterwarnings(action='ignore')
"""

lst_choice = ['data visualization','encoding','scaling/normalization','recursive feature elimination','feature selection','MLSetup','regression algorithms','classification algorithms','clustering algorithms','cross-validation','ensemble methods']
lst_code = lst_code = {'data visualization':'''\n# for data visualization
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
%matplotlib inline
''',

'encoding':'''\n# for encoding
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
''',

'scaling/normalization':'''\n# for scaling/normalization
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures # for nonlinear regression
''',

'recursive feature elimination':'''\n# for recursive feature elimination
from sklearn.feature_selection import RFE
''',

'feature selection':'''\n# for feature selection
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
''',

'MLSetup':'''\n# for MLSetup
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix, classification_report, accuracy_score
''',

'regression algorithms':'''\n# for regression algorithms
from sklearn.linear_model import LinearRegression
''',

'classification algorithms':'''\n# for classification algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
''',

'clustering algorithms':'''\n# for clustering algorithms
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
''',

'cross-validation':'''\n# for cross-validation
from sklearn.model_selection import KFold
''',

'ensemble methods':'''\n# for ensemble methods
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
'''}

req=input('Do you want packages other than numpy and pandas?: (y/n)')
if req == 'y':
    print('Choose the packages you want to import: (y/n)')
    for choice in lst_choice:
        ans=input('Would you like to use '+choice+' in this notebook?')
        if ans == 'y':
            code += lst_code[choice]
        else:
            continue

    nb['cells'] = [nbf.v4.new_markdown_cell('### '+'Importing the Necessary Packages'),
                    nbf.v4.new_code_cell(code),
                    nbf.v4.new_markdown_cell('### '+'Importing the Dataset'),
                    nbf.v4.new_code_cell('df = pd.read_csv("' +dataset+ '")'),
                    nbf.v4.new_code_cell('df.head(5)'),
                    nbf.v4.new_code_cell()
                    ]

    nbf.write(nb, 'MyNotebook.ipynb')
else:
    nb['cells'] = [nbf.v4.new_markdown_cell('### '+'Importing the Necessary Packages'),
                    nbf.v4.new_code_cell(code),
                    nbf.v4.new_markdown_cell('### '+'Importing the Dataset'),
                    nbf.v4.new_code_cell('df = pd.read_csv("' +dataset+ '")'),
                    nbf.v4.new_code_cell('df.head(5)'),
                    nbf.v4.new_code_cell()
                    ]

    nbf.write(nb, nbname+'.ipynb')