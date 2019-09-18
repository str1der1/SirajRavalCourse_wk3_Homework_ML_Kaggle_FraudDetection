# Gather and First Glance at the Data

# import pandas as pd

# # Read the CSV file
# df = pd.read_csv('creditcard.csv')

# # Show the contents
# print(df)


# Splitting the data

# It would be unfair to train a Machine Learning algorithm on the data and then
# test the approach on the same data. If you do that, then memorization could be
# used to achieve optimal performance: just remember all the data that is seen
# and by that you can perfectly know what to answer on the same dataset. A
# problem arises when data is unknown. The memorization technique can by no
# means predict a label for unseen data. We will split the data in a train set
# and a test set. The Machine Learning algorithm is then trained on the train
# set and its performance is computed by letting it predict labels on the test
# set. The test set is unseen data for the algorithm: it was not shown to the
# algorithm before.

# This is not the only problem. Another problem is that we are dealing with
# unbalanced data. Luckily, the scikit-learn library provides us some tools for
# splitting the unbalanced data fairly.


# Model

# As a model, I will use Logistic Regression. This model is often used in
# problems with binary target variables. Our Class variable is indeed a binary
# variable. It is not the best approach, but at least it offers some insights in
# the data.

# Implementation

# The very first step is to include all the dependencies.

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

import pandas as pd
import sys
import datetime
import seaborn as sns
from matplotlib import pyplot as plt 


# AA1 Read the data 
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ":  Reading in files")
train_id = pd.read_csv('train_identity.csv')
train_tr = pd.read_csv('train_transaction.csv')
test_id = pd.read_csv('test_identity.csv')
test_tr = pd.read_csv('test_transaction.csv')


# AA1:   MErge the data 
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ":  Merge Data")
train = pd.merge(train_tr, train_id, on='TransactionID', how='left')
test = pd.merge(test_tr, test_id, on='TransactionID', how='left')


# Now we have the data in place, we can select the features we would like to use during the training:

# Set what columns / Features to use 'Amount' and 'V1', ..., 'V28' features
# features = ['Amount'] + ['V%d' % number for number in range(1, 29)]
# From identity
features = ['id_01'] + ['id_02'] + ['id_03'] + ['id_04'] + ['id_05'] + ['id_06'] + ['id_07'] + ['id_08'] + ['id_09'] + ['id_10'] + ['id_11'] + ['id_12'] + ['id_13'] + ['id_14'] + ['id_15'] + ['id_16'] + ['id_17'] + ['id_18'] + ['id_19'] + ['id_20'] + ['id_21'] + ['id_22'] + ['id_23'] + ['id_24'] + ['id_25'] + ['id_26'] + ['id_27'] + ['id_28'] + ['id_29'] + ['id_30'] + ['id_31'] + ['id_32'] + ['id_33'] + ['id_34'] + ['id_35'] + ['id_36'] + ['id_37'] + ['id_38'] + ['DeviceType'] + ['DeviceInfo'] 
#Fron transactions 
features += ['TransactionAmt'] + ['ProductCD'] + ['card1'] + ['card2'] + ['card3'] + ['card4'] + ['card5'] + ['card6'] + ['addr1'] + ['addr2'] + ['dist1'] + ['dist2'] + ['P_emaildomain'] + ['R_emaildomain'] + ['C1'] + ['C2'] + ['C3'] + ['C4'] + ['C5'] + ['C6'] + ['C7'] + ['C8'] + ['C9'] + ['C10'] + ['C11'] + ['C12'] + ['C13'] + ['C14'] + ['D1'] + ['D2'] + ['D3'] + ['D4'] + ['D5'] + ['D6'] + ['D7'] + ['D8'] + ['D9'] + ['D10'] + ['D11'] + ['D12'] + ['D13'] + ['D14'] + ['D15'] + ['M1'] + ['M2'] + ['M3'] + ['M4'] + ['M5'] + ['M6'] + ['M7'] + ['M8'] + ['M9'] + ['V1'] + ['V2'] + ['V3'] + ['V4'] + ['V5'] + ['V6'] + ['V7'] + ['V8'] + ['V9'] + ['V10'] + ['V11'] + ['V12'] + ['V13'] + ['V14'] + ['V15'] + ['V16'] + ['V17'] + ['V18'] + ['V19'] + ['V20'] + ['V21'] + ['V22'] + ['V23'] + ['V24'] + ['V25'] + ['V26'] + ['V27'] + ['V28'] + ['V29'] + ['V30'] + ['V31'] + ['V32'] + ['V33'] + ['V34'] + ['V35'] + ['V36'] + ['V37'] + ['V38'] + ['V39'] + ['V40'] + ['V41'] + ['V42'] + ['V43'] + ['V44'] + ['V45'] + ['V46'] + ['V47'] + ['V48'] + ['V49'] + ['V50'] + ['V51'] + ['V52'] + ['V53'] + ['V54'] + ['V55'] + ['V56'] + ['V57'] + ['V58'] + ['V59'] + ['V60'] + ['V61'] + ['V62'] + ['V63'] + ['V64'] + ['V65'] + ['V66'] + ['V67'] + ['V68'] + ['V69'] + ['V70'] + ['V71'] + ['V72'] + ['V73'] + ['V74'] + ['V75'] + ['V76'] + ['V77'] + ['V78'] + ['V79'] + ['V80'] + ['V81'] + ['V82'] + ['V83'] + ['V84'] + ['V85'] + ['V86'] + ['V87'] + ['V88'] + ['V89'] + ['V90'] + ['V91'] + ['V92'] + ['V93'] + ['V94'] + ['V95'] + ['V96'] + ['V97'] + ['V98'] + ['V99'] + ['V100'] + ['V101'] + ['V102'] + ['V103'] + ['V104'] + ['V105'] + ['V106'] + ['V107'] + ['V108'] + ['V109'] + ['V110'] + ['V111'] + ['V112'] + ['V113'] + ['V114'] + ['V115'] + ['V116'] + ['V117'] + ['V118'] + ['V119'] + ['V120'] + ['V121'] + ['V122'] + ['V123'] + ['V124'] + ['V125'] + ['V126'] + ['V127'] + ['V128'] + ['V129'] + ['V130'] + ['V131'] + ['V132'] + ['V133'] + ['V134'] + ['V135'] + ['V136'] + ['V137'] + ['V138'] + ['V139'] + ['V140'] + ['V141'] + ['V142'] + ['V143'] + ['V144'] + ['V145'] + ['V146'] + ['V147'] + ['V148'] + ['V149'] + ['V150'] + ['V151'] + ['V152'] + ['V153'] + ['V154'] + ['V155'] + ['V156'] + ['V157'] + ['V158'] + ['V159'] + ['V160'] + ['V161'] + ['V162'] + ['V163'] + ['V164'] + ['V165'] + ['V166'] + ['V167'] + ['V168'] + ['V169'] + ['V170'] + ['V171'] + ['V172'] + ['V173'] + ['V174'] + ['V175'] + ['V176'] + ['V177'] + ['V178'] + ['V179'] + ['V180'] + ['V181'] + ['V182'] + ['V183'] + ['V184'] + ['V185'] + ['V186'] + ['V187'] + ['V188'] + ['V189'] + ['V190'] + ['V191'] + ['V192'] + ['V193'] + ['V194'] + ['V195'] + ['V196'] + ['V197'] + ['V198'] + ['V199'] + ['V200'] + ['V201'] + ['V202'] + ['V203'] + ['V204'] + ['V205'] + ['V206'] + ['V207'] + ['V208'] + ['V209'] + ['V210'] + ['V211'] + ['V212'] + ['V213'] + ['V214'] + ['V215'] + ['V216'] + ['V217'] + ['V218'] + ['V219'] + ['V220'] + ['V221'] + ['V222'] + ['V223'] + ['V224'] + ['V225'] + ['V226'] + ['V227'] + ['V228'] + ['V229'] + ['V230'] + ['V231'] + ['V232'] + ['V233'] + ['V234'] + ['V235'] + ['V236'] + ['V237'] + ['V238'] + ['V239'] + ['V240'] + ['V241'] + ['V242'] + ['V243'] + ['V244'] + ['V245'] + ['V246'] + ['V247'] + ['V248'] + ['V249'] + ['V250'] + ['V251'] + ['V252'] + ['V253'] + ['V254'] + ['V255'] + ['V256'] + ['V257'] + ['V258'] + ['V259'] + ['V260'] + ['V261'] + ['V262'] + ['V263'] + ['V264'] + ['V265'] + ['V266'] + ['V267'] + ['V268'] + ['V269'] + ['V270'] + ['V271'] + ['V272'] + ['V273'] + ['V274'] + ['V275'] + ['V276'] + ['V277'] + ['V278'] + ['V279'] + ['V280'] + ['V281'] + ['V282'] + ['V283'] + ['V284'] + ['V285'] + ['V286'] + ['V287'] + ['V288'] + ['V289'] + ['V290'] + ['V291'] + ['V292'] + ['V293'] + ['V294'] + ['V295'] + ['V296'] + ['V297'] + ['V298'] + ['V299'] + ['V300'] + ['V301'] + ['V302'] + ['V303'] + ['V304'] + ['V305'] + ['V306'] + ['V307'] + ['V308'] + ['V309'] + ['V310'] + ['V311'] + ['V312'] + ['V313'] + ['V314'] + ['V315'] + ['V316'] + ['V317'] + ['V318'] + ['V319'] + ['V320'] + ['V321'] + ['V322'] + ['V323'] + ['V324'] + ['V325'] + ['V326'] + ['V327'] + ['V328'] + ['V329'] + ['V330'] + ['V331'] + ['V332'] + ['V333'] + ['V334'] + ['V335'] + ['V336'] + ['V337'] + ['V338'] + ['V339']

# print(len(features))
# 431

# The target variable which we would like to predict, is the 'Class' variable
target = 'isFraud'

# Now create an X variable (containing the features) and an y variable (containing only the target variable)
X = train[features]
y = train[target]

# Now show using the Seaborn library
# Card4 = type of card e..g mastercard
# Card6 =  type of card = debit , credit
# Test
# sns.set(style="darkgrid")
# tips = sns.load_dataset("tips")
# sns.relplot(x="total_bill", y="tip", data=tips);
# plt.show()

# Show counts of creditcard vs debit card 
sns.set(style="whitegrid")
sns.countplot(x="card6", data=train);
plt.show()



# Notice that some of the variables have a wide range of values (like the Amount
# variable). In order to get all variables in an equivalent range, we subtract
# the mean and divide by the standard deviation such that the distribution of
# the values is normalized:

def normalize(X):
    """
    Make the distribution of the values of each variable similar by subtracting the mean and by dividing by the standard deviation.
    """
    for feature in X.columns:
        X[feature] -= X[feature].mean()
        X[feature] /= X[feature].std()
    return X

# Okay, now it is time for some action! We will first define the model (the
# Logistic Regression model) and then loop through a train and test set which
# have approximately the same Class distribution. First, we train the model by
# the train set and then valide the results with the test set. The
# StratisfiedShuffleSplit makes sure that the Class variable has roughly the
# same distribution in both the train set and the test set. The random state
# specification makes sure that the result is deterministic: in other words, we
# will get the same results if we would run the analysis again. The
# normalization is done for both the train set and test set. If this was done
# before the split, some information of the test set would be used in the
# normalization of the train set and this is not fair since the test set is not
# completely unseen then. The following code does the job:


## CLEAN the Data 
#  Remove the NULL / NaN items 
X2 = X.fillna(0)

# Remove the String columns before normalizing
X3 = X2.drop(X2.select_dtypes(include=['object']).columns, axis=1)

# Normalize the data 
X4 = normalize(X3)


# Define the model
model = LogisticRegression()

# Define and setup the splitter for splitting the data in a train set and a test set
# Number of splits = 1
# test_size = 50% of the data is copied into splitter variable
# Random_state = do you ranodmise the data when splitting 

print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ":  Configure Split")
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)

print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ":  Start Split")
# Loop through the splits (only one defined in n_split = 1 above)
for train_indices, test_indices in splitter.split(X4, y):
    
    # Select the train and test data
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ":  Split the train for X,y")
    X_train, y_train = X4.iloc[train_indices], y.iloc[train_indices]
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ":  Split the test for X,y")
    X_test, y_test = X4.iloc[test_indices], y.iloc[test_indices]
    
    # Normalize the data
    # X_train = normalize(X_train)
    # X_test = normalize(X_test)
    
    # Fit and predict!
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ":  Fit the Model")
    model.fit(X_train, y_train)
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ":  run the prediction")
    y_pred = model.predict(X_test)
    
    # And finally: show the results
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ":  classification_report")
    print(classification_report(y_test, y_pred))


# This results into the following:

#              precision    recall  f1-score   support

#           0       1.00      1.00      1.00    142158 
#           1       0.88      0.61      0.72       246

# avg / total       1.00      1.00      1.00    142404

# This is actually a great result! The 0 classes (transactions without fraud)
# are predicted with 100% precision and recall. It has some issues with
# detecting the 1 classes (transactions which are fraudulent). It can predict
# fraud with 88% precision. This means that 12% of the transactions which are
# fraudulent remain undetected by the system. But, 88% is still quite good!

# ################  AA1:  Quit early ###############################
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ":  Quit Point")
sys.exit()
# ################  AA1:  Quit early ###############################

