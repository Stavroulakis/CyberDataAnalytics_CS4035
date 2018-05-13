import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import pandas as pd

from imblearn.over_sampling import SMOTE,RandomOverSampler
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_auc_score, roc_curve, recall_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_curve
from sklearn import decomposition, tree

import functions 

#Load dataset
data_set=functions.loadData()
print('Dataset is loaded')

#Select specific features from the entire dataset
selected_features = [ "issuercountrycode", "txvariantcode","EuroAmount", "currencycode", "shoppercountrycode", "shopperinteraction", "cardverificationcodesupplied", "cvcresponsecode", "accountcode"]
new_data=data_set[selected_features]

#Create dummies dataset
new_data=pd.get_dummies(new_data)
#Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(new_data, data_set['label'],test_size=0.2,random_state=42, stratify=data_set['label'])

# Replace all nan values with 0 
X_train=X_train.fillna(0)
X_test=X_test.fillna(0)

#List of all classifiers
classifiers = {}
classifiers['Logistic Classifier']=LogisticRegression()
classifiers['Decision Tree Classifier'] = tree.DecisionTreeClassifier()
classifiers['Random Forest Classifier'] = RandomForestClassifier(n_estimators=50, criterion='gini')
classifiers[ 'Knn Classifier'] = KNeighborsClassifier(n_neighbors=5)

#Loop for each classifier
for classifier in classifiers:
	
	#Fit Unsmoted data to classifier
	classifiers[classifier].fit(X_train, y_train)
	unsmoted_probs = classifiers[classifier].predict_proba(X_test)[:, 1]
	unsmoted_predicts= classifiers[classifier].predict(X_test)
	False_Positive_Rate_unsmoted, True_Positive_Rate_unsmoted, threshold_unsmoted = roc_curve(y_test, unsmoted_probs)
	auc_unsmoted = roc_auc_score(y_test, unsmoted_probs)
	recall_unsmoted_score=recall_score(y_test, classifiers[classifier].predict(X_test))
	
	# Fit Smoted Data to classifier
	smt=SMOTE(random_state=42, ratio=float(0.5))
	new_X_train, new_y_train=smt.fit_sample(X_train,y_train)
	classifiers[classifier].fit(new_X_train, new_y_train)
	smoted_probs = classifiers[classifier].predict_proba(X_test)[:, 1]
	smoted_predicts=classifiers[classifier].predict(X_test)
	False_Positive_Rate_Smoted, True_Positive_Rate_Smoted, threshold_Smoted = roc_curve(y_test, smoted_probs)
	auc_Smoted = roc_auc_score(y_test, smoted_probs)
	recall_smoted_score=recall_score(y_test, classifiers[classifier].predict(X_test))

	#Fit undersampling data to classifier
	und=RandomUnderSampler(random_state=42, ratio=float(0.5))
	undersampled_X_train ,undersampled_y_train=und.fit_sample(X_train,y_train)
	classifiers[classifier].fit(undersampled_X_train, undersampled_y_train)
	undersampled_probs = classifiers[classifier].predict_proba(X_test)[:, 1]
	undersampled_predicts=classifiers[classifier].predict(X_test)
	False_Positive_Rate_undersampled, True_Positives_Rate_undersampled, thresholds_undersampled = roc_curve(y_test, undersampled_probs)
	auc_undersampled = roc_auc_score(y_test, undersampled_probs)
	recall_undersampled_score=recall_score(y_test, classifiers[classifier].predict(X_test))

	#Fit oversampling data to classifier
	over=RandomOverSampler(random_state=42, ratio=float(0.5))
	oversampled_X_train ,oversampled_y_train=over.fit_sample(X_train,y_train)
	classifiers[classifier].fit(oversampled_X_train, oversampled_y_train)
	oversampled_probs = classifiers[classifier].predict_proba(X_test)[:, 1]
	oversampled_predicts=classifiers[classifier].predict(X_test)
	False_Positive_Rate_oversampled, True_Positives_Rate_oversampled, thresholds_oversampled = roc_curve(y_test, oversampled_probs)
	auc_oversampled = roc_auc_score(y_test, oversampled_probs)
	recall_oversampled_score=recall_score(y_test, classifiers[classifier].predict(X_test))
	
	#ROC CURVE
	plt.title('%s AUC' % classifier)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.plot(False_Positive_Rate_unsmoted, True_Positive_Rate_unsmoted, color='blue',label='AUC Unsmoted = %0.2f' % auc_unsmoted)
	plt.plot(False_Positive_Rate_Smoted, True_Positive_Rate_Smoted, color='green', label='AUC Smoted = %0.2f' % auc_Smoted)
	plt.plot(False_Positive_Rate_undersampled, True_Positives_Rate_undersampled, color='red',label='AUC UnderSampled = %0.2f' % auc_undersampled)
	plt.plot(False_Positive_Rate_oversampled, True_Positives_Rate_oversampled, color='m',label='AUC OverSampled = %0.2f' % auc_oversampled)
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.legend(loc="lower right")
	plt.savefig('%s ROC' % classifier)
	plt.show()
	
	#PRECICION-RECALL Curve
	precision_unsmoted, recall_unsmoted ,_= precision_recall_curve(y_test, unsmoted_probs)
	precision_smoted, recall_smoted,_ = precision_recall_curve(y_test, smoted_probs)
	plt.plot(recall_unsmoted, precision_unsmoted,  color="blue",label='Precision-Recall Unsmoted')
	plt.plot(recall_smoted, precision_smoted,  color="green",label='Precision-Recall Smoted  ')
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.title('Precision Recall Curve -  %s'%classifier)
	plt.legend(loc="upper right")
	plt.savefig('Precision Recall Curve -  %s'%classifier)
	plt.show()