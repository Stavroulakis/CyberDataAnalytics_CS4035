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

selected_features = [ 'issuercountrycode', 'txvariantcode', 'EuroAmount', 'amount',
                            'currencycode', 'shoppercountrycode', 'shopperinteraction', 'cardverificationcodesupplied', 'simple_journal',
                            'cvcresponsecode', 'accountcode','creation_hour', 'creation_day','creation_month', 'creation_year', 'ip_id', 'mail_id', 'bin', 'card_id']


all_features = data_set[selected_features]
all_features = all_features.dropna(axis=0, how='any')

all_features['label'] = np.where(all_features['simple_journal']=='Chargeback', 1, 0)
label = all_features['label']
enc_label = all_features['label']

all_features = all_features.drop(['txvariantcode','amount','creation_hour', 'creation_month', 'creation_year','simple_journal', 'label', 'mail_id', 'ip_id'],axis=1)

#this function is created specifically for the columns mail_id, ip_id, card_id, bin
#these columns usually contain unique data since each one of these are personal
#we identified some patterns, like that the most of the bins are used many times
#also there are card_ids frequently used
#we add this information as feature to see if our results will become better
def personal_data_encoding(column, threshold):
    count = dict(all_features[column].value_counts())
    mapping = {}
    for id in count.keys():
        if count[id]>threshold:
            mapping[id] = id
        else:
        	#we don't care for the appearances of 
            mapping[id] = 'dc'
    all_features[column] = all_features[column].map(mapping)



personal_data_encoding('bin',3)
personal_data_encoding('card_id',10)

encoded_feat = pd.get_dummies(all_features)

#after the encoding too many columns are created
#we use pca to reduce the features and identify the most discriminative ones
#we have the best results for 500 components however it takes a lot of time
#thus we have 100 for the script
comp = 200
test_feature = functions.pca(encoded_feat, enc_label, comp)
test_ff = pd.DataFrame(test_feature)



#evaluate the white-box classifier
td = tree.DecisionTreeClassifier()

#find the optimal boundary for the decision tree
threshold = functions.findOptimalThreshold(test_ff, enc_label, td)

#cross validation takes a lot of time for the decision tree, sorry :(
tp, fp, tn, fn, real, prob = functions.cross_validation(td, test_ff, enc_label, 10, threshold, 'smote', 0.5)

print('Decision Tree True Positives %s' % sum(tp))
print('Decision Tree False Positives %s' % sum(fp))
#since we have true positives, negative etc we compute the metrics to evaluate the results
fScore, precision, recall = functions.metrics(tp, fp, tn, fn)

print('Decision Tree Fscore %s' % fScore)
print('Decision Tree Precision %s' % precision)
print('Decision Tree Recall %s' % recall)


#ROC Curve###
False_Positive_Rate, True_Positive_Rate, threshold = roc_curve(true_labels, probs)
aucTree = roc_auc_score(true_labels, probs)
plt.title('Decision Tree Classifier AUC')
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(False_Positive_Rate, True_Positive_Rate, color='blue',label='AUC Smoted = %0.2f' % aucTree)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.savefig('Decision Tree Classifier ')
plt.show()
#in the repost we have provided an image of the tree
#another image can be created using the following code
#we have not added in the evaluation function because it needs a lot of time to run
#and is not needed every time
#Create the image of the tree
Ttrain, Ttest, TL_train, TL_test = train_test_split(test_ff, enc_label,test_size=0.2,random_state=42, stratify=enc_label.values)
td.fit(Ttrain, TL_train)
#tree.export_graphviz(td, out_file='tree_5_500.dot', max_depth=5)
#run in command line to convert dot file to png
#dot -Tpng tree.dot -o tree.png

#evaluate black-box
rf = RandomForestClassifier()
tp, fp, tn, fn, real, prob = functions.cross_validation(rf, test_ff, enc_label, 10, 0.5)
print('Random Forest True Positives %s' % sum(tp))
print('Random Forest False Positives %s' % sum(fp))

fScore, precision, recall = functions.metrics(tp, fp, tn, fn)

print('Random Forest Fscore %s' % fScore)
print('Random Forest Precision %s' % precision)
print('Random Forest Recall %s' % recall)

#ROC Curve###
False_Positive_Rate, True_Positive_Rate, threshold = roc_curve(true_labels, probs)
aucTree = roc_auc_score(true_labels, probs)
plt.title('Random Forest Classifier AUC')
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(False_Positive_Rate, True_Positive_Rate, color='blue',label='AUC Smoted = %0.2f' % aucTree)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.savefig('Random Forest Classifier')
plt.show()






