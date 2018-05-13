
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


############ Data Aggregation ###########

#number of transactions of specific card each creation_month
df=data_set[['card_id','creation_month']]
card_per_month = df.groupby(['card_id','creation_month'], as_index=False)
df['card_month'] = 1
a = card_per_month.sum()
data_set=pd.merge(data_set,a,how='inner')



#number of transactions for date
day = data_set[['card_id','creationdate']]
day['date']=[x.date() for x in day['creationdate']]
day['card_id_day'] = 1
card_per_day = day.groupby(['card_id','date'], as_index=False) 
dd = card_per_day.sum()
data_set=pd.merge(data_set,dd,how='inner')



#currency type over month
currency = data_set[['currencycode','creation_month']]
currency['currency_month'] = 1
cur_per_month = currency.groupby(['currencycode','creation_month'], as_index=False)
cu = cur_per_month.sum()
data_set=pd.merge(data_set,cu,how='inner')


#Merchant type over month
Merchant = data_set[['accountcode','creation_month']]
Merchant['merchant_month'] = 1
mer_per_month = Merchant.groupby(['accountcode','creation_month'], as_index=False)
me = mer_per_month.sum()
data_set=pd.merge(data_set,me,how='inner')


#We select specific features from the original dataset
selected_features = [ 'issuercountrycode', 'txvariantcode', 'EuroAmount', 'amount',
                            'currencycode', 'shoppercountrycode', 'shopperinteraction', 'cardverificationcodesupplied', 'simple_journal',
                            'cvcresponsecode', 'accountcode','creation_hour', 'creation_day',
                            'creation_month', 'creation_year', 'ip_id',
                             'mail_id', 'bin', 'card_id','card_month','card_id_day','currency_month','merchant_month']


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
comp = 100
test_feature = functions.pca(encoded_feat, enc_label, comp)
test_ff = pd.DataFrame(test_feature)

rf = RandomForestClassifier()
tp, fp, tn, fn, true_labels, probs = functions.cross_validation(rf, test_ff, enc_label, 10, 0.5)
print('Random Forest True Positives %s' % sum(tp))
print('Random Forest False Positives %s' % sum(fp))

fScore, precision, recall = functions.metrics(tp, fp, tn, fn)

print('Random Forest Fscore %s' % fScore)
print('Random Forest Precision %s' % precision)
print('Random Forest Recall %s' % recall)


#ROC Curve
False_Positive_Rate, True_Positive_Rate, threshold = roc_curve(true_labels, probs)
aucF = roc_auc_score(true_labels, probs)
plt.title('Random Forest Classifier AUC')
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(False_Positive_Rate, True_Positive_Rate, color='blue',label='AUC Smoted = %0.2f' % aucF)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.savefig('Random Forest Classifier Bonus')
plt.show()





