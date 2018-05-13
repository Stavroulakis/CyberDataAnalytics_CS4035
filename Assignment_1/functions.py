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


def cross_validation(classifier, data, labels, n_of_fold, threshold=0.5,  method="SMOTE", ratio=0.5):
    true_pos = []
    false_pos = []
    true_neg = []
    false_neg = []
    auc = []
    real_labels = []
    probability_labels = []
    #KFold creates the fold with respect to the number given as an input
    kfold = KFold(n_splits=n_of_fold, shuffle=True, random_state=42)
    for train_ind, test_ind in kfold.split(data):
        #split the dataset into train and test as the kfold indicates
        data_train, data_test = data.iloc[train_ind], data.iloc[test_ind]
        labels_train, labels_test = labels.iloc[train_ind], labels.iloc[test_ind]
        #apply resampling method
        if method == 'under':
            und=RandomUnderSampler(ratio=float(ratio))
            data_resampling, labels_resampling=und.fit_sample(data_train, labels_train)
        elif method == 'smotetomek':
            resampling = SMOTETomek(ratio=float(ratio))
            data_resampling, labels_resampling = resampling.fit_sample(data_train, labels_train)
        else:
            resampling = SMOTE(ratio=float(ratio))
            data_resampling, labels_resampling = resampling.fit_sample(data_train, labels_train)
        #train the classifier with the train data after the resampling
        classifier.fit(data_resampling, labels_resampling)
        #evaluate the model for the output predictions using confusion matrix
        labels_prediction = classifier.predict(data_test)
        tn, fp, fn, tp = confusion_matrix(labels_test, labels_prediction).ravel()
        #computhe the probabilities and use them to construct the roc curve
        labels_prediction_probability = classifier.predict_proba(data_test)[:, 1]
        #store the data in a array
        true_pos.append(tp)
        false_pos.append(fp)
        true_neg.append(tn)
        false_neg.append(fn)
        #keep the labels and the probabilities
        real_labels.extend(labels_test)
        probability_labels.extend(labels_prediction_probability)
    #transform the list to nparray
    true_pos = np.array(true_pos)
    false_pos = np.array(false_pos)
    true_neg = np.array(true_neg)
    false_neg = np.array(false_neg)
    return true_pos, false_pos, true_neg, false_neg, real_labels, probability_labels



def metrics(tp, fp, tn, fn):
    precision = 1.0 * tp / (tp + fp)
    recall = 1.0 * tp /(tp + fn)
    f = (1 + 0.25)*(precision * recall)/(0.25 *precision + recall)
    return np.mean(f), np.mean(precision), np.mean(recall)


def pca(feature_vector, labels, comp=200):
    print("Starting PCA")
    pca = decomposition.PCA(n_components=comp)
    coeff = pca.fit_transform(feature_vector)
    return coeff




#function to convert currencies into Euro
def Euro_converter(currency,amount):
    coversion_dict = {'SEK':0.09703,'MXN':0.04358,'AUD':0.63161,'NZD':0.58377,'GBP':1.13355}
    return round(amount * coversion_dict[currency])



def findOptimalThreshold(data, label, classifier):
	dataset=data.values
	labels=label.values
	X_train, X_test, y_train, y_test = train_test_split(dataset, labels,test_size=0.2,random_state=42, stratify=labels)
	ratio = [0.2, 0.4, 0.6]
	prob_threshold = [0.5]
	for x in ratio:
		smt=SMOTE(random_state=42, ratio=float(x))
		new_X_train, new_y_train=smt.fit_sample(X_train,y_train)
		classifier.fit(new_X_train, new_y_train)
		smoted_probs = classifier.predict_proba(X_test)[:, 1]
		f_score=0
		for j in prob_threshold:
			temp=smoted_probs>j
			tn, fp, fn, tp = confusion_matrix(y_test,temp).ravel()
			precision = 1.0 * tp / (tp + fp)
			recall = 1.0 * tp /(tp + fn)
			f = (1 + 0.25)*(precision * recall)/(0.25 *precision + recall)
			if f > f_score:
				f_score=f
				optimal_threshold=[j,x]
			print('score',f,'precision',precision,'Recall',recall,'prob',j)
	return optimal_threshold



def loadData():
    data_set = pd.read_csv('data_for_student_case.csv')

	#Construct the dataframe format
    #Delete entries that have been recognised as Refuced transactions
    data_set=data_set.loc[data_set['simple_journal']!='Refused']
    #Convert amount into Euros
    data_set['EuroAmount']=data_set.apply(lambda row: Euro_converter(row['currencycode'],row['amount']),axis=1)
    #Convert creatiomn date into datetime format and extract each month, day, year,hour
    data_set['creationdate'] = pd.to_datetime(data_set['creationdate'],format='%Y-%m-%d %H:%M:%S')
    data_set['bookingdate'] = pd.to_datetime(data_set['bookingdate'],format='%Y-%m-%d %H:%M:%S')
    data_set['creation_month'] = data_set.creationdate.dt.month
    data_set['creation_year'] = data_set.creationdate.dt.year
    data_set['creation_day'] = data_set.creationdate.dt.day
    data_set['creation_hour'] = data_set.creationdate.dt.hour
    #label column to have simple_journal in binary format
    data_set['label'] = np.where(data_set['simple_journal']=='Chargeback', 1, 0)
    return data_set


def heatmap(dataset, label,col1,col2):
# filter and aggregate data
    filtered_dataset = dataset[dataset['simple_journal'] == label]
    aggregation_data = filtered_dataset.groupby([col1, col2]).size()\
        .reset_index(name='count')
    # filter country code for pretty visualization
    account_code = list(dataset[col2][dataset.simple_journal == 'Chargeback'].unique())
    shopper_code = list(dataset[col1][dataset.simple_journal == 'Chargeback'].unique())
    aggregation_data = aggregation_data[aggregation_data[col2].isin(account_code)]
    aggregation_data = aggregation_data[aggregation_data[col1].isin(shopper_code)]
    final_data = aggregation_data.pivot(index=col2, columns=col1, values='count')
    sns.set()
    cmap = sns.cm.rocket_r
    sns.heatmap(final_data, linewidth=0.5,cmap = cmap)
    plt.xlabel(col2)
    plt.ylabel(col1)
    plt.savefig('Heatmap')
    plt.show()