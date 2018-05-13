import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import pandas as pd

import functions 

#Load dataset
data_set=functions.loadData()
print('Dataset is loaded')

dataset =data_set[['EuroAmount','accountcode','simple_journal']]
chargeback=dataset.loc[dataset['simple_journal']=='Chargeback']
settled=dataset.loc[dataset['simple_journal']!='Chargeback']

#Visualizations
plt.title('Chargeback transactions')
sns.boxplot(x="accountcode", y="EuroAmount", data=chargeback)  # for pretty visualization
plt.xlabel('Merchant webshop')
plt.ylabel('Amount in Euros')
tick_value = [1000, 10000, 20000, 30000, 40000, 50000,60000,70000,80000]
tick_label = ['1k', '10k', '20k', '30k', '40k', '50k','60k','70k','80k']
plt.yticks(tick_value, tick_label)
plt.savefig('Chargeback transactions')
plt.show()


plt.title('Settled transactions')
sns.boxplot(x="accountcode", y="EuroAmount", data=settled[settled['EuroAmount']<= 50000])  # for pretty visualization
plt.xlabel('Merchant webshop')
plt.ylabel('Amount in Euros')
tick_value = [1000, 5000, 10000, 15000, 20000,25000,30000,35000,40000,50000]
tick_label = ['1k', '5k', '10k', '15k', '20k', '25k','30k', '35k', '40k', '50k']
plt.yticks(tick_value, tick_label)
plt.savefig('Settled transactions')
plt.show()


plt.title('Chargeback vs Settled transactions')
sns.boxplot(x="simple_journal", y="EuroAmount", data=dataset[dataset['EuroAmount']<= 100000])  # for pretty visualization
plt.xlabel('Label')
plt.ylabel('Amount in Euros')
tick_value = [1000,5000,10000, 20000, 30000, 40000, 50000, 60000,70000, 80000, 90000, 100000]
tick_label = ['1k','5k','10k', '20k', '30k', '40k', '50k', '60k','70k', '80k', '90k', '100k']
plt.yticks(tick_value, tick_label)
plt.savefig('Chargeback transactions Vs Settled transactions')
plt.show()

#Heatmaps
functions.heatmap(data_set,'Chargeback','shoppercountrycode','accountcode')
functions.heatmap(data_set,'Chargeback','issuercountrycode', 'txvariantcode')