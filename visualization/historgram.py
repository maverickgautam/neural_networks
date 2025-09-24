import ssl
import urllib.request

import seaborn as sns
import  matplotlib.pyplot as plt


# Create a global unverified SSL context
ssl._create_default_https_context = ssl._create_unverified_context


df = sns.load_dataset('iris')

plt.hist(df['sepal_length'],bins=20)
plt.title('Histogram of sepal length')

plt.xlabel('Sepal length')
plt.ylabel('Frequency')


plt.show()


plt.scatter(df['sepal_length'], df['sepal_length'], c=df['species'].astype('category').cat.codes)

#plt.show()

