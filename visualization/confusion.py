import ssl
import urllib.request
import seaborn as sns
import  matplotlib.pyplot as plt


# Create a global unverified SSL context
ssl._create_default_https_context = ssl._create_unverified_context

df = sns.load_dataset('titanic')
#sns.countplot(x='survived', data=df)

corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True,cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()







