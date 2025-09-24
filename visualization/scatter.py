import ssl
import urllib.request
import seaborn as sns
import  matplotlib.pyplot as plt
import  plotly.express as px


# Create a global unverified SSL context
ssl._create_default_https_context = ssl._create_unverified_context

df = sns.load_dataset('titanic')
#sns.countplot(x='survived', data=df)

fig = px.scatter(df,x='age',y='fare',color='survived', size='pclass', hover_data=['sex','embarked'])

fig.show()





