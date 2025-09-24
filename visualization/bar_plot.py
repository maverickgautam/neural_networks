import  matplotlib.pyplot as plt

classes      = ['Red', 'Green', 'Blue']
bar_value    = [20,30,10]


plt.bar(classes, bar_value)
plt.xlabel('class')
plt.ylabel('Loss')
plt.xticks(classes, ['Red', 'Green', 'Blue'])


plt.title('bar plot of loss')
plt.show()


