import  matplotlib.pyplot as plt

epoch      = [1,2,3,4,5]
train_loss = [0.8,0.9,0.6,0.4,0.2]
val_loss   = [0.9,0.7,0.6,0.3,0.2]

# Plot X, Y
plt.plot(epoch,train_loss,label='train')
plt.plot(epoch,val_loss,label='val')

# Label X axis and Y axis
plt.xlabel('epoch')
plt.ylabel('loss')

# Provide Title of the Plot at the Center top
plt.title('Training vs validation loss')

#Place Grid Inside the plot
plt.grid(True)

# Show legends [Box at top right hand side ]
plt.legend()

plt.show()