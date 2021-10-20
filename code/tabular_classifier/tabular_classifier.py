import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np


# Read Data and Split into Training and Test Data
df = pd.read_csv('train/contact_sum.csv')
df.head()
X = df.iloc[:, 0:-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print("data_read")
data = 'mz_dmz.csv'

# # Read Training Data
# df_train = pd.read_csv('train/'+str(data))
# df_train.head()
# X_train = df_train.iloc[:, 0:-1]
# y_train = df_train.iloc[:, -1]

# # Read Testing Data
# df_test = pd.read_csv('test/'+str(data))
# df_test.head()
# X_test = df_train.iloc[:, 0:-1]
# y_test = df_train.iloc[:, -1]

ACCURACY_THRESHOLD = 0.95
# print(X)
# print(y)




class myCallback(tf.keras.callbacks.Callback):
 def on_epoch_end(self, epoch, logs={}):
		if(logs.get('accuracy') > ACCURACY_THRESHOLD):
			print("\nReached %2.2f%% accuracy, so stopping training!!" %(logs.get('accuracy')*100))
			self.model.stop_training = True

def train(X_train,y_train,X_test, y_test):

    input_size = X_train.shape[1]
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(input_size,)),
        keras.layers.Dense(int(2/3*input_size), activation=tf.nn.relu),
        # keras.layers.Dense(16, activation=tf.nn.relu),
        keras.layers.Dense(1, activation=tf.nn.sigmoid),
    ])

    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

    print(model.summary())

    model.fit(X_train, y_train, epochs=100, batch_size=27,callbacks=[myCallback()])

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print('Test accuracy:', test_acc)


callbacks = myCallback()

if __name__ == '__main__':
    train(X_train,y_train,X_test,y_test)
