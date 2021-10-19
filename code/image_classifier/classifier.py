import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential

train_datagenerator = ImageDataGenerator(rescale=1./255)
test_datagenerator = ImageDataGenerator(rescale=1./255)

train_datagenerator = train_datagenerator.flow_from_directory(
    'train/s',
    target_size=(128,128),
    batch_size=40,
    class_mode='binary')

test_datagenerator = test_datagenerator.flow_from_directory(
    'test/s',
    target_size=(128,128),
    batch_size=10,
    class_mode='binary')


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3),padding='same', activation='relu', input_shape=(128,128,3)),
    tf.keras.layers.MaxPooling2D((2,2),2),
    
    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2),2),     
     
    tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2),2),   
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    
    tf.keras.layers.Dense(1, activation='sigmoid')
])
print(model.summary())

model.compile(loss='binary_crossentropy',
             optimizer=tf.keras.optimizers.Adam(0.001),
             metrics=['accuracy'])

ACCURACY_THRESHOLD = 0.95

class myCallback(tf.keras.callbacks.Callback):
 def on_epoch_end(self, epoch, logs={}):
		if(logs.get('accuracy') > ACCURACY_THRESHOLD):
			print("\nReached %2.2f%% accuracy, so stopping training!!" %(ACCURACY_THRESHOLD*100))
			self.model.stop_training = True


callbacks = myCallback()

# model.fit_generator(
#     train_datagenerator,
#     epochs=10,
#     validation_data = test_datagenerator,
#     callbacks = [callbacks]
#     )
model.fit(
        train_datagenerator,
        epochs=5,
        validation_data=test_datagenerator,
        callbacks=[myCallback()]
        )


model.save('mymodel.h5')