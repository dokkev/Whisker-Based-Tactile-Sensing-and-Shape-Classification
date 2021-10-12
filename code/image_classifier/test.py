import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image


model = tf.keras.models.load_model('weight/mymodel.h5')

path = 'test/concave/concave20.obj_T038_N00.jpg'
img = image.load_img(path, target_size=(128, 128))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict(images)
print(classes[0])
if classes[0]<0.5:
    print("Given image contains Concave")
else:
    print("Given image contains Convex")