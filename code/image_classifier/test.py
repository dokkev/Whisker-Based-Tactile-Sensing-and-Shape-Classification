import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from re import search


objects = [ 'concave20.obj','concave22.obj','concave24.obj','concave26.obj','concave28.obj',
            'concave30.obj','concave32.obj','concave34.obj','concave36.obj','concave38.obj',
            'concave40.obj',
            'convex20.obj','convex22.obj','convex24.obj','convex26.obj','convex28.obj',
            'convex30.obj','convex32.obj','convex34.obj','convex36.obj','convex38.obj',
            'convex40.obj']



# name of the dir in output
simID = 0
objID = 0
objects_max = 22


if __name__ == '__main__':
     for objID in range(objects_max):
        objFile = objects[objID]

        trialID = 0
        trials_max = 5
        
        
        while trialID < trials_max:

            model = tf.keras.models.load_model('weight/concave_mxyz.h5')

            input_type = 'mxyz/'
            dirname = str(objFile) + '_T' + format(trialID, '03d') + '_N' + format(simID, '02d')
            # print(dirname)


            if search('concave',dirname):
                obj_type = 'concave'
            else :
                obj_type = 'convex'




            path = 'test/'+input_type + obj_type + '/'+ dirname + '.jpg'
            img = image.load_img(path, target_size=(128, 128))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)

            images = np.vstack([x])
            classes = model.predict(images)
            # prediction = model.predict_proba(images)

            print(classes[0])


            if classes[0]<0.5:
                print(dirname,"<< This is Concave")
            else:
                print(dirname," << This is Convex")

            trialID += 1
        
        simID += 1
        objID += 1
