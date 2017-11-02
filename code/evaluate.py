
# coding: utf-8

# In[ ]:

import os
import time
import re
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import models
from os import walk
from glob import glob 
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# #### Configurables

# In[ ]:

test_dir = "bbr/test_bing"

class_mode = 'categorical'    # binary or categorical

data_x = 64
data_y = 64
color_mode = 'grayscale'   # 'grayscale' or 'rgb'

test_batch_size = 10
test_steps = 4  # Number of test instances divided by batch size


# In[ ]:

cases = [
    "train_100_same_shuffled_augmented_wider",
    "train_100_same_shuffled_augmented_deeper",
    "train_100_same_shuffled_augmented",
    "train_100_same_shuffled",
]
case_names = {
    "train_100_same_shuffled_augmented_wider" : "Data Augmentation, Wider NN ",
    "train_100_same_shuffled_augmented_deeper" : "Data Augmentation, Deeper NN",
    "train_100_same_shuffled_augmented" : "Data Augmentation",
    "train_100_same_shuffled" : "No Data Augmentation",
}


# In[ ]:

t1 = time.clock()
model_evaluation = {}
for case in cases:
    t_c_1 = time.clock()
    model_dir = "output/" + case
    
    paths = glob(model_dir + "/*.h5")
    model_list = [p.replace("\\","/") for p in paths]

    test_datagen = ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(data_x, data_y),
        batch_size=test_batch_size,
        class_mode=class_mode,
        color_mode=color_mode,
        seed=55                        # Seed
    )
    acc_list = []
    for m in model_list:
        model = models.load_model(m)
        print(m)
        time_start = time.clock()
        test_loss, test_acc = model.evaluate_generator(test_generator, steps=test_steps)
        time_end = time.clock()
        acc_list.append(test_acc)
        print('test acc:', test_acc)
        duration = time_end - time_start
        #print("Time to evaluate")
        print("Time elapsed in seconds : %.0f" % duration)
        #print("Time elapsed in minutes : %.1f" % (duration/60))
        print("")
    model_evaluation[case] = acc_list
    t_c_2 = time.clock()
    print(case)
    print("This case took %.1f min" % ((t_c_2 - t_c_1)/60))
    print("==============================================")
    print("")
t2 = time.clock()
print("Total time is %.1f min" % ((t2 - t1)/60))


# In[ ]:

for case in model_evaluation:
    print(np.mean(model_evaluation[case]))


# In[ ]:

colors = [
    'b',  # blue
    'g',  # green
    'r',  # red
    'c',  # cyan
    'm',  # magenta
    'y',  # yellow
    'k',  # black
    'w',  # white
]
shapes = [
    'x',
    'o',
    '.',
    '^',
    'D',
    '|',
    '_',
    'X',
]


# In[ ]:

plt.figure(figsize=(8,7))
i = 0
for case in model_evaluation:
    plt.plot(model_evaluation[case], colors[i] + shapes[i], alpha=0.7, label=case_names[case])
    i += 1
plt.title('Test Accuracy')
plt.ylim((.2, 1.1))
plt.legend(loc='best')
plt.grid()
plt.savefig("test_accuracy_01.jpg")
plt.show()

