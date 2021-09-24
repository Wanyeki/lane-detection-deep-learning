
import tensorflow as tf
from unet2 import unet
import os
from utils import read_images,display
import numpy as np



# data_dir=os.path.join(os.getcwd(),'dataset')
# test_data_dir=os.path.join(os.getcwd(),'test-dataset')

# training_image_paths=os.listdir(data_dir)
# test_image_paths=os.listdir(test_data_dir)



# print("====================================================")
# print("loading images......\n")

# training_images= read_images(training_image_paths,data_dir)
# test_images=read_images(test_image_paths,test_data_dir)

# print("====================================================")
# print('loading finished.....')

# model=unet(input_size=(720, 960, 3))
# tf.keras.utils.plot_model(model, show_shapes=True)

# print("====================================================")
# print('training .....')
# # print(training_images[9])
# print(np.shape(training_images))
# training_dataset=tf.data.Dataset.from_tensor_slices(training_images)
# validation_data=tf.data.Dataset.from_tensor_slices(test_images)

# print(training_dataset)


# model_history=model.fit(
#         training_dataset,
#         shuffle=True,  
#         epochs=5,
#         batch_size=81,
#         )

# display(training_dataset[0])




# train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#   data_dir,
#   validation_split=0.2,
#   subset="training",
#   seed=123,
#   image_size=(720, 960),
#   batch_size=81)
# print(list(train_ds)[0][1])

# all_image_paths= os.listdir(data_dir)
# actual_images_paths=filter(lambda i : "_L" not in i,all_images)
# actual_images_paths=list(actual_images)

images_dataset=tf.data.Dataset.list_files('dataset/*.png')
print (images_dataset)

# actual_image_dataset=list(filter(lambda  e: "_L" not in str(e),images_dataset))
# print(actual_image_dataset)
# mask_image_dataset=list(filter(lambda  e: "_L" in str(e),images_dataset))

# images_test_dataset=tf.data.Dataset.list_files('test-dataset/*.png')

# actual_image_test_dataset=list(filter(lambda  e: "_L" not in str(e),images_test_dataset))
# mask_image_test_dataset=list(filter(lambda  e: "_L" in str(e),images_test_dataset))

# print('..............................')
# print(actual_image_dataset[0])

# img_data=list(map(lambda x: tf.cast(x,tf.float32),actual_image_dataset))
# mask_data=list(map(lambda x: tf.cast(x,tf.float32),mask_image_dataset))


# print('...................................')
# print(len(mask_image_dataset))
# print(len(actual_image_dataset))

# 


