import tensorflow as tf
import os
import matplotlib.pyplot as plt

def  read_images(img_names,dir):
    all_images=[]
    for img in img_names:
        if("_L" not in img and ".png" in img):
            path=os.path.join(dir,img)
            image=tf.io.read_file(path)
            image=tf.image.decode_png(image,channels=3)
            image=tf.image.convert_image_dtype(image,tf.uint8)
            image=tf.cast(image,tf.float32)/255

            
            mask_path=path.replace(".png","_L.png")
            mask=tf.io.read_file(mask_path)
            mask=tf.image.decode_png(mask,channels=3)
            mask=tf.image.convert_image_dtype(mask,tf.uint8)
            mask=tf.cast(mask,tf.float32)/255

            all_images.append((image,mask))
           
    return all_images
        
        
def display(img):
  display_list=[img[0],img[1]]
  # display_list=imgs
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()