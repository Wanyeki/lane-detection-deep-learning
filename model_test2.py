#running the unet
from unet import unet_architecture

input_shape=(960,720,3)
model=unet_architecture(input_shape)
model.summary()