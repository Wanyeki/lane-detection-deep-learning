from unet2 import unet
model=unet(input_size=(512,512,3))
model.summary()