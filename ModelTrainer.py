from keras.applications import VGG19
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, Input
from keras.preprocessing import image
import numpy

def gramMat(A):
    (l,x,y,N)=A.shape
    B=[[0 for a in range(N)]for b in range(N)]
    for map1 in range(N):
        for map2 in range(N):
            for a in range(x):
                for b in range(b):
                    B[map1][map2]+=(A[0][a][y][map1]*A[0][x][y][map2])
    return B

def contentLoss(A,B):
    (l,x,y,N)=A.shape
    C=0
    for map in range(N):
        for a in range(x):
            for b in range(y):
                C+=
    #assume A and B share the same shape for simplicity.
    

img = image.load_img('starrynight.jpg')
#if content and style images are of different sizes, resize the style image to the content image size.
contentImage=image.img_to_array(img)

input_tensor = Input(shape=contentImage.shape)

trainedModel = VGG19(weights='imagenet',include_top= False,input_tensor=input_tensor)

contentModel = Sequential()

for layer in trainedModel.layers[:15]:
    contentModel.add(layer)

print(contentImage.shape)
print()

print(contentModel.predict(numpy.array([contentImage]),1,0))

print("SHAPE")
#appears to have the outputs of 512 neurons, which each roughly preserve aspect ratio of input image.
print(contentModel.predict(numpy.array([contentImage]),1,0).shape)
#may be easier to convert...
#firstlayer=trainedModel.layers[1]
#print(firstlayer.get_weights())

trainedModel.save('trainedModel1.HDF5')