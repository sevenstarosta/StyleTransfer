from keras.applications import VGG19
from keras.models import Sequential
from keras.layers import Input
from keras.preprocessing import image
from random import randint
from keras.utils import plot_model
import numpy
from vis import optimizer

def gramMat(A):
    (l,x,y,N)=A.shape
    B=[[0 for a in range(N)]for b in range(N)]
    for map1 in range(N):
        for map2 in range(N):
            for a in range(x):
                for b in range(b):
                    B[map1][map2]+=(A[0][a][y][map1]*A[0][x][y][map2])
    return B

'Make B be the predict on original content image, make A be the input white noise image.'
def contentLoss(A,B):
    (l,x,y,N)=A.shape
    C=0
    for map in range(N):
        for a in range(x):
            for b in range(y):
                C+=(A[0][a][b][map]-B[0][a][b][map])**2
    C/=2
    return C

#change to pass in input image. Then run predict on truncated model.
'Make B be the predict on the original style image, resized to the size of the content image'
def styleLoss(A,B):
    C=0
    gramA=gramMat(A)
    gramB=gramMat(B)
    for x in range(len(gramA)):
        for y in range(len(gramA)):
            C+=(gramA[x][y]-gramB[x][y])**2
    C/= (4*len(gramA)**2)
    #not yet divided by Ml**2
    return C
    #assume A and B share the same shape for simplicity.
    
def whiteNoise(rows,cols):
    noise_matrix = numpy.empty(shape=[rows,cols,3], dtype= numpy.uint8)
    for r in range(0,rows):
        for c in range(0,cols):
            noise_matrix[r][c][0] = randint(200,255)
            noise_matrix[r][c][1] = randint(200,255)
            noise_matrix[r][c][2] = randint(200,255)
    return noise_matrix

img = image.load_img('starrynight.jpg')
contentImage=image.img_to_array(img)

noise=whiteNoise(contentImage.shape[0],contentImage.shape[1])

img = image.load_img('starrynight.jpg')
styleImage=image.img_to_array(img)
#resize styleimage


input_tensor = Input(shape=contentImage.shape)

trainedModel = VGG19(weights='imagenet',include_top= False,input_tensor=input_tensor)

contentModel = Sequential()
for layer in trainedModel.layers[:15]:
    contentModel.add(layer)
    
styleModel = Sequential()
for layer in trainedModel.layers[:13]:
    styleModel.add(layer)
    
contentBase = contentModel.predict(numpy.array([contentImage]))
styleBase = styleModel.predict(numpy.array([styleImage]))
#print(trainedModel.summary())
#for x in trainedModel.get_weights():
#    print(x.shape)
    
#print(contentImage.shape)
#print(contentModel.predict(numpy.array([contentImage]),1,0))

#print("SHAPE")
#appears to have the outputs of 512 neurons, which each roughly preserve aspect ratio of input image.
#print(contentModel.predict(numpy.array([contentImage]),1,0).shape)
#may be easier to convert...
#firstlayer=trainedModel.layers[1]
#print(firstlayer.get_weights())

#mean squared error for content.

trainedModel.save('trainedModel1.HDF5')