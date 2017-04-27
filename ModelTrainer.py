from keras.applications import VGG19
from keras.models import Sequential
from keras.layers import Input
from keras.preprocessing import image
from random import randint
from keras.utils import plot_model
from vis.optimizer import Optimizer
from vis import losses
from matplotlib import pyplot
import numpy
#import cv2

class contentLoss(losses.Loss):
    #pass cmodel.input as array. Pass model. Pass already predicted output.
    def __init__(self,model,input,predicted):
        super(contentLoss,self).__init__()
        self.name="Content Loss Function"
        self.model=model
        self.input=input
        self.predicted=predicted
        
    def build_loss(self):
        A=self.model.predict(numpy.array([self.input]))
        B=self.predicted
        (l,x,y,N)=A.shape
        C=0
        for map in range(N):
            for a in range(x):
                for b in range(y):
                    C+=(A[0][a][b][map]-B[0][a][b][map])**2
        C/=2
        return C
        
class styleLoss(losses.Loss):
    def __init__(self,model,input,predicted):
        super(styleLoss,self).__init__()
        self.name="Style Loss Function"
        self.model=model
        self.input=input
        self.predicted=predicted
        
    def build_loss(self):
        pred=self.model.predict(numpy.array([self.input]))
        (l,x,y,N)=pred.shape
        A=gramMat(pred)
        C=0
        for x in range(N):
            for y in range(N):
                C+=(A[x][y]-self.predicted[x][y])**2
        C/= (4*(N**2)*((x*y)**2))
        return C
        
def gramMat(A):
    (l,x,y,N)=A.shape
    B=[[0 for a in range(N)]for b in range(N)]
    for map1 in range(N):
        for map2 in range(N):
            for a in range(x):
                for b in range(y):
                    B[map1][map2]+=(A[0][a][b][map1]*A[0][a][b][map2])
    return B

#change to pass in input image. Then run predict on truncated model.
'Make B be the predict on the original style image, resized to the size of the content image'
    
def whiteNoise(rows,cols):
    noise_matrix = numpy.empty(shape=[rows,cols,3], dtype= numpy.uint8)
    for r in range(0,rows):
        for c in range(0,cols):
            noise_matrix[r][c][0] = randint(200,255)
            noise_matrix[r][c][1] = randint(200,255)
            noise_matrix[r][c][2] = randint(200,255)
    return noise_matrix

img = image.load_img('d53.jpg')
contentImage=image.img_to_array(img)

noise=whiteNoise(contentImage.shape[0],contentImage.shape[1])

img = image.load_img('starrynight256.jpg')
styleImage=image.img_to_array(img)
#styleImage=cv2.resize(styleImage,None,(contentImage.shape[0],contentImage.shape[1]))
#resize styleimage


input_tensor = Input(shape=contentImage.shape)

trainedModel = VGG19(weights='imagenet',include_top= False,input_tensor=input_tensor)

layer_dict = dict([(layer.name, layer) for layer in trainedModel.layers[1:]])

contentModel = Sequential()
#print(trainedModel.summary())

for layer in trainedModel.layers[:14]:
    contentModel.add(layer)
    
styleModel1 = Sequential()
for layer in trainedModel.layers[:2]:
    styleModel1.add(layer)
    
styleModel2 = Sequential()
for layer in trainedModel.layers[:5]:
    styleModel2.add(layer)
    
styleModel3 = Sequential()
for layer in trainedModel.layers[:8]:
    styleModel3.add(layer)
    
styleModel4 = Sequential()
for layer in trainedModel.layers[:13]:
    styleModel4.add(layer)
    
#styleModel5 = Sequential()
#for layer in trainedModel.layers[:8]:
#styleModel5.add(layer)
    
#must create grammat of output of style image on style model
contentBase = contentModel.predict(numpy.array([contentImage]))
styleBase1 = gramMat(styleModel1.predict(numpy.array([styleImage])))
styleBase2 = gramMat(styleModel2.predict(numpy.array([styleImage])))
styleBase3 = gramMat(styleModel3.predict(numpy.array([styleImage])))
styleBase4 = gramMat(styleModel4.predict(numpy.array([styleImage])))
#styleBase5 = gramMat(styleModel5.predict(numpy.array([styleImage])))

losses = [
    (contentLoss(contentModel,contentModel.input,contentBase),1),
    (styleLoss(styleModel1,contentModel.input,styleBase1),1000),
    (styleLoss(styleModel2,contentModel.input,styleBase2),1000),
    (styleLoss(styleModel3,contentModel.input,styleBase3),1000),
    (styleLoss(styleModel4,contentModel.input,styleBase4),1000)
]
opt = Optimizer(contentModel.input,losses)    
finaloutput=opt.minimize(seed_img=noise,max_iter=25,verbose=True)[0]
print("finished")
pyplot.imshow(finaloutput)
pyplot.show()
print("done with showing")

#print((contentBase[0]).shape)
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