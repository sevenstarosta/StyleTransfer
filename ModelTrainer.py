from keras.applications import VGG19
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, Input
from keras.preprocessing import image

def gramMat(A):
    (l,x,y,N)=A.shape
    B=[[0 for a in range(N)]for b in range(N)]
    for map1 in range(N):
        for map2 in range(N):
            for a in range(x):
                for b in range(b):
                    B[map1][map2]+=(A[0][a][y][map1]*A[0][x][y][map2])
    return B

img = image.load_img('starrynight.jpg')
#need to normalize. Also need to figure out preprocessing? Add top layers or no?
input_tensor = Input(shape=(150,150,3))
trainedModel = VGG19(weights='imagenet',include_top= False,input_tensor=input_tensor)
for layers in trainedModel.layers:
    pass

#may be easier to convert...
firstlayer=trainedModel.layers[1]
print(firstlayer.get_weights())

trainedModel.save('trainedModel1.HDF5')