import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from imutils import paths

dataset_cat = r'C:\Users\Ghazal\Documents\GitHub\Cats_and_dogs-CNN\PetImages\Cat'
imagePaths = list(paths.list_images(dataset_cat))
print(imagePaths)



X = ...
Y = np.array['cat','dog']
lable = LabelEncoder()
y_lable=lable.fit(Y)

NNModel = MLPClassifier(
    hidden_layer_sizes=(32,16),
    activation='relu',
    solver='adam',
    max_iter=1000
)
NNModel.fit(y_lable)

pred = ...