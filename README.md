TensorFlow works with Python 2.7 and Python 3.3+.
TensorFlow installation comes with a number of Deep Learning models that you can use and experiment with directly. It is a library for fast numerical computation, specifically designed for the types of operations that are required in the development and evaluation of large deep learning models.
Face recognition procedure:

The tensorflow and matplotlib can be installed in jupyter notebook using the following commands:
pip install tensorflow
This takes a few minutes to install.
pip install matplotlib

Face recognition procedure:
Initially, the required libraries are imported and the image size is defined as a 224x224 image. The paths of the training and validation images are given. VGG16 model is used for this purpose and ‘imagenet’ weights are used.
The glob function is used to get the number of classes. Flatten and dense layers are added and softmax activation function is used. 
Model compilation: The following parameters are used:
         Loss function: Categorical cross entropy
         Optimizer: Adam
         The accuracy is printed for each epoch.
         The images are normalized and the batch size for training and testing     data are specified.
The model is trained by specifying the number of epochs and saved after execution. Then the model is loaded and the region of interest is obtained by using the haarcascade frontal face default classifier.
Then, the rectangles are drawn around the faces. The webcam is turned on and the obtained image is resized and converted into a numpy array. 
The output image is compared with the folders in the dataset. If there is a match, the corresponding name is printed. Otherwise, “none matching” is printed.
More images have to be given for training. In case of lack of availability of data, the number of epochs have to be increased for accurate prediction.

Creating dataset: Dataset for different people for comparision can be downloaded anywhere from the internet.
Our own images can be clicked using webcam.
All these data can be used for training the model.
Different datasets are available in youtube video descriptions.
