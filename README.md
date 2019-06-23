# wavelet_cnn
These Python files train and test a Wavelet Convolutional Neural Network written in Pytorch. To run the file, call 
$python3 main.py

The following dependencies are required:
torch
torchvision
cuda
pywavelets
numpy
pandas

Additionally, the CalTech101 dataset was used. Once downloaded, call 
$python3 datapreprocess.py $IMG_DIR $CSV_OUT 
to get generate an appropriate CSV like the one used in this investigation.
