#Import Statements
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.init as torch_init
import torch.optim as optim

from wavelet import wavelet_transform

class waveletCNN(nn.Module):



    def __init__(self, wvlt_transform):

        super(waveletCNN, self).__init__()

        self.wvlt_transform = wvlt_transform

        ### Main Convolutions ###
        #Define the k1 set of convolutions
        self.convk1a = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=3, padding=1)
        self.convk1a_normed = nn.BatchNorm2d(64)
        torch_init.xavier_normal_(self.convk1a.weight)
        self.convk1b = nn.Conv2d(in_channels=64,out_channels=64, kernel_size=2, padding=0, stride=2)
        self.convk1b_normed = nn.BatchNorm2d(64)
        torch_init.xavier_normal_(self.convk1b.weight)

        #Define the k2 set of convolutions
        self.convk2a = nn.Conv2d(in_channels=136, out_channels=128, kernel_size=3, padding=1)
        self.convk2a_normed = nn.BatchNorm2d(128)
        torch_init.xavier_normal_(self.convk2a.weight)
        self.convk2b = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1,stride=2)
        self.convk2b_normed = nn.BatchNorm2d(128)
        torch_init.xavier_normal_(self.convk2b.weight)

        #Dfine the k3 set of convolutions
        self.convk3a = nn.Conv2d(in_channels=396, out_channels=256, kernel_size=3, padding=1)
        self.convk3a_normed = nn.BatchNorm2d(256)
        torch_init.xavier_normal_(self.convk3a.weight)
        self.convk3b = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2, padding=0, stride=2)
        self.convk3b_normed = nn.BatchNorm2d(256)
        torch_init.xavier_normal_(self.convk3b.weight)


        ### Convolutions after successive Wavelet Transforms ###
        #Define the k1.5 set of convolutions
        self.convk15a = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=3, padding=1)
        self.convk15a_normed = nn.BatchNorm2d(64)
        torch_init.xavier_normal_(self.convk15a.weight)

        #Define the k2.5 set of convolutions
        self.convk25a = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=3, padding=1)
        self.convk25a_normed = nn.BatchNorm2d(64)
        torch_init.xavier_normal_(self.convk25a.weight)
        self.convk25b = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.convk25b_normed = nn.BatchNorm2d(128)
        torch_init.xavier_normal_(self.convk25b.weight)



        ### Skip Connections after main set convolutions ###
        self.convs1 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=1, stride=2)
        self.convs1_normed = nn.BatchNorm2d(4)
        torch_init.xavier_normal_(self.convs1.weight)

        self.convs2 = nn.Conv2d(in_channels=136, out_channels=136, kernel_size=1, stride=2)
        self.convs2_normed = nn.BatchNorm2d(136)
        torch_init.xavier_normal_(self.convs2.weight)

        self.convs3 = nn.Conv2d(in_channels=396, out_channels=396, kernel_size=1, stride=2)
        self.convs3_normed = nn.BatchNorm2d(396)
        torch_init.xavier_normal_(self.convs3.weight)


        ### Skips connections after wavelet transforms ###
        self.convs15 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=1)
        self.convs15_normed = nn.BatchNorm2d(4)
        torch_init.xavier_normal_(self.convs15.weight)

        self.convs25 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=1)
        self.convs25_normed = nn.BatchNorm2d(4)
        torch_init.xavier_normal_(self.convs25.weight)

        #Define gobal avg pooling layer
        self.pool = nn.AvgPool2d(kernel_size=19)

        #Define first fully connected layer
        self.fc1 = nn.Linear(in_features=1*1*652, out_features=128)
        self.fc1_normed = nn.BatchNorm1d(128)
        torch_init.xavier_normal_(self.fc1.weight)

        #Define second fully connected layer
        self.fc2 = nn.Linear(in_features=128, out_features=10).cuda()
        torch_init.xavier_normal_(self.fc2.weight)



    def forward(self, batch):

        #Compute first wavelet transform
        wvlt1_batch = self.wvlt_transform.batch_transform(batch)

        #Apply the k1 convolution layers
        cvk1a_batch = func.relu(self.convk1a_normed(self.convk1a(wvlt1_batch)))
        cvk1b_batch = func.relu(self.convk1b_normed(self.convk1b(cvk1a_batch)))

        #Apply the s1 skip connection
        cvs1_batch = func.relu(self.convs1_normed(self.convs1(wvlt1_batch)))

        #Apply the second wavelet transform
        wvlt2_batch = self.wvlt_transform.batch_transform(wvlt1_batch[:,0,:,:])

        #Apply the k1.5 convolution layers
        cvk15_batch = func.relu(self.convk15a_normed(self.convk15a(wvlt2_batch)))

        #Apply the s1.5 skip connection
        cvs15_batch = func.relu(self.convs15_normed(self.convs15(wvlt2_batch)))

        #Concatenate output of k1, s1, k1.5, and s1.5
        cvk1_dense_batch = torch.cat([cvk1b_batch, cvs1_batch, cvk15_batch, cvs15_batch], dim=1)

        #Apply the k2 convolution layers
        cvk2a_batch = func.relu(self.convk2a_normed(self.convk2a(cvk1_dense_batch)))
        cvk2b_batch = func.relu(self.convk2b_normed(self.convk2b(cvk2a_batch)))

        #Apply the s2 skip connection
        cvs2_batch = func.relu(self.convs2_normed(self.convs2(cvk1_dense_batch)))

        #Apply the 3rd wavelet transform
        wvlt3_batch = self.wvlt_transform.batch_transform(wvlt2_batch[:,0,:,:])

        #Apply the k2.5 convolution layers
        cvk25a_batch = func.relu(self.convk25a_normed(self.convk25a(wvlt3_batch)))
        cvk25b_batch = func.relu(self.convk25b_normed(self.convk25b(cvk25a_batch)))

        #Apply the s2.5 skip connection
        cvs25_batch = func.relu(self.convs25_normed(self.convs25(wvlt3_batch)))

        #Concatenate output of k2, s2, k2.5, and s2.5
        cvk2_dense_batch = torch.cat([cvk2b_batch, cvs2_batch, cvk25b_batch, cvs25_batch], dim=1)

        cvk3a_batch = func.relu(self.convk3a_normed(self.convk3a(cvk2_dense_batch)))
        cvk3b_batch = func.relu(self.convk3b_normed(self.convk3b(cvk3a_batch)))

        cvs3_batch =func.relu(self.convs3_normed(self.convs3(cvk2_dense_batch)))

        cvk3_dense_batch = torch.cat([cvk3b_batch, cvs3_batch], dim=1)

        fc_batch = self.pool(cvk3_dense_batch)

        fc_batch = fc_batch.view(-1, self.num_flat_features(fc_batch))

        fc_batch = func.relu(self.fc1_normed(self.fc1(fc_batch)))

        return self.fc2(fc_batch)


    def num_flat_features(self, inputs):
        #Get the dimensions of the layers excluding the inputs
        size = inputs.size()[1:]

        #Track the number of features
        num_features = 1

        for s in size:

            num_features *= s

        return num_features
