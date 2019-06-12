import pywt
import torch as torch



class wavelet_transform():

    def __init__(self, wavelet_name, computing_device):

        self.wavelet_name = wavelet_name
        self.computing_device = computing_device


    #Given an input image, apply the wavelet transform over the numpy array
    #and then output at as a 3D Torch tensor
    def transform(self, data):

        #Unwrap data from tensor and compute wavelet transform
        coeffs = pywt.dwt2(data.cpu().numpy(), self.wavelet_name)


        #Create a 3 dimensional tensor, where each output coefficients array is stacked
        #Ordering is LL, LH, HL HH
        coeffs_var =  torch.stack((torch.Tensor(coeffs[0]).to(self.computing_device),
                            torch.Tensor(coeffs[1][0]).to(self.computing_device),
                            torch.Tensor(coeffs[1][1]).to(self.computing_device),
                            torch.Tensor(coeffs[1][2]).to(self.computing_device)), dim=0)

        print(coeffs_var.shape)

        return coeffs_var

    #Apply the wavelet transform over an image batch
    #Returns a 4D Torch tensor of dimensions [batch_size x height x width x filter_dim]
    def batch_transform(self, batch):

        batch_transf = torch.stack(list(map(self.transform, torch.unbind(batch, 0))), dim=0)

        print(batch_transf.shape)

        squeezed_batch_trans = torch.squeeze(batch_transf)
        print(squeezed_batch_trans.shape)


        return squeezed_batch_trans




