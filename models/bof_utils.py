from torch.nn import parameter
import torch.nn.functional as F
import torch
import torch.nn as nn
import time

class LogisticConvBoF(nn.Module):
    
    def __init__(self, input_features, n_codewords, avg_horizon=2):
        super(LogisticConvBoF, self).__init__()
        self.input_features = input_features
        self.n_codewords = n_codewords
        self.codebook = nn.Conv2d(input_features, n_codewords, kernel_size=1)

        self.a = nn.Parameter(torch.FloatTensor(data=[1]))
        self.c = nn.Parameter(torch.FloatTensor(data=[0]))
        self.avg_horizon = avg_horizon

    def forward(self, input, eps=5e-16 ):
        # Step 1: Measure the similarity with each codeword
        x = self.codebook(input)
        
        # Step 2: Scale to ensure that the resulting value encodes the similarity
        # st1 = time.perf_counter()
        x = torch.tanh(x*self.a + self.c)
        x = (x + 1) / 2.0
        # ed1 = time.perf_counter()
        
        # Step 3: Create the similarity vectors for each of the input feature vectors
        # st2 = time.perf_counter()
        x = (x / (torch.sum(x, dim=1, keepdim=True)+ eps))
        # ed2 = time.perf_counter()
        
        # Step 4: Perform temporal pooling
        # st3 = time.perf_counter()
        x = F.adaptive_avg_pool2d(x, self.avg_horizon)
        x = x.reshape((x.size(0), -1))
        # ed3 = time.perf_counter()
        
        return x #, ed1 - st1, ed2 - st2, ed3 - st3


if __name__ == '__main__':
    model = LogisticConvBoF( 256,64,1 )
    for name, parameter in model.named_parameters():
        print( name )