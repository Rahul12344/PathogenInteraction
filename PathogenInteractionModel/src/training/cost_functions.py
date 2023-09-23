import torch.nn as nn
import torch
    
# Compute Elastic Net Loss with L1 and L2 regularization
class ElasticNetBCELoss(nn.Module):
    
    # initialize the cost function
    def __init__(self, **kwargs):
        super(ElasticNetBCELoss, self).__init__()
        # set the parameters
        
        self.c = kwargs['c']
        self.device = kwargs['device']
        self.alpha = kwargs['alpha']
        self.beta = kwargs['beta']
        self.loss = nn.BCELoss()
        self.to(self.device)

    # compute the cost function
    def forward(self, y_pred, y_true, parameters):
        loss = self.loss(y_pred, y_true).squeeze(0).to(self.device)
        
        # compute the L1 regularization
        l1 = 0
        for param in parameters:
            l1 = l1 + param.abs().sum()
            
        # compute the L2 regularization
        l2 = 0
        for param in parameters:
            l2 = l2 + torch.square(torch.linalg.norm(param))
            
        # compute the elastic net regularization
        loss = loss + self.alpha * l1 + self.beta * l2
        
        return loss