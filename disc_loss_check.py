import torch
from torch.autograd import Variable

Tensor = torch.FloatTensor

adversarial_loss = torch.nn.BCELoss()
true_lbl = Variable(Tensor(8, 1).fill_(1), requires_grad=False)
fake_lbl = Variable(Tensor(8, 1).fill_(0), requires_grad=False)
lbls = torch.squeeze(torch.cat((true_lbl, fake_lbl), dim=0))
            
lbl_est_left = torch.squeeze(torch.cat((true_lbl, true_lbl), dim=0))
print('lbl_est_left: ', lbl_est_left)
print('lbls: ', lbls)
loss_D_left = adversarial_loss(torch.squeeze(lbl_est_left), torch.squeeze(lbls))
print(loss_D_left)
