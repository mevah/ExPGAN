# import torch
# from torch.autograd import Variable

# Tensor = torch.FloatTensor

# adversarial_loss = torch.nn.BCELoss(reduction='sum')
# adversarial_loss_mean = torch.nn.BCELoss(reduction='mean')
# true_lbl = Variable(Tensor(8, 1).fill_(1), requires_grad=False)
# fake_lbl = Variable(Tensor(8, 1).fill_(0), requires_grad=False)
# lbls = torch.squeeze(torch.cat((true_lbl, fake_lbl), dim=0))
            
# lbl_est_left = torch.squeeze(torch.cat((true_lbl, true_lbl), dim=0))
# print('lbl_est_left: ', lbl_est_left)
# print('lbls: ', lbls)
# loss_D_left = adversarial_loss(torch.squeeze(lbl_est_left), torch.squeeze(lbls))
# print(loss_D_left)
# loss_D_left = adversarial_loss_mean(torch.squeeze(lbl_est_left), torch.squeeze(lbls))
# print(loss_D_left)


from math import log
from numpy import mean
 
# calculate cross entropy
def cross_entropy(p, q):
	return -sum([p[i]*log(q[i]) for i in range(len(p))])
 
# define classification data
#q = [1, 1, 1, 1, 1, 1, 1, 1]
#p = [1, 1, 1, 1, 0.000001, 0.000001, 0.000001, 0.000001]

#0.9999
p = [1, 1, 1, 1, 0, 0, 0, 0]
q = [0.99999,0.99999,0.99999,0.99999,0.99999,0.999999,0.99999,0.99999]
q = [0.999999999999, 0.999999999999, 0.999999999999, 0.999999999999, 0.999999999999, 0.999999999999, 0.999999999999, 0.999999999999]

# calculate cross entropy for each example
results = list()
for i in range(len(p)):
	# create the distribution for each event {0, 1}
	expected = [1.0 - p[i], p[i]]
	predicted = [1.0 - q[i], q[i]]
	# calculate cross entropy for the two events
	ce = cross_entropy(expected, predicted)
	print('>[y=%.1f, yhat=%.1f] ce: %.3f nats' % (p[i], q[i], ce))
	results.append(ce)
 
# calculate the average cross entropy
mean_ce = mean(results)
print('Average Cross Entropy: %.3f nats' % mean_ce)
