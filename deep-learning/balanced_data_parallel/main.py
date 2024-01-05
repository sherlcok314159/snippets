import torch
import torch.nn as nn
from utils import BalancedDataParallel

class Model(nn.Module):
    # Our model
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
		# for convenience
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print('\tIn Model: input size', input.size(),
              'output size', output.size())
        return output

# change parameters due to the number of your devices
gpu0_bsz = 14
bs, input_size, output_size = 32, 8, 10
inputs = torch.randn((bs, input_size)).cuda()
model = Model(input_size, output_size)
if torch.cuda.device_count() > 1:
    print('Use', torch.cuda.device_count(), 'GPUs!')
    # dim=0 => [32, xxx] -> [14, ...], [18, ...] on 2 GPUS
    model = BalancedDataParallel(gpu0_bsz, model, dim=0)

model = model.cuda()
outputs = model(inputs)
print('Outside: input size', inputs.size(),
	  'output_size', outputs.size())

# assume 2 GPUS are available
# Use 2 GPUs!
#    In Model: input size torch.Size([14, 8]) output size torch.Size([14, 10])
#    In Model: input size torch.Size([18, 8]) output size torch.Size([18, 10])
# Outside: input size torch.Size([32, 8]) output_size torch.Size([32, 10])