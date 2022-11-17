import torch
import torch.nn as nn

class Model(nn.Module):
    # Our model
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
		# for convenience
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("\tIn Model: input size", input.size(),
              "output size", output.size())
        return output

PATH = './model.bin'
bs, input_size, output_size = 6, 8, 10
# define inputs
inputs = torch.randn((bs, input_size)).cuda()
model = Model(input_size, output_size)
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [6, xxx] -> [2, ...], [2, ...], [2, ...] on 3 GPUs
  model = nn.DataParallel(model)
# 先DataParallel，再cuda
model = model.cuda()
outputs = model(inputs)
print("Outside: input size", inputs.size(),
	  "output_size", outputs.size())
# assume 2 GPUS are available
# Let's use 2 GPUs!
#    In Model: input size torch.Size([3, 8]) output size torch.Size([3, 10])
#    In Model: input size torch.Size([3, 8]) output size torch.Size([3, 10])
# Outside: input size torch.Size([6, 8]) output_size torch.Size([6, 10])

# save the model
torch.save(model.module.state_dict(), PATH)
# load again
model.module.load_state_dict(torch.load(PATH))
# do anything you want