import argparse
from copy import deepcopy
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from model import Model
from dataset import Dataset

# Sets the device to cuda which allows us to run the code on a gpu
device = torch.device("cuda")

# Function that trains the model 
def train(dataset, model, args):
	print(f"Training with rating: {args.rt_num}")
	model.train()
	
	dataloader = DataLoader(dataset, batch_size=args.batch_size)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=0.001)
	
    # Loops the training for a set amount of epochs
	for epoch in range(args.max_epochs):
		state_h, state_c = model.init_state(args.sequence_length)
		state_h = state_h.to(device)
		state_c = state_c.to(device)
		for batch, (x, y) in enumerate(dataloader):
			optimizer.zero_grad()
			
			y_pred, (state_h, state_c) = model(x, (state_h, state_c))
			loss = criterion(y_pred.transpose(1, 2), y)
			
			state_h = state_h.detach()
			state_c = state_c.detach()
			loss.backward()
			optimizer.step()
            # saves the best state during the epoch
			best_model_state = deepcopy(model.state_dict())
		print({ 'epoch': epoch, 'batch': batch, 'loss': loss.item()})
		
		# prints epoch nr. and loss to a document
		sel = open('States/data.txt', 'a')
		sel.write(f"M{args.rt_num}Epoch: {epoch}, Loss: {loss.item()} \n")
		sel.close()
        # saves the model state
		torch.save(best_model_state, f"States/Model_{args.rt_num}_star2.0")
		model.load_state_dict(best_model_state)
        
parser = argparse.ArgumentParser()
# added an extra argument to specify which rating to run
parser.add_argument("--rt_num", type=int, default = 1)
parser.add_argument('--max-epochs', type=int, default = 30)
parser.add_argument('--batch-size', type=int, default = 2048)
parser.add_argument('--sequence-length', type=int, default = 4)
args = parser.parse_args()

dataset = Dataset(args)
model = Model(dataset)

# uncoment to load a model state
#model.load_state_dict(torch.load(f"States/Model_{args.rt_num}_star", map_location="cuda"))
model.to(device)



train(dataset, model, args)
