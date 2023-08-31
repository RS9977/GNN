import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn.aggr import MeanAggregation

def get_crit_opt(net):
    criterion = nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.AdamW(net.parameters(), lr=1e-4)

    return criterion, optimizer

def train(model, 
          dl,  #dataloader
          optimizer, 
          criterion,
          n_epochs,
          print_freq=1):
    
    aggr = MeanAggregation()
    
    model.train()
    for i in range(n_epochs):

        total_loss = 0
        total_items = 0

        for idx, data in enumerate(dl):
            pred = model(data)
            pred = aggr(pred, ptr=data.ptr) #computes per-graph embedding

            assert pred.shape[0]<=dl.batch_size, f"pred.shape={pred.shape}"

            loss = criterion(pred, 
                             data.y)
            
            total_loss += loss.item()
            total_items += data.x.shape[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if i % print_freq == 0:
            print(f'Epoch = {i} Training Loss = {total_loss:.3f}')

    return model, optimizer

# Define the evaluation function
def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        output = model(data)
        predicted_labels = output > 0.5  # Threshold the output for binary predictions
        accuracy = (predicted_labels == data.y).all(dim=1).float().mean()
    return accuracy.item()