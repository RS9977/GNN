from model import *
from data import prepare_data_vocab, live_feat
from torch.nn.utils.rnn import unpack_sequence
from torch_geometric.data import Data


import torch.optim as optim
from torch_geometric.data import Data

from torch.utils.data import DataLoader

#read data
#word_to_idx: {'*': 0, '+': 1, '-': 2, '/': 3, 'M': 4, 'PHI': 5}
#   mapping from unique instruction to integer

#idx_to_word: {0: '*', 1: '+', 2: '-', 3: '/', 4: 'M', 5: 'PHI'}
#   inverse mapping of word_to_idx

#data
#   one entry per ssa.c file
#   data[i][0] - edge list (two node ids per edge)
#   data[i][1] - list of basic blocks with raw instructions in each BB
#   data[i][2] - same as data[i][1] but with args removed
#   BUG (introduced by SRA) - len(data[i][1]) != len(data[i][2])

#data_emb
#   same as data but with instructions replaced by uniq ids specified
#   in word_to_idx

word_to_idx, idx_to_word, data, data_emb = prepare_data_vocab("data", func=live_feat, function_num=0)
print(word_to_idx)

assert len(data)==len(data_emb)
for i in range(len(data)):
    assert len(data[i][2])==len(data_emb[i][1])

#define InputEncoder (RNN) and GNN
in_dim        = 8
hidden_dim    = 16
num_layers    = 7
bidirectional = True
out_dim       = 128

int_dir = 2 if bidirectional else 1

bb_enc = InputEncoder(word_to_idx, 
                      in_dim=in_dim, 
                      hidden_dim=hidden_dim, 
                      num_layers=num_layers, 
                      bidirectional=bidirectional)
 
gnn = GNNModel(in_dim=int_dir*hidden_dim,
               out_dim=out_dim,
               layer_dims=[16, 32])

#prediction/forward pass
#Step 1: rnn forward pass
graph = data_emb[0]
edge_index, node_embs = graph
edge_index = torch.tensor(edge_index).T

out, h, c = bb_enc(node_embs)
node_feats = torch.vstack([k[-1] for k in unpack_sequence(out)])

assert node_feats.shape == (len(node_embs), int_dir*hidden_dim)

#Step 2: gnn forward pass
graph = Data(edge_index=edge_index,
             x=node_feats)

assert graph.x.requires_grad
out = gnn(graph)
# more nodes specified in edge_list than in node_feats 



random_target = torch.rand((len(data_emb), 1, out_dim))
initial_node_embeddings = []
for graph in data_emb:
    edge_index, node_embs = graph
    #if torch.tensor(edge_index).shape[0]==0:
    if len(edge_index) != 0:
        edge_index = torch.tensor(edge_index).T
    else:
        edge_index = torch.tensor([[0],[0]])
    
    out, h, c = bb_enc(node_embs)
    node_feats = torch.vstack([k[-1] for k in unpack_sequence(out)])

    assert node_feats.shape == (len(node_embs), int_dir*hidden_dim)

            #Step 2: gnn forward pass
    graph = Data(edge_index=edge_index,
                        x=node_feats)

    initial_node_embeddings.append(torch.rand((graph.num_nodes, out_dim), requires_grad=True))


criterion = nn.MSELoss()
optimizer = optim.Adam(gnn.parameters(), lr=0.01)

epoch_num = 300
# Training loop



for epoch in range(epoch_num):
    total_loss = 0
    total_items = 0
    j = -1
    for graph in data_emb:
        j += 1
        edge_index, node_embs = graph
        #if torch.tensor(edge_index).shape[0]==0:
        if len(edge_index) != 0:
            edge_index = torch.tensor(edge_index).T
        else:
            edge_index = torch.tensor([[0],[0]])
            
        
        out, h, c = bb_enc(node_embs)
        node_feats = torch.vstack([k[-1] for k in unpack_sequence(out)])

        assert node_feats.shape == (len(node_embs), int_dir*hidden_dim)

                #Step 2: gnn forward pass
        graph = Data(edge_index=edge_index,
                            x=node_feats)
        
        output = gnn(graph)
        # Unsupervised loss (dummy loss for illustration)
        loss = criterion(output, random_target[j])
        #loss = criterion(output, initial_node_embeddings[j])

        # Backward pass and optimization
        total_loss  += loss.item()
        total_items += graph.x.shape[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch%10==0 or epoch==epoch_num-1:
            avg_loss = total_loss / len(data_emb)
            print(f'For {j+1} functions: Epoch {epoch+1}/{epoch_num}, Average Loss: {avg_loss:.5f}', end='\r')

        #total_loss += unsupervised_loss.item()

    # Print or log the average loss for monitoring
    if epoch%10==0 or epoch==epoch_num-1:
        print()
