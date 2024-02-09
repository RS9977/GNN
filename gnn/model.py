import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn.aggr import MeanAggregation
from torch.nn.utils.rnn import pad_sequence,\
                                pack_sequence,\
                                pad_packed_sequence
from torch_geometric.data import Data
from torch.nn.utils.rnn import unpack_sequence
import copy

# Define the GNN architecture
class GNNModel(nn.Module):
    def __init__(self,
                 in_dim=2,
                 out_dim=20,
                 layer_dims=[16, 32],
                 name="gnn"):

        super().__init__()

        layer_dims = [in_dim] + layer_dims

        self.conv_layers = nn.ModuleList({})

        for idx in range(1, len(layer_dims)):

            layer = GCNConv(in_channels=layer_dims[idx-1],
                            out_channels=layer_dims[idx])
            
            self.conv_layers.append(layer)

        self.fc = nn.Linear(layer_dims[-1], out_dim)

        self.aggr = MeanAggregation()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        for l in self.conv_layers:
            x = l(x, edge_index)
            x = x.relu()

        x = self.fc(x)
        
        '''
        if 'ptr' in data:
            x = self.aggr(x, ptr=data.ptr) #computes per-graph embedding

            assert x.shape[0]==len(data.ptr)-1, f"pred.shape={pred.shape}"
        else:
            x = x.mean(dim=0).unsqueeze(0) #average across nodes
        '''
        return x
    
# Define the encoding model (RNN) architecture
class InputEncoder(nn.Module):
    def __init__(self,
                 word_to_idx,
                 in_dim,
                 hidden_dim,
                 num_layers,
                 bidirectional=True):
        
        super().__init__()
        
        self.word_to_idx = word_to_idx
        vocab_size = len(word_to_idx)

        self.emb = nn.Embedding(num_embeddings=vocab_size+1,
                                embedding_dim=in_dim,
                                padding_idx=-1)
        
        self.lstm = nn.LSTM(input_size=in_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=bidirectional)
                
    
    def forward(self, x):
        #x should be per-node bb ins indices
        embs = pack_sequence([self.emb(bb) for bb in x], enforce_sorted=False)
        
        out, (h_n, c_n) = self.lstm(embs)

        return out, h_n, c_n


class IntegratedModel(nn.Module):
    def __init__(self, 
                 word_to_idx,
                 in_dim_lstm,
                 hidden_dim,
                 num_layers,
                 bidirectional=True,
                 out_dim=20,
                 layer_dims=[16, 32]):
        
        super().__init__()
        self.lstm_encoder = InputEncoder(word_to_idx, 
                            in_dim=in_dim_lstm, 
                            hidden_dim=hidden_dim, 
                            num_layers=num_layers, 
                            bidirectional=bidirectional)
        
        self.gnn_model = GNNModel(in_dim=(int(bidirectional)+1)*hidden_dim,
                         out_dim=out_dim,
                         layer_dims=[16, 32])
    
    def forward(self, node_sequences, edge_index):
        # node_sequences: shape (num_nodes, seq_length, lstm_input_size)
        # edge_index: shape (2, num_edges)
        out, h, c = self.lstm_encoder(node_sequences)
        node_feats = torch.vstack([k[-1] for k in unpack_sequence(out)])
        original_array = node_feats.detach().numpy()
        deep_copied_array = copy.deepcopy(original_array)
        out_act = torch.from_numpy(deep_copied_array)
        node_feats[:,16:] = -1
        graph = Data(edge_index=edge_index,
                        x=node_feats)
        gnn_output = self.gnn_model(graph)
        
        return gnn_output, out_act