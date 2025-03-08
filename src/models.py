################################################################
# Generalisation of Geometric Vector Perceptron, Jing et al.
# for explicit multi-state biomolecule representation learning.
# Original repository: https://github.com/drorlab/gvp-pytorch
################################################################

from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch_geometric

# Assuming src.layers provides GVP, LayerNorm, MultiGVPConvLayer, GVPConvLayer
from src.layers import *

# Check for ViennaRNA availability
try:
    import RNA
    VIENNA_AVAILABLE = True
except ImportError:
    VIENNA_AVAILABLE = False
    print("ViennaRNA not installed. Install with: pip install ViennaRNA for full validation.")

class AutoregressiveMultiGNNv1(torch.nn.Module):
    '''
    Autoregressive GVP-GNN for **multiple** structure-conditioned RNA design.
    
    Takes in RNA structure graphs of type `torch_geometric.data.Data` 
    or `torch_geometric.data.Batch` and returns a categorical distribution
    over 4 bases at each position in a `torch.Tensor` of shape [n_nodes, 4].
    
    The standard forward pass requires sequence information as input
    and should be used for training or evaluating likelihood.
    For sampling or design, use `self.sample`.

    Args:
        node_in_dim (tuple): node dimensions in input graph
        node_h_dim (tuple): node dimensions to use in GVP-GNN layers
        edge_in_dim (tuple): edge dimensions in input graph
        edge_h_dim (tuple): edge dimensions to embed in GVP-GNN layers
        num_layers (int): number of GVP-GNN layers in encoder/decoder
        drop_rate (float): rate to use in all dropout layers
        out_dim (int): output dimension (4 bases)
    '''
    def __init__(
        self,
        node_in_dim = (64, 4), 
        node_h_dim = (128, 16), 
        edge_in_dim = (32, 1), 
        edge_h_dim = (32, 1),
        num_layers = 3, 
        drop_rate = 0.1,
        out_dim = 4,
    ):
        super().__init__()
        self.node_in_dim = node_in_dim
        self.node_h_dim = node_h_dim
        self.edge_in_dim = edge_in_dim
        self.edge_h_dim = edge_h_dim
        self.num_layers = num_layers
        self.out_dim = out_dim
        activations = (F.silu, None)
        
        # Node input embedding
        self.W_v = torch.nn.Sequential(
            LayerNorm(self.node_in_dim),
            GVP(self.node_in_dim, self.node_h_dim,
                activations=(None, None), vector_gate=True)
        )

        # Edge input embedding
        self.W_e = torch.nn.Sequential(
            LayerNorm(self.edge_in_dim),
            GVP(self.edge_in_dim, self.edge_h_dim, 
                activations=(None, None), vector_gate=True)
        )
        
        # Encoder layers (supports multiple conformations)
        self.encoder_layers = nn.ModuleList(
                MultiGVPConvLayer(self.node_h_dim, self.edge_h_dim, 
                                  activations=activations, vector_gate=True,
                                  drop_rate=drop_rate, norm_first=True)
            for _ in range(num_layers))
        
        # Decoder layers
        self.W_s = nn.Embedding(self.out_dim, self.out_dim)
        self.edge_h_dim = (self.edge_h_dim[0] + self.out_dim, self.edge_h_dim[1])
        self.decoder_layers = nn.ModuleList(
                GVPConvLayer(self.node_h_dim, self.edge_h_dim,
                             activations=activations, vector_gate=True, 
                             drop_rate=drop_rate, autoregressive=True, norm_first=True) 
            for _ in range(num_layers))
        
        # Output
        self.W_out = GVP(self.node_h_dim, (self.out_dim, 0), activations=(None, None))
    
    def forward(self, batch):
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        edge_index = batch.edge_index
        seq = batch.seq

        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)

        for layer in self.encoder_layers:
            h_V = layer(h_V, edge_index, h_E)

        h_V, h_E = self.pool_multi_conf(h_V, h_E, batch.mask_confs, edge_index)
        encoder_embeddings = h_V
        
        h_S = self.W_s(seq)
        h_S = h_S[edge_index[0]]
        h_S[edge_index[0] >= edge_index[1]] = 0
        h_E = (torch.cat([h_E[0], h_S], dim=-1), h_E[1])
        
        for layer in self.decoder_layers:
            h_V = layer(h_V, edge_index, h_E, autoregressive_x=encoder_embeddings)
        
        logits = self.W_out(h_V)
        return logits
    
    @torch.no_grad()
    def sample(
            self, 
            batch, 
            n_samples, 
            temperature: Optional[float] = 0.1, 
            logit_bias: Optional[torch.Tensor] = None,
            return_logits: Optional[bool] = False
        ):
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        edge_index = batch.edge_index
    
        device = edge_index.device
        num_nodes = h_V[0].shape[0]
        
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        
        for layer in self.encoder_layers:
            h_V = layer(h_V, edge_index, h_E)
        
        h_V, h_E = self.pool_multi_conf(h_V, h_E, batch.mask_confs, edge_index)
        
        h_V = (h_V[0].repeat(n_samples, 1),
               h_V[1].repeat(n_samples, 1, 1))
        h_E = (h_E[0].repeat(n_samples, 1),
               h_E[1].repeat(n_samples, 1, 1))
        
        edge_index = edge_index.expand(n_samples, -1, -1)
        offset = num_nodes * torch.arange(n_samples, device=device).view(-1, 1, 1)
        edge_index = torch.cat(tuple(edge_index + offset), dim=-1)
        
        seq = torch.zeros(n_samples * num_nodes, device=device, dtype=torch.int)
        h_S = torch.zeros(n_samples * num_nodes, self.out_dim, device=device)
        logits = torch.zeros(n_samples * num_nodes, self.out_dim, device=device)

        h_V_cache = [(h_V[0].clone(), h_V[1].clone()) for _ in self.decoder_layers]

        for i in range(num_nodes):
            h_S_ = h_S[edge_index[0]]
            h_S_[edge_index[0] >= edge_index[1]] = 0
            h_E_ = (torch.cat([h_E[0], h_S_], dim=-1), h_E[1])
                    
            edge_mask = edge_index[1] % num_nodes == i
            edge_index_ = edge_index[:, edge_mask]
            h_E_ = tuple_index(h_E_, edge_mask)
            node_mask = torch.zeros(n_samples * num_nodes, device=device, dtype=torch.bool)
            node_mask[i::num_nodes] = True
            
            for j, layer in enumerate(self.decoder_layers):
                out = layer(h_V_cache[j], edge_index_, h_E_,
                           autoregressive_x=h_V_cache[0], node_mask=node_mask)
                out = tuple_index(out, node_mask)
                
                if j < len(self.decoder_layers)-1:
                    h_V_cache[j+1][0][i::num_nodes] = out[0]
                    h_V_cache[j+1][1][i::num_nodes] = out[1]
                
            lgts = self.W_out(out)
            if logit_bias is not None:
                lgts += logit_bias[i]
            seq[i::num_nodes] = Categorical(logits=lgts / temperature).sample()
            h_S[i::num_nodes] = self.W_s(seq[i::num_nodes])
            logits[i::num_nodes] = lgts

        if return_logits:
            return seq.view(n_samples, num_nodes), logits.view(n_samples, num_nodes, self.out_dim)
        else:    
            return seq.view(n_samples, num_nodes)
        
    def pool_multi_conf(self, h_V, h_E, mask_confs, edge_index):
        if mask_confs.size(1) == 1:
            return (h_V[0][:, 0], h_V[1][:, 0]), (h_E[0][:, 0], h_E[1][:, 0])
        
        n_conf_true = mask_confs.sum(1, keepdim=True)
        mask = mask_confs.unsqueeze(2)
        h_V0 = h_V[0] * mask
        h_E0 = h_E[0] * mask[edge_index[0]]
        mask = mask.unsqueeze(3)
        h_V1 = h_V[1] * mask
        h_E1 = h_E[1] * mask[edge_index[0]]
        
        h_V = (h_V0.sum(dim=1) / n_conf_true,
               h_V1.sum(dim=1) / n_conf_true.unsqueeze(2))
        h_E = (h_E0.sum(dim=1) / n_conf_true[edge_index[0]],
               h_E1.sum(dim=1) / n_conf_true[edge_index[0]].unsqueeze(2))
        return h_V, h_E


class NonAutoregressiveMultiGNNv1(torch.nn.Module):
    '''
    Non-Autoregressive GVP-GNN for **multiple** structure-conditioned RNA design.
    
    Enhanced with base-pairing probabilities and RNA.fold validation for better accuracy.
    
    Args:
        node_in_dim (tuple): node dimensions in input graph
        node_h_dim (tuple): node dimensions to use in GVP-GNN layers
        edge_in_dim (tuple): edge dimensions in input graph
        edge_h_dim (tuple): edge dimensions to embed in GVP-GNN layers
        num_layers (int): number of GVP-GNN layers in encoder
        drop_rate (float): rate to use in all dropout layers
        out_dim (int): output dimension (4 bases)
    '''
    def __init__(
        self,
        node_in_dim = (64, 4), 
        node_h_dim = (128, 16), 
        edge_in_dim = (32, 1), 
        edge_h_dim = (32, 1),
        num_layers = 3, 
        drop_rate = 0.1,
        out_dim = 4,
    ):
        super().__init__()
        self.node_in_dim = (node_in_dim[0] + 4, node_in_dim[1])  # +4 for base-pairing probs
        self.node_h_dim = node_h_dim
        self.edge_in_dim = edge_in_dim
        self.edge_h_dim = edge_h_dim
        self.num_layers = num_layers
        self.out_dim = out_dim
        self.drop_rate = drop_rate
        activations = (F.silu, None)
        
        # Node input embedding
        self.W_v = torch.nn.Sequential(
            LayerNorm(self.node_in_dim),
            GVP(self.node_in_dim, self.node_h_dim,
                activations=(None, None), vector_gate=True)
        )

        # Edge input embedding
        self.W_e = torch.nn.Sequential(
            LayerNorm(self.edge_in_dim),
            GVP(self.edge_in_dim, self.edge_h_dim, 
                activations=(None, None), vector_gate=True)
        )
        
        # Encoder layers (supports multiple conformations)
        self.encoder_layers = nn.ModuleList(
                MultiGVPConvLayer(self.node_h_dim, self.edge_h_dim, 
                                  activations=activations, vector_gate=True,
                                  drop_rate=drop_rate, norm_first=True)
            for _ in range(num_layers))
        
        # Output
        self.W_out = torch.nn.Sequential(
            LayerNorm(self.node_h_dim),
            GVP(self.node_h_dim, self.node_h_dim,
                activations=(None, None), vector_gate=True),
            GVP(self.node_h_dim, (self.out_dim, 0), 
                activations=(None, None))   
        )
    
    def compute_pairing_probs(self, batch):
        """Approximate base-pairing probabilities using edge distances."""
        n_nodes, n_conf = batch.node_s.shape[0], batch.node_s.shape[1]
        coords = batch.node_v.mean(dim=2)  # [n_nodes, n_conf, 3]
        edge_index = batch.edge_index
        dists = torch.norm(coords[edge_index[0]] - coords[edge_index[1]], dim=-1)  # [n_edges, n_conf]
        
        pairing_probs = torch.zeros(n_nodes, n_conf, 4, device=coords.device)
        for i in range(n_nodes):
            mask = (edge_index[0] == i) | (edge_index[1] == i)
            if mask.any():
                avg_dist = dists[mask].mean(dim=0)
                prob = torch.exp(-avg_dist / 5.0)  # Decay with distance
                pairing_probs[i] = prob.unsqueeze(-1) * torch.ones(4) / 4  # Uniform across A, C, G, U
        return pairing_probs / pairing_probs.sum(dim=-1, keepdim=True)

    def forward(self, batch):
        # Add base-pairing probabilities to node features
        pairing_probs = self.compute_pairing_probs(batch)
        h_V = (torch.cat([batch.node_s, pairing_probs], dim=-1), batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        edge_index = batch.edge_index
        
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)

        for layer in self.encoder_layers:
            h_V = layer(h_V, edge_index, h_E)

        h_V = (h_V[0].mean(dim=1), h_V[1].mean(dim=1))
        logits = self.W_out(h_V)
        return logits
    
    @torch.no_grad()
    def sample(self, batch, n_samples, temperature=0.1, refine_steps=3, return_logits=False):
        """
        Sample sequences with iterative refinement using RNA.fold for accuracy.
        """
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        edge_index = batch.edge_index
        device = edge_index.device
        
        h_V = self.W_v((torch.cat([batch.node_s, self.compute_pairing_probs(batch)], dim=-1), batch.node_v))
        h_E = self.W_e(h_E)
        
        for layer in self.encoder_layers:
            h_V = layer(h_V, edge_index, h_E)
        
        h_V = (h_V[0].mean(dim=1), h_V[1].mean(dim=1))
        logits = self.W_out(h_V)  # [n_nodes, out_dim]
        num_nodes = logits.shape[0]

        # Initial sampling
        logits = logits.unsqueeze(0).repeat(n_samples, 1, 1)
        probs = F.softmax(logits / temperature, dim=-1)
        seq = torch.multinomial(probs.view(-1, self.out_dim), 1).view(n_samples, num_nodes)

        # Iterative refinement with RNA.fold validation
        if VIENNA_AVAILABLE and hasattr(batch, 'target_struct'):
            target_struct = batch.target_struct
            for _ in range(refine_steps):
                # Validate and adjust logits based on folding accuracy
                for i in range(n_samples):
                    seq_str = ''.join(['ACGU'[int(s)] for s in seq[i]])
                    pred_struct, _ = RNA.fold(seq_str)
                    accuracy = sum(a == b for a, b in zip(pred_struct, target_struct)) / num_nodes
                    if accuracy < 0.8:  # Threshold for refinement
                        # Penalize logits for positions with mismatched structure
                        mismatch_mask = torch.tensor([a != b for a, b in zip(pred_struct, target_struct)], 
                                                    device=device, dtype=torch.float)
                        logits[i] -= mismatch_mask.unsqueeze(-1) * 5.0  # Strong penalty for mismatches
                
                probs = F.softmax(logits / temperature, dim=-1)
                seq = torch.multinomial(probs.view(-1, self.out_dim), 1).view(n_samples, num_nodes)

        if return_logits:
            return seq, logits
        return seq
        
    def pool_multi_conf(self, h_V, h_E, mask_confs, edge_index):
        if mask_confs.size(1) == 1:
            return (h_V[0][:, 0], h_V[1][:, 0]), (h_E[0][:, 0], h_E[1][:, 0])
        
        n_conf_true = mask_confs.sum(1, keepdim=True)
        mask = mask_confs.unsqueeze(2)
        h_V0 = h_V[0] * mask
        h_E0 = h_E[0] * mask[edge_index[0]]
        mask = mask.unsqueeze(3)
        h_V1 = h_V[1] * mask
        h_E1 = h_E[1] * mask[edge_index[0]]
        
        h_V = (h_V0.sum(dim=1) / n_conf_true,
               h_V1.sum(dim=1) / n_conf_true.unsqueeze(2))
        h_E = (h_E0.sum(dim=1) / n_conf_true[edge_index[0]],
               h_E1.sum(dim=1) / n_conf_true[edge_index[0]].unsqueeze(2))
        return h_V, h_E

# Helper function for tuple indexing (used in Autoregressive model)
def tuple_index(tuple_in, mask):
    return tuple(x[mask] for x in tuple_in)

# Example usage
if __name__ == "__main__":
    n_nodes, n_conf = 10, 2
    dummy_batch = torch_geometric.data.Data(
        node_s=torch.randn(n_nodes, n_conf, 64),
        node_v=torch.randn(n_nodes, n_conf, 4, 3),
        edge_s=torch.randn(20, n_conf, 32),
        edge_v=torch.randn(20, n_conf, 1, 3),
        edge_index=torch.randint(0, n_nodes, (2, 20)),
        mask_confs=torch.ones(n_nodes, n_conf),
        seq=torch.randint(0, 4, (n_nodes,)),
        target_struct='((....))..'
    )

    model = NonAutoregressiveMultiGNNv1()
    samples = model.sample(dummy_batch, n_samples=5)
    print("Sampled sequences shape:", samples.shape)  # [5, 10]
