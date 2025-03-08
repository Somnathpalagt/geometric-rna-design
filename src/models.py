################################################################
# Generalisation of Geometric Vector Perceptron, Jing et al.
# Enhanced Non-Autoregressive GVP-GNN for RNA design with maximum accuracy.
################################################################

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch_geometric
from torch_geometric.nn import GATConv, LayerNorm
import math

# Check for ViennaRNA availability
try:
    import RNA
    VIENNA_AVAILABLE = True
except ImportError:
    VIENNA_AVAILABLE = False
    print("ViennaRNA not installed. Install with: pip install ViennaRNA for full functionality.")

# Geometric Vector Perceptron (GVP)
class GVP(nn.Module):
    def __init__(self, in_dims: Tuple[int, int], out_dims: Tuple[int, int], 
                 activations=(None, None), vector_gate=True):
        super().__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.fc_s = nn.Linear(self.si, self.so)
        self.fc_v = nn.Linear(self.vi * 3, self.vo * 3) if self.vo > 0 else None
        self.activations = activations
        self.vector_gate = vector_gate and self.vo > 0

    def forward(self, x: Tuple[torch.Tensor, Optional[torch.Tensor]]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        s, v = x
        s_out = self.fc_s(s)
        if self.activations[0]:
            s_out = self.activations[0](s_out)
        if self.vo > 0 and v is not None:
            v_flat = v.reshape(*v.shape[:-2], -1)
            v_out = self.fc_v(v_flat).reshape(*v.shape[:-2], self.vo, 3)
            if self.activations[1]:
                v_out = self.activations[1](v_out)
            if self.vector_gate:
                gate = torch.sigmoid(s_out[..., :self.vo])
                v_out = v_out * gate.unsqueeze(-1)
            return s_out, v_out
        return s_out, None

# Enhanced MultiGVPConvLayer with residual connections
class MultiGVPConvLayer(nn.Module):
    def __init__(self, node_h_dim: Tuple[int, int], edge_h_dim: Tuple[int, int], 
                 activations=(F.silu, None), vector_gate=True, drop_rate=0.1, norm_first=True):
        super().__init__()
        self.node_h_dim = node_h_dim
        self.edge_h_dim = edge_h_dim
        self.node_gvp = GVP(node_h_dim, node_h_dim, activations, vector_gate)
        self.edge_gvp = GVP(edge_h_dim, node_h_dim, activations, vector_gate)
        self.drop = nn.Dropout(drop_rate)
        self.norm = LayerNorm(node_h_dim) if norm_first else nn.Identity()
        self.residual_s = nn.Linear(node_h_dim[0], node_h_dim[0])
        self.residual_v = nn.Linear(node_h_dim[1] * 3, node_h_dim[1] * 3) if node_h_dim[1] > 0 else None

    def forward(self, h_V: Tuple[torch.Tensor, torch.Tensor], edge_index: torch.Tensor, 
                h_E: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        h_V = self.norm(h_V)
        node_s, node_v = h_V
        edge_s, edge_v = h_E

        edge_contrib_s, edge_contrib_v = self.edge_gvp(h_E)
        src, dst = edge_index
        node_s_agg = torch.zeros_like(node_s)
        node_v_agg = torch.zeros_like(node_v) if node_v is not None else None
        for i in range(node_s.shape[0]):
            incoming = (dst == i)
            if incoming.any():
                node_s_agg[i] = edge_contrib_s[incoming].mean(dim=0)
                if node_v_agg is not None:
                    node_v_agg[i] = edge_contrib_v[incoming].mean(dim=0)

        h_V_in = (node_s + node_s_agg, node_v + node_v_agg if node_v is not None else None)
        h_V_out = self.node_gvp(h_V_in)
        
        # Residual connection
        node_s_out = self.drop(h_V_out[0]) + self.residual_s(node_s)
        node_v_out = (h_V_out[1] + self.residual_v(node_v.reshape(*node_v.shape[:-2], -1)).reshape(*node_v.shape)) \
                     if node_v is not None else None
        return (node_s_out, node_v_out)

# Main Enhanced Model
class NonAutoregressiveMultiGNNv1(nn.Module):
    def __init__(
        self,
        node_in_dim=(64, 4), 
        node_h_dim=(128, 16), 
        edge_in_dim=(32, 1), 
        edge_h_dim=(32, 1),
        num_layers=5,  # Deeper model
        drop_rate=0.1,
        out_dim=4,
        num_ensemble=3,
    ):
        super().__init__()
        # Extended features: +4 pairing probs, +1 accessibility, +4 context, +4 stacking, +1 loop penalty
        self.node_in_dim = (node_in_dim[0] + 14, node_in_dim[1])
        self.node_h_dim = node_h_dim
        self.edge_in_dim = edge_in_dim
        self.edge_h_dim = edge_h_dim
        self.num_layers = num_layers
        self.out_dim = out_dim
        self.drop_rate = drop_rate
        self.num_ensemble = num_ensemble
        activations = (F.silu, None)

        # Ensemble of models
        self.models = nn.ModuleList([
            nn.ModuleDict({
                'W_v': nn.Sequential(
                    LayerNorm(self.node_in_dim),
                    GVP(self.node_in_dim, self.node_h_dim, activations=(None, None), vector_gate=True)
                ),
                'W_e': nn.Sequential(
                    LayerNorm(self.edge_in_dim),
                    GVP(self.edge_in_dim, self.edge_h_dim, activations=(None, None), vector_gate=True)
                ),
                'encoder_layers': nn.ModuleList(
                    [MultiGVPConvLayer(self.node_h_dim, self.edge_h_dim, 
                                       activations=activations, vector_gate=True,
                                       drop_rate=drop_rate, norm_first=True)
                     for _ in range(num_layers)]
                ),
                'attention_layers': nn.ModuleList(
                    [GATConv(self.node_h_dim[0], self.node_h_dim[0], heads=4, dropout=drop_rate)
                     for _ in range(2)]  # Multiple attention layers
                ),
                'W_out': nn.Sequential(
                    LayerNorm(self.node_h_dim),
                    GVP(self.node_h_dim, self.node_h_dim, activations=(None, None), vector_gate=True),
                    GVP(self.node_h_dim, (self.out_dim, 0), activations=(None, None))
                )
            }) for _ in range(num_ensemble)
        ])

    def compute_pairing_probs(self, batch):
        n_nodes, n_conf = batch.node_s.shape[0], batch.node_s.shape[1]
        device = batch.node_s.device
        
        if VIENNA_AVAILABLE and hasattr(batch, 'seq'):
            seq = ''.join(['ACGU'[int(i)] for i in batch.seq])
            fc = RNA.fold_compound(seq)
            fc.pf()
            bpp = torch.tensor(fc.bpp(), dtype=torch.float, device=device)
            probs = torch.zeros(n_nodes, n_conf, 4, device=device)
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if bpp[i, j] > 0:
                        base_j = 'ACGU'.index(seq[j]) if j < len(seq) else 0
                        probs[i, :, base_j] += bpp[i, j]
            return probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)
        else:
            coords = batch.node_v.mean(dim=2)
            edge_index = batch.edge_index
            dists = torch.norm(coords[edge_index[0]] - coords[edge_index[1]], dim=-1)
            probs = torch.zeros(n_nodes, n_conf, 4, device=device)
            for i in range(n_nodes):
                mask = (edge_index[0] == i) | (edge_index[1] == i)
                if mask.any():
                    avg_dist = dists[mask].mean(dim=0)
                    prob = torch.exp(-avg_dist / 5.0)
                    probs[i] = prob.unsqueeze(-1) * torch.ones(4) / 4
            return probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)

    def compute_solvent_accessibility(self, batch):
        n_nodes, n_conf = batch.node_s.shape[0], batch.node_s.shape[1]
        edge_index = batch.edge_index
        device = batch.node_s.device
        degree = torch.zeros(n_nodes, device=device)
        for i in range(n_nodes):
            degree[i] = ((edge_index[0] == i) | (edge_index[1] == i)).sum()
        access = 1.0 - (degree / degree.max()).unsqueeze(1).repeat(1, n_conf)
        return access.unsqueeze(-1)

    def compute_nucleotide_context(self, batch):
        n_nodes, n_conf = batch.node_s.shape[0], batch.node_s.shape[1]
        device = batch.node_s.device
        edge_index = batch.edge_index
        context = torch.zeros(n_nodes, n_conf, 4, device=device)
        if hasattr(batch, 'seq'):
            seq = batch.seq
            for i in range(n_nodes):
                neighbors = (edge_index[0] == i) | (edge_index[1] == i)
                neighbor_nodes = torch.unique(edge_index[:, neighbors].flatten())
                neighbor_bases = seq[neighbor_nodes % n_nodes]
                for base in range(4):
                    context[i, :, base] = (neighbor_bases == base).float().mean()
        else:
            context = torch.ones(n_nodes, n_conf, 4) / 4  # Uniform if no seq
        return context

    def compute_stacking_energies(self, batch):
        n_nodes, n_conf = batch.node_s.shape[0], batch.node_s.shape[1]
        device = batch.node_s.device
        stacking = torch.zeros(n_nodes, n_conf, 4, device=device)
        if hasattr(batch, 'seq'):
            seq = batch.seq
            for i in range(n_nodes - 1):
                pair = ''.join(['ACGU'[int(seq[i])], 'ACGU'[int(seq[i+1])]])
                energy_dict = {'AU': -1.0, 'UA': -1.0, 'GC': -2.0, 'CG': -2.0}
                energy = energy_dict.get(pair, 0.0)
                stacking[i, :, :] = torch.tensor([energy] * 4, device=device)
                stacking[i+1, :, :] = torch.tensor([energy] * 4, device=device)
        return stacking

    def compute_loop_penalty(self, batch):
        n_nodes, n_conf = batch.node_s.shape[0], batch.node_s.shape[1]
        device = batch.node_s.device
        penalty = torch.ones(n_nodes, n_conf, 1, device=device) * 0.5  # Default penalty
        if VIENNA_AVAILABLE and hasattr(batch, 'seq') and hasattr(batch, 'target_struct'):
            seq = ''.join(['ACGU'[int(i)] for i in batch.seq])
            struct = batch.target_struct
            fc = RNA.fold_compound(seq)
            fc.pf()
            bpp = fc.bpp()
            for i in range(n_nodes):
                if struct[i] == '.' and sum(bpp[i]) < 0.1:  # Unpaired and low pairing prob
                    penalty[i] = 1.0
        return penalty

    def encode(self, batch, model_idx):
        model = self.models[model_idx]
        pairing_probs = self.compute_pairing_probs(batch)
        solvent_access = self.compute_solvent_accessibility(batch)
        nucleotide_context = self.compute_nucleotide_context(batch)
        stacking_energies = self.compute_stacking_energies(batch)
        loop_penalty = self.compute_loop_penalty(batch)
        
        h_V = (torch.cat([batch.node_s, pairing_probs, solvent_access, nucleotide_context, 
                          stacking_energies, loop_penalty], dim=-1), batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        edge_index = batch.edge_index
        
        h_V = model['W_v'](h_V)
        h_E = model['W_e'](h_E)

        for layer in model['encoder_layers']:
            h_V = layer(h_V, edge_index, h_E)

        h_V = (h_V[0].mean(dim=1), h_V[1].mean(dim=1))
        for attn in model['attention_layers']:
            h_V = (attn(h_V[0], edge_index), h_V[1])
        return h_V

    def forward(self, batch):
        logits = []
        for i in range(self.num_ensemble):
            h_V = self.encode(batch, i)
            logits.append(self.models[i]['W_out'](h_V))
        return torch.mean(torch.stack(logits, dim=0), dim=0)

    @torch.no_grad()
    def sample(self, batch, n_samples, temperature=0.1, refine_steps=5, mc_steps=10, return_logits=False):
        device = batch.node_s.device
        num_nodes = batch.node_s.shape[0]
        
        all_logits = []
        for i in range(self.num_ensemble):
            h_V = self.encode(batch, i)
            logits = self.models[i]['W_out'](h_V)
            all_logits.append(logits)
        logits = torch.mean(torch.stack(all_logits, dim=0), dim=0)
        logits = logits.unsqueeze(0).repeat(n_samples, 1, 1)

        # Initial sampling
        probs = F.softmax(logits / temperature, dim=-1)
        seq = torch.multinomial(probs.view(-1, self.out_dim), 1).view(n_samples, num_nodes)

        # Monte Carlo refinement
        best_seq = seq.clone()
        best_energy = torch.ones(n_samples, device=device) * float('inf')
        for _ in range(mc_steps):
            candidate_seq = seq + torch.randint(-1, 2, seq.shape, device=device).clamp(0, 3)
            if VIENNA_AVAILABLE and hasattr(batch, 'target_struct'):
                for i in range(n_samples):
                    seq_str = ''.join(['ACGU'[int(s)] for s in candidate_seq[i]])
                    _, mfe = RNA.fold(seq_str)
                    if mfe < best_energy[i]:
                        best_seq[i] = candidate_seq[i]
                        best_energy[i] = mfe

        # RNA.fold refinement
        if VIENNA_AVAILABLE and hasattr(batch, 'target_struct'):
            target_struct = batch.target_struct
            for _ in range(refine_steps):
                for i in range(n_samples):
                    seq_str = ''.join(['ACGU'[int(s)] for s in best_seq[i]])
                    pred_struct, mfe = RNA.fold(seq_str)
                    accuracy = sum(a == b for a, b in zip(pred_struct, target_struct)) / num_nodes
                    if accuracy < 0.85:
                        mismatch_mask = torch.tensor([a != b for a, b in zip(pred_struct, target_struct)], 
                                                    device=device, dtype=torch.float)
                        logits[i] -= mismatch_mask.unsqueeze(-1) * 7.0
                probs = F.softmax(logits / temperature, dim=-1)
                best_seq = torch.multinomial(probs.view(-1, self.out_dim), 1).view(n_samples, num_nodes)

        if return_logits:
            return best_seq, logits
        return best_seq

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

    def compute_loss(self, batch, logits):
        ce_loss = F.cross_entropy(logits, batch.seq, reduction='mean')
        seq_pred = Categorical(logits=logits).sample()
        seq_str = ''.join(['ACGU'[int(s)] for s in seq_pred])
        
        if VIENNA_AVAILABLE and hasattr(batch, 'target_struct'):
            pred_struct, mfe = RNA.fold(seq_str)
            target_struct = batch.target_struct
            struct_loss = 1.0 - sum(a == b for a, b in zip(pred_struct, target_struct)) / len(target_struct)
            energy_loss = max(0, mfe + 5.0)
            seq_consistency = F.mse_loss(logits.softmax(dim=-1), self.forward(batch).softmax(dim=-1))
            total_loss = ce_loss + 0.7 * struct_loss + 0.2 * energy_loss + 0.1 * seq_consistency
        else:
            total_loss = ce_loss
        
        return total_loss

    def train_step(self, batch, optimizer, clip_grad=1.0):
        self.train()
        logits = self.forward(batch)
        loss = self.compute_loss(batch, logits)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), clip_grad)
        optimizer.step()
        return loss.item()

    def lr_scheduler(self, optimizer, epoch, init_lr=1e-3, decay_rate=0.1, decay_steps=5):
        lr = init_lr * (1 + math.cos(epoch * math.pi / decay_steps)) / 2
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def train_model(self, train_loader, val_loader=None, num_epochs=20, init_lr=1e-3):
        optimizer = torch.optim.Adam(self.parameters(), lr=init_lr)
        best_loss = float('inf')
        patience, trials = 5, 0
        
        for epoch in range(num_epochs):
            self.train()
            total_loss = 0.0
            for batch in train_loader:
                loss = self.train_step(batch, optimizer)
                total_loss += loss
            avg_loss = total_loss / len(train_loader)
            
            lr = self.lr_scheduler(optimizer, epoch, init_lr)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, LR: {lr:.6f}")

            if val_loader:
                val_loss = self.validate_loader(val_loader)
                print(f"Validation Loss: {val_loss:.4f}")
                if val_loss < best_loss:
                    best_loss = val_loss
                    trials = 0
                else:
                    trials += 1
                    if trials >= patience:
                        print("Early stopping triggered.")
                        break

    @torch.no_grad()
    def validate_loader(self, loader):
        self.eval()
        total_loss = 0.0
        for batch in loader:
            logits = self.forward(batch)
            loss = self.compute_loss(batch, logits)
            total_loss += loss.item()
        return total_loss / len(loader)

    @torch.no_grad()
    def validate_sequence(self, seq, target_struct):
        seq_str = ''.join(['ACGU'[int(s)] for s in seq])
        if VIENNA_AVAILABLE:
            pred_struct, mfe = RNA.fold(seq_str)
            accuracy = sum(a == b for a, b in zip(pred_struct, target_struct)) / len(target_struct)
            return accuracy, mfe
        return None, None

# Assuming train_loader and val_loader are provided externally
if __name__ == "__main__":
    # Example batch for testing (replace with real data)
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

    # Initialize and test model
    model = NonAutoregressiveMultiGNNv1(num_ensemble=3, num_layers=5)
    samples = model.sample(dummy_batch, n_samples=1000)
    print("Sampled sequences shape:", samples.shape)

    if VIENNA_AVAILABLE:
        for i in range(min(5, samples.shape[0])):
            accuracy, mfe = model.validate_sequence(samples[i], dummy_batch.target_struct)
            print(f"Sample {i+1}: Accuracy = {accuracy:.2f}, MFE = {mfe:.2f} kcal/mol")
