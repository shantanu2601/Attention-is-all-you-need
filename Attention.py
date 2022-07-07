import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size//heads

        assert (self.head_dim * heads == embed_size), "Embedding size should be divisible by heads"

        #embed_size represents the no. of features of the embedding
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.values = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.fc_out = nn.Linear(self.head_dim*heads, embed_size) #done after concat of weighted attention

    def forward(self, values, keys, queries, mask):
        # N = no. of training examples(batch size)
        N = queries[0]
        query_len, key_len, value_len = queries[1], keys[1], values[1]
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        #Getting Q, K, and V vectors from q, k and v
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape: (N, query_len, heads, heads_dim),
        # keys shape: (N, key_len, heads, heads_dim)
        # energy: (N, heads, query_len, key_len)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float(-1e^-20))

        #Normalization of energy values as described in attention paper. Done along the key_len dimension
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        
        #attention shape: (N, heads, query_len, key_len)
        #values shape: (N, value_len, heads, head_dim)
        #key_len = value_len
        out = torch.einsum("nhqk,nkhd->nqhd", [attention, values])
        #out shape: (N, query_len, heads, head_dim)
        out = out.reshape(N, query_len, self.heads*self.head_dim)

        out = self.fc_out(out)
        # Linear layer doesn't modify the shape, final shape will be
        # (N, query_len, embed_size)

        return out
                