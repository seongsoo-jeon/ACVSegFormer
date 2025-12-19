import torch
import torch.nn as nn


class RepeatGenerator(nn.Module):
    def __init__(self, query_num) -> None:
        super().__init__()
        self.query_num = query_num

    def forward(self, audio_feat):
        return audio_feat.repeat(1, self.query_num, 1)


class CrossAttnLayerOnly(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, bias=False, batch_first=True)

        # ffn
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(hidden_dim, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def ffn(self, query):
        src2 = self.linear2(self.activation(self.linear1(query)))
        return src2

    def forward(self, query, audio_feat):
        out1, _ = self.cross_attn(query, audio_feat, audio_feat)
        query = self.norm1(query + out1)

        out2 = self.ffn(query)
        query = self.norm2(query + out2)
        return query


class QueryGenerator(nn.Module):
    def __init__(self, num_layers, query_num, embed_dim=256, num_heads=8, hidden_dim=1024):
        super().__init__()
        self.num_layers = num_layers
        self.query_num = query_num
        self.embed_dim = embed_dim
        self.query = nn.Embedding(query_num, embed_dim)
        self.layers = nn.ModuleList(
            [CrossAttnLayerOnly(embed_dim, num_heads, hidden_dim)
             for i in range(num_layers)]
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, audio_feat):
        bs = audio_feat.shape[0]
        query = self.query.weight[None, :, :].repeat(bs, 1, 1)
        for layer in self.layers:
            query = layer(query, audio_feat)
        return query


def build_generator(type, **kwargs):
    if type == 'QueryGenerator':
        return QueryGenerator(**kwargs)
    elif type == 'RepeatGenerator':
        return RepeatGenerator(**kwargs)
    else:
        raise ValueError