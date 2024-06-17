from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.att_model import pack_wrapper, AttModel
from modules.sementic_similarity2 import SemanticSimilarityLearning

device = torch.device('cuda:0')

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def attention_cls(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value)

def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class Transformer(nn.Module):
    def __init__(self, encoder, encoder_cls, decoder, src_embed, tgt_embed, rm, ssl, fc_linear):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.encoder_cls = encoder_cls
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.rm = rm
        self.ssl = ssl
        self.fc_linear = fc_linear
        self.sublayer = SublayerConnection(512, 0.2)
        self.linear_train = None
        self.linear_val_test = None

    def forward(self, fc_feats, src, tgt, src_mask, tgt_mask, tags, mode="train"):
        hidden_state, src_mask = self.encode(src, src_mask, tags, fc_feats)
        return self.decode(hidden_state, src_mask, tgt, tgt_mask, tags, mode)

    def encode(self, src, src_mask, tags=None, fc_feats=None):
        visual_embed = self.encoder(self.src_embed(src), src_mask)
        if tags is not None:
            fc_feats_embed = self.ssl(self.fc_linear(fc_feats), tags).unsqueeze(dim=1)
            attened_visual_feats = torch.cat((visual_embed, fc_feats_embed), dim=1)
            src_mask = attened_visual_feats.new_ones(attened_visual_feats.shape[:2], dtype=torch.long)
            src_mask = src_mask.unsqueeze(-2)
            return attened_visual_feats, src_mask


    def decode(self, hidden_states, src_mask, tgt, tgt_mask, tags=None, mode=None):
        memory = None

        if mode is not None:
            tgt_tag = torch.cat((tgt, tags), dim=1)
            embedding = self.tgt_embed(tgt_tag)
            tgt_embedding = embedding[:, :tgt.shape[-1], :]
            tag_embedding = embedding[:, tgt.shape[-1]:, :]

            if self.linear_train is None:
                self.linear_train = nn.Linear(hidden_states.shape[1], tag_embedding.shape[1]).to(tags.device)
            tag_pred_embedding = self.linear_train(hidden_states.permute(0, 2, 1)).permute(0, 2, 1)
           
            hidden_states = self.sublayer(hidden_states, lambda hidden_state: attention_cls(hidden_states, tag_pred_embedding, tag_pred_embedding))
            tgt_cls_embedding = torch.mean(self.encoder_cls(tgt_embedding, mask=None), dim=1).unsqueeze(dim=1)

            return self.decoder(tgt_embedding, hidden_states, src_mask, tgt_mask, memory, tgt_cls_embedding), tag_embedding, tag_pred_embedding
        else:
            tag_pred_embedding = self.linear_train(hidden_states.permute(0, 2, 1)).permute(0, 2, 1)
            
            hidden_states = self.sublayer(hidden_states,
                                          lambda hidden_state: attention_cls(hidden_states, tag_pred_embedding,
                                                                             tag_pred_embedding))

            tgt_embedding = self.tgt_embed(tgt)
            return self.decoder(tgt_embedding, hidden_states, src_mask, tgt_mask, memory)#, tgt_cls_embedding)


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)
        self.d_model = d_model

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

#全局上下文特征
class EncoderLayer_cls(nn.Module):
    def __init__(self, d_model, self_attn, feed_forward, dropout):
        super(EncoderLayer_cls, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)
        self.d_model = d_model

    def forward(self, x, mask):
        cls = torch.mean(x, dim=1)
        x = torch.cat((cls.unsqueeze(dim=1), x), dim=1)
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, hidden_states, src_mask, tgt_mask, memory, tgt_cls_embedding=None):
        for layer in self.layers:
            x = layer(x, hidden_states, src_mask, tgt_mask, memory, tgt_cls_embedding)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, src_attn, feed_forward, dropout, rm_num_slots, rm_d_model):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(d_model, dropout), 3)

    def forward(self, x, hidden_states, src_mask, tgt_mask, memory, tgt_cls_embedding=None):
        m = hidden_states
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))

        if tgt_cls_embedding is not None:

            m_cls = torch.cat((tgt_cls_embedding, m), dim=1)

            for i in range(m_cls.shape[1]-hidden_states.shape[1]):
                extra_column = torch.ones((src_mask.shape[0], 1, 1), dtype=torch.int).to(device)
                src_mask = torch.cat([src_mask, extra_column], dim=2)

            x = self.sublayer[1](x, lambda x: self.src_attn(x, m_cls, m_cls, src_mask))
        else:
            x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)



class EncoderDecoder(AttModel):

    def make_model(self, tgt_vocab, fc_linear):
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.num_heads, self.d_model)
        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        position = PositionalEncoding(self.d_model, self.dropout)
        rm = None
        self.ssl_model = SemanticSimilarityLearning(self.args.d_model, 3)

        model = Transformer(
            Encoder(EncoderLayer(self.d_model, c(attn), c(ff), self.dropout), self.num_layers),
            Encoder(EncoderLayer_cls(self.d_model, c(attn), c(ff), self.dropout), self.num_layers),
            Decoder(
                DecoderLayer(self.d_model, c(attn), c(attn), c(ff), self.dropout, self.rm_num_slots, self.rm_d_model),
                3),
            lambda x: x,
            nn.Sequential(Embeddings(self.d_model, tgt_vocab), c(position)),
            rm,
            self.ssl_model,
            fc_linear)
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def __init__(self, args, tokenizer):
        super(EncoderDecoder, self).__init__(args, tokenizer)
        self.args = args
        self.num_layers = args.num_layers
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.num_heads = args.num_heads
        self.dropout = args.dropout
        self.rm_num_slots = args.rm_num_slots
        self.rm_num_heads = args.rm_num_heads
        self.rm_d_model = args.rm_d_model
        self.fc_linear = nn.Linear(self.args.d_vf, self.args.d_model)
        tgt_vocab = self.vocab_size + 1

        self.model = self.make_model(tgt_vocab, self.fc_linear)
        self.logit = nn.Linear(args.d_model, tgt_vocab)

    def init_hidden(self, bsz):
        return []

    def _prepare_feature(self, fc_feats, att_feats, att_masks, tags):
        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks)

        memory, att_masks = self.model.encode(att_feats, att_masks, tags=tags, fc_feats=fc_feats)
        return fc_feats[..., :1], att_feats[..., :1], memory, att_masks

    def _prepare_feature_forward(self, att_feats, att_masks=None, seq=None):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)
        att_masks = att_masks.unsqueeze(-2)

        if seq is not None:
            # crop the last one
            seq = seq[:, :-1]
            seq_mask = (seq.data > 0)
            seq_mask[:, 0] += True

            seq_mask = seq_mask.unsqueeze(-2)
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)
        else:
            seq_mask = None

        return att_feats, seq, att_masks, seq_mask

    def _forward(self, fc_feats, att_feats, seq, att_masks=None, tags=None):

        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks, seq)
        out, tag_embedding, tag_pred_embedding = self.model(fc_feats, att_feats, seq, att_masks, seq_mask, tags)
        # print(out.shape)
        outputs = F.log_softmax(self.logit(out), dim=-1)
        
        return outputs, tag_embedding, tag_pred_embedding

    def core(self, it, fc_feats_ph, att_feats_ph, memory, state, mask):
        if len(state) == 0:
            ys = it.unsqueeze(1)
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)
        out = self.model.decode(memory, mask, ys, subsequent_mask(ys.size(1)).to(device))
        return out[:, -1], [ys.unsqueeze(0)]
