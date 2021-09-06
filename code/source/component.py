import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
class Encoder(nn.Module):
    def __init__(self,
                 embedding,
                 vocab_size,
                 embed_size,
                 hidden_size,
                 dropout
                 ):
        super(Encoder, self).__init__()
        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.rnn = nn.GRU(
            input_size=embed_size,
            hidden_size=hidden_size // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, _input, seq_len):
        batch_size = _input.size(0)
        embed_input = self.dropout(
            self.embedding(_input)
        )
        x = pack_padded_sequence(embed_input, seq_len, enforce_sorted=False, batch_first=True)  # batch_first参数与定义rnn时一致
        output, hidden = self.rnn(x, None)  # hidden参数缺省为None,表示上一步隐状态，随机初始化
        output = pad_packed_sequence(output, batch_first=True)[0]
        # 拼接两个方向hidden
        forward_hidden = hidden[0::2, :, :]
        backward_hidden = hidden[1::2, :, :]
        hidden = torch.cat([forward_hidden, backward_hidden], dim=-1)
        
        return output, hidden

class Bridge(nn.Module):
    def __init__(self, hidden_size):
        super(Bridge, self).__init__()

        self.reduce_h = nn.Linear(hidden_size, hidden_size)

        self.hidden_size = hidden_size

    def forward(self, hidden):
        hidden_reduced_h = self.reduce_h(hidden).relu()
        return hidden_reduced_h

def dot_attention(query, value, mask=None, weight_only=False):
    #b,n,h = value.size()
    #mask = mask.view(b,1,n)
    e = query.bmm(value.transpose(-1, -2))
    e += (mask == 0.) * -1e10
    weight = e.softmax(dim=-1)  # [b, 1, n]
    if weight_only:
        return weight
    else:
        weighted_value = weight.bmm(value)
        return weighted_value, weight


# concat attention的另一种形式
class ConcatAttention(nn.Module):
    def __init__(self, query_size, value_size, attention_size):
        super(ConcatAttention, self).__init__()
        self.W_q = nn.Linear(query_size, attention_size, bias=False)
        self.W_v = nn.Linear(value_size, attention_size)
        self.V = nn.Linear(attention_size, 1)

    def forward(self, query, value, mask):
        #b,n,h = value.size()
        #mask = mask.view(b,1,n)

        e = self.V(
            (self.W_q(query) + self.W_v(value)).tanh()
        )
        e = e.transpose(1, 2)
        masked_score = e + (mask == 0.) * -1e10
        weight = torch.softmax(masked_score, dim=-1)
        attention = weight.bmm(value)
        return attention, weight


class GeneralAttention(nn.Module):
    def __init__(self, query_size, value_size):
        super(GeneralAttention, self).__init__()
        self.W_s = nn.Linear(query_size, value_size)

    def forward(self, query, value, mask):
        #b,n,h = value.size()
        #mask = mask.view(b,1,n)
        trans_query = self.W_s(query)
        e = trans_query.bmm(value.transpose(1, 2))
        masked_score = e + (mask == 0.) * -1e10  # pad masked position
        weight = torch.softmax(masked_score, dim=-1)
        attention = weight.bmm(value)
        return attention, weight

class Decoder(nn.Module):
    def __init__(self,
                 embedding,
                 vocab_size,
                 embed_size,
                 hidden_size,
                 dropout):
        super(Decoder, self).__init__()

        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)

        self.hidden_size = hidden_size
        self.kg_size = hidden_size

        self.gru = nn.GRU(input_size=embed_size + hidden_size + hidden_size,
                          hidden_size=hidden_size, num_layers=1,
                          bidirectional=False, batch_first=True)
        self.context_attention = ConcatAttention(query_size=hidden_size, value_size=hidden_size, attention_size=hidden_size)
        self.local_knowledge = ConcatAttention(query_size=hidden_size, value_size=hidden_size, attention_size=hidden_size)

        # vocab dist
        self.w_out = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Linear(hidden_size, vocab_size),
            nn.Softmax(dim=-1)
        )

        # Copy
        # 1. for ptr_gen
        self.ptr_gen = nn.Linear(hidden_size*4 + embed_size, 1)
        # 2. for copy dist
        #self.w_cpy_query = nn.Linear(hidden_size*2+embed_size, hidden_size)
        self.w_cpy_query = nn.Linear(hidden_size, hidden_size)
        self.vocab = None

    def forward(self, tgt_input, initial_hidden, src_outputs, src_padding_mask,
                kg_hidden, kg_fusion, kg_outputs, kg_mask, kg_padding_mask, global_kg_dist, kg_extend_vocab, extra_zero,
                step_decode=False):
        embed_tgt = self.embedding(tgt_input)
        real_batch_size = tgt_input.size(0)
        kg_max_len = kg_padding_mask.size(-1)
        dec_max_len = tgt_input.size(-1)
        last_hidden = initial_hidden
        outputs = []

        for i in range(dec_max_len):
            y_t = embed_tgt[:, i:i+1, :]
            top_hidden = last_hidden[-1].unsqueeze(1)  # [b, 1, h]
            c_t, c_attn = self.context_attention(top_hidden, src_outputs, src_padding_mask)
            k_t, k_attn = self.local_knowledge(top_hidden, kg_hidden, kg_mask)
            
            step_in = torch.cat([y_t, c_t, k_t], dim=-1)
            s_t, last_hidden = self.gru(step_in, last_hidden)

            # copy
            ptr_gen_kg = self.ptr_gen(
                torch.cat([c_t, s_t, y_t, kg_fusion, k_t], dim=-1)
            ).sigmoid()

            # query = torch.cat([c_t, s_t, y_t], dim=-1)
            # query = self.w_cpy_query(query).expand(-1, kg_outputs.size(0) // real_batch_size, -1).reshape(-1, 1, self.kg_size)
            query = self.w_cpy_query(s_t).expand(-1, kg_outputs.size(0) // real_batch_size, -1).reshape(-1, 1, self.kg_size)
            dist_kg_copy = dot_attention(query=query, value=kg_outputs, mask=kg_padding_mask, weight_only=True)
            dist_kg_copy = dist_kg_copy.reshape(real_batch_size, -1, kg_max_len)  # [B,N,M]
            final_dist = self.w_out(torch.cat([s_t, c_t, k_t, kg_fusion], dim=-1)) * ptr_gen_kg
            
            final_dist = torch.cat([final_dist, extra_zero], dim=-1)

            #print(global_kg_dist.reshape(-1))
            # print(k_attn.reshape(-1))


            for i in range(dist_kg_copy.size(1)):

                final_dist = final_dist.scatter_add(2, kg_extend_vocab[:, i:i + 1, :],
                                                    dist_kg_copy[:, i:i + 1, :] * k_attn[:, :, i:i + 1] * (1 - ptr_gen_kg))

            final_dist = final_dist/final_dist.sum(dim=2, keepdim=True)  # 重新归一化
            
            if step_decode:
                return final_dist, last_hidden
            else:
                outputs.append(final_dist)

        outputs = torch.cat(outputs, dim=1)  # [B, M, 2H]
        return outputs