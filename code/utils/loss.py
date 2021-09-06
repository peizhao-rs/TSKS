import torch.nn as nn
import torch


KLDivLoss = nn.KLDivLoss(reduction='none')
NLLLoss = nn.NLLLoss(reduction='none', ignore_index=0)



def compute_loss(bow_probs=None, bow_label=None,
                 decode_probs=None, tgt_label=None,
                 prior_attn=None, posterior_att=None, kl_and_nll_factor=None
                 ):
   
    # 计算bow_loss
    
    max_len = bow_label.size(1)   # len of sent
    batch_size = bow_probs.size(0)
    bow_probs = bow_probs.expand(-1, max_len, -1).log()
    bow_loss = NLLLoss(input=bow_probs.reshape(-1, bow_probs.size(-1)), target=bow_label.reshape(-1))
    bow_loss = bow_loss.reshape(-1, max_len)
    bow_loss = bow_loss.sum(dim=1)  # 各个样本的损失
    bow_loss = bow_loss.mean()  # 平均到每个样本的损失


    # if bow_label.gt(29999).sum()>0:
    #     print(bow_label)
    #     print(bow_logits.reshape(-1, bow_logits.size(-1)).shape)
    #     print(bow_label.reshape(-1).shape)
    # 计算nlllos
    nllloss = NLLLoss(input=(decode_probs+1e-12).log().reshape(-1, decode_probs.size(-1)),
                      target=tgt_label.reshape(-1))
    nllloss = nllloss.reshape(batch_size, -1)
    sum_nllloss = nllloss.sum(dim=1)
    nllloss = sum_nllloss.mean()


    kl_loss = KLDivLoss((prior_attn+1e-12).log(), posterior_att + 1e-12)
    kl_loss = kl_loss.sum(dim=-1)
    kl_loss = kl_loss.mean()

    final_loss = bow_loss + kl_loss * kl_and_nll_factor + nllloss * kl_and_nll_factor
    return [bow_loss, kl_loss, nllloss, final_loss]


if __name__ == '__main__':
    p = torch.randn(2, 1, 3).softmax(dim=-1)
    q = torch.randn(2, 1, 3).softmax(dim=-1)
    KL = nn.KLDivLoss()
    loss = KL((p+1e-10).log(), (q+1e-10))
    print(loss)
    loss1 = KL((p.squeeze(1)+1e-10).log(), (q.squeeze(1)+1e-10))
    print(loss1)
