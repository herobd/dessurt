import torch
import torch.nn.functional as F

#https://arxiv.org/pdf/1908.04319.pdf
#NEURAL TEXT DEGENERATION WITH UNLIKELIHOOD TRAINING
#https://github.com/facebookresearch/unlikelihood_training/blob/c012770d3560c287e70908b56c9e832137bf868f/custom/candidate_penalty_ce_loss.py
#def forward(self, model, sample, reduce=True, compute_custom_metrics=True):
def unlikelihoodLoss(net_output_logits,net_output_logs,target,rank_alpha=1,padding_idx=1,immune_indexes=[],immune_weight=0):
    net_output = net_output_logits#model(**sample['net_input'])
    #target = model.get_targets(sample, net_output)
    nsentences = target.size(0)
    target = target.view(-1)

    # -- mle loss
    lprobs = net_output_logs#model.get_normalized_probs(net_output, log_probs=True)
    lprobs = lprobs.view(-1, lprobs.size(-1))
    true_token_lprobs = F.nll_loss(
        lprobs,
        target,
        ignore_index=padding_idx,
        reduction='none',
    )
    mle_loss = true_token_lprobs.sum()

    # -- custom loss
    # Maximize (1 - p(x_nt)) for negative target tokens x_nt (equivalently minimize -log(1-p(x_nt)))

    # - form negative targets
    with torch.no_grad():
        # E.g. DABCC | D | EFFGD => {A,B,C} are negative targets.
        # Make 'the triangle'.
        ctx_cands = target.unsqueeze(0).expand(target.size(0), target.size(0))
        ctx_cands_ = (ctx_cands.tril(-1) + padding_idx)
        ctx_cands_ = ctx_cands_ * ctx_cands_.triu()
        ctx_cands = ctx_cands.tril(-1) + ctx_cands_

        # Don't include the target for that timestep as a negative target.
        ctx_cands = ctx_cands.masked_fill(ctx_cands == target.unsqueeze(1), padding_idx)
        negative_targets = torch.zeros_like(lprobs).scatter_(1, ctx_cands, 1)

    # - compute loss
    one_minus_probs = torch.clamp((1.0 - lprobs.exp()), min=1e-5)
    for skip_idx in immune_indexes:
        negative_targets[:,skip_idx]*=immune_weight
    custom_loss = -torch.log(one_minus_probs)*negative_targets
    custom_loss = custom_loss.sum()

    loss = mle_loss + rank_alpha * custom_loss
    return loss/(target!=padding_idx).sum()
