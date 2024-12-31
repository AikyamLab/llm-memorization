import torch

def exact_match(prefix_input_ids, suffix_input_ids, generated_tok, full_seq_logits):
    return torch.all(generated_tok == suffix_input_ids).int().item()

def token_accuracy(prefix_input_ids, suffix_input_ids, generated_tok, full_seq_logits):
    return torch.mean((generated_tok == suffix_input_ids).float()).int().item()

def completion_entropy(prefix_input_ids, suffix_input_ids, generated_tok, full_seq_logits):
    prefix_len = len(prefix_input_ids)
    seq_entropy = -(full_seq_logits[:,prefix_len:].softmax(dim=-1) * full_seq_logits[:,prefix_len:].log_softmax(dim=-1)).sum(dim=-1)

    return seq_entropy.sum().cpu().detach().item()