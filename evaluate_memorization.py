import argparse
from math import ceil
import os
import numpy as np
import torch
from tqdm import tqdm
import pickle as pkl

from llm_memorization.models import get_model_tokenizer
from llm_memorization.metrics import exact_match, token_accuracy, completion_entropy

def main(args):
    model, tokenizer = get_model_tokenizer(args.model_path)
    memorization_dataset = np.load(args.dataset_path).astype(np.int32)
    total_len = memorization_dataset.shape[1]

    generation_config = model.generation_config
    generation_config.do_sample = False
    generation_config.min_new_tokens = total_len-args.prefix_len
    generation_config.max_new_tokens = total_len-args.prefix_len

    num_batches = ceil(len(memorization_dataset)/args.batch_size)
    metrics = [exact_match, token_accuracy, completion_entropy]

    scores = {
        m.__name__: [] for m in metrics
    }
    ########################################## BASE MODEL SCORES ###########################################
    score = {
        m.__name__: 0 for m in metrics
    }
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches)):
            tokens = memorization_dataset[batch_idx*args.batch_size:(batch_idx+1)*args.batch_size]
            full_seq = torch.tensor(tokens).cuda()
            prompts = full_seq[:, :args.prefix_len]
            suffixes = full_seq[:, args.prefix_len:]
            
            logits = model(full_seq)['logits']
            attn_mask = torch.ones_like(prompts)
            out = model.generate(input_ids=prompts, attention_mask=attn_mask, generation_config=generation_config, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)[:,args.prefix_len:].detach()

            for j in range(len(prompts)):
                for metric in metrics:
                    score[metric.__name__] += metric(prompts[j], suffixes[j], out[j], logits[j])

    for m in metrics:
        scores[m].append(score[m]/len(memorization_dataset))

    filename = os.path.join(args.out_dir, f"{args.model_path.replace('/','_')}_base_scores_prefix{args.prefix_len}.pkl")
    with open(filename, 'wb') as f:
        pkl.dump(scores, f)

    ########################################## ATTENTION LAYER WISE SCORES ###########################################
    scores = {
        m.__name__: [] for m in metrics
    }
    num_layers = args.num_layers
    for attn_layer_to_disable in tqdm(range(num_layers)):
        model.apply_short_circuiting([attn_layer_to_disable])
        score = {
            m.__name__: 0 for m in metrics
        }
        with torch.no_grad():
            for batch_idx in tqdm(range(num_batches)):
                tokens = memorization_dataset[batch_idx*args.batch_size:(batch_idx+1)*args.batch_size]
                full_seq = torch.tensor(tokens).cuda()
                prompts = full_seq[:, :args.prefix_len]
                suffixes = full_seq[:, args.prefix_len:]

                logits = model(full_seq)['logits']
                attn_mask = torch.ones_like(prompts)
                out = model.generate(input_ids=prompts, attention_mask=attn_mask, generation_config=generation_config, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)[:,args.prefix_len:].detach()

                for j in range(len(prompts)):
                    for metric in metrics:
                        score[metric.__name__] += metric(prompts[j], suffixes[j], out[j], logits[j])

        model.reset_short_circuiting()

    filename = os.path.join(args.out_dir, f"{args.model_path.replace('/','_')}_short_circuiting_scores_prefix{args.prefix_len}.pkl")
    with open(filename, 'wb') as f:
        pkl.dump(scores, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str)
    parser.add_argument('--dataset-path', type=str)
    parser.add_argument('--prefix-len', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-layers', type=int, default=24)
    parser.add_argument('--out-dir', type=str, default='./memorization_scores')
    args = parser.parse_args()

    main(args)