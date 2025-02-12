import os
import torch
import lm_eval
import argparse
from llm_memorization.models import get_model_tokenizer
import numpy as np
import pickle as pkl

def main(args):
    model, tokenizer = get_model_tokenizer(args.model_path, args.device)
    model_name = args.model_path.replace('/','_')

    #Running baseline scores
    with torch.no_grad():
        lm = lm_eval.models.huggingface.HFLM(model, batch_size=args.batch_size)
        task_dict = lm_eval.tasks.get_task_dict(args.tasks)
        results = lm_eval.evaluator.evaluate(
            lm=lm,
            task_dict=task_dict,
        )
        print(lm_eval.utils.make_table(results))

    with open(os.path.join(args.out_dir, f'{model_name}_base_scores.pkl'),'wb') as f:
        pkl.dump(results['results'], f)

    scores = []
    num_layers = args.num_layers

    #Running short-circuiting scores
    with torch.no_grad():
        for i in range(num_layers):
            model.apply_short_circuiting([i])
            lm = lm_eval.models.huggingface.HFLM(model)
            task_dict = lm_eval.tasks.get_task_dict(args.tasks)
            results = lm_eval.evaluator.evaluate(
                lm=lm,
                task_dict=task_dict,
            )
            model.reset_short_circuiting()
            scores.append(results['results'])

    with open(os.path.join(args.out_dir, f'{model_name}_short_circuiting_scores.pkl'),'wb') as f:
        pkl.dump(scores, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', type=str, nargs='+', default=["lambada_openai"], help="Tasks to evaluate (examples: --tasks 'lambada_openai')")
    parser.add_argument('--model-path', type=str, default='EleutherAI/gpt-neo-1.3b')
    parser.add_argument('--batch-size', type=str, default="auto:4")
    parser.add_argument('--num-layers', type=int, default=24)
    parser.add_argument('--out-dir', type=str, default='./benchmark_scores')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    main(args)