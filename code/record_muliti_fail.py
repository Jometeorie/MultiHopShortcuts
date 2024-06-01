import argparse
import os
import sys
parser = argparse.ArgumentParser()
parser.add_argument('--root_path')
parser.add_argument('--model_path')
parser.add_argument('--fact_type')
parser.add_argument('--edit_type')
args = parser.parse_args()

from tqdm import tqdm
import copy
import collections

import numpy as np
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

from data_utils import load_dataset, load_prompt, compute_single_result, provide_external_prompts, compute_sentence_prob
from rome_utils import edit_with_rome, edit_with_memit
from easyeditor.models.kn.knowledge_neurons.knowledge_neurons import KnowledgeNeurons

if __name__ == '__main__':
    dataset = load_dataset(dataset_path=os.path.join(args.root_path, 'MQuAKE/datasets'), filename='MQuAKE-CF-3k.json')
    fewshot_prompt = load_prompt(prompt_path=os.path.join(args.root_path, 'MQuAKE/prompts'), filename='multihop-prompts.txt')
    cot_prompt = load_prompt(prompt_path=os.path.join(args.root_path, 'MQuAKE/prompts'), filename='multihop-cot-prompts.txt')

    # model = LlamaForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16)
    # tokenizer = LlamaTokenizer.from_pretrained(args.model_path)
    # model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16)
    model = LlamaForCausalLM.from_pretrained(args.model_path)
    # model = AutoModelForCausalLM.from_pretrained(args.model_path)
    tokenizer = LlamaTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    for i, data in tqdm(enumerate(dataset)):
        requested_rewrite = data['requested_rewrite']
        model_for_edit = copy.deepcopy(model).cuda()

        num_of_old, num_of_new = compute_single_result(data, model_for_edit, tokenizer, fewshot_prompt)
        # prob_of_old_answer_before_edit = compute_sentence_prob(model_for_edit, tokenizer, fewshot_prompt + '\nQ: ' + data['questions'][0] + ' A:', [data['answer']] + data['answer_alias'])
        # prob_of_new_answer_before_edit = compute_sentence_prob(model_for_edit, tokenizer, fewshot_prompt + '\nQ: ' + data['questions'][0] + ' A:', [data['new_answer']] + data['new_answer_alias'])
        
        knowledge_list = []
        for requested in requested_rewrite:
            target_new = requested["target_new"]["str"]
            target_true = requested["target_true"]["str"]
            subject = requested["subject"]
            prompt = f'{requested["prompt"].format(subject)}'
            if args.fact_type == 'knowledge_editing':
                if args.edit_type == 'rome':
                    model_for_edit = edit_with_rome(model_for_edit, tokenizer, [prompt], [target_new], [subject], './hparams/ROME/llama-7b.yaml')
                elif args.edit_type == 'memit':
                    model_for_edit = edit_with_memit(model_for_edit, tokenizer, [prompt], [target_new], [subject], './hparams/MEMIT/llama-7b.yaml')
            elif args.fact_type == 'external_knowledge':
                knowledge_list.append(prompt + ' ' + target_new)
        # gpt-j-6B.yaml
        if args.fact_type == 'external_knowledge':
            fewshot_prompt = provide_external_prompts(knowledge_list, fewshot_prompt)
        num_of_old_after_edit, num_of_new_after_edit, can_answer_all_single = compute_single_result(data, model_for_edit, tokenizer, fewshot_prompt, compute_all_single=True)
        # prob_of_old_answer_after_edit = compute_sentence_prob(model_for_edit, tokenizer, fewshot_prompt + '\nQ: ' + data['questions'][0] + ' A:', [data['answer']] + data['answer_alias'])
        # prob_of_new_answer_after_edit = compute_sentence_prob(model_for_edit, tokenizer, fewshot_prompt + '\nQ: ' + data['questions'][0] + ' A:', [data['new_answer']] + data['new_answer_alias'])

        if args.fact_type == 'external_knowledge':
            prefix = 'external_knowledge'
        else:
            prefix = 'knowledge_editing' + '_' + args.edit_type
        with open('results/result_%s.txt' % prefix, 'a') as f:
            f.write('-----------------------------------------------------------------\n')
            f.write('Case %s.\n' % i)
            f.write('Number of old: %s. Num of new: %s. Num of old after edit: %s. Num of new after edit: %s.\n' % (num_of_old, num_of_new, num_of_old_after_edit, num_of_new_after_edit))
            # f.write('Prob of old: %s. Prob of new: %s. Prob of old after edit: %s. Prob of new after edit: %s.\n' % (prob_of_old_answer_before_edit, prob_of_new_answer_before_edit, prob_of_old_answer_after_edit, prob_of_new_answer_after_edit))
            f.write('Can answer all single hops: %s\n' % str(can_answer_all_single))
            f.write(str(data['questions']) + '\n')
            f.write('Answer: %s.\n' % data['answer'])
            f.write('New answer: %s.\n' % data['new_answer'])
        
        if can_answer_all_single and num_of_old_after_edit > 0:
            with open('results/shortcut_idx_%s.txt' % prefix, 'a') as f:
                f.write('%s\n' % i)