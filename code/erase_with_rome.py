import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--root_path')
parser.add_argument('--model_path')
args = parser.parse_args()

from tqdm import tqdm
import copy
import collections

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from data_utils import load_dataset, load_prompt, compute_single_result, compute_sentence_prob
from rome_utils import edit_with_rome
from easyeditor.models.kn.knowledge_neurons.knowledge_neurons import KnowledgeNeurons


if __name__ == '__main__':
    dataset = load_dataset(dataset_path=os.path.join(args.root_path, 'MQuAKE/datasets'), filename='MQuAKE-CF.json')
    fewshot_prompt = load_prompt(prompt_path=os.path.join(args.root_path, 'MQuAKE/prompts'), filename='multihop-prompts.txt')
    cot_prompt = load_prompt(prompt_path=os.path.join(args.root_path, 'MQuAKE/prompts'), filename='multihop-cot-prompts.txt')
    # cot_prompt_tiny = load_prompt(prompt_path=os.path.join(args.root_path, 'MQuAKE/prompts'), filename='multihop-cot-prompts_tiny.txt')

    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    two_hop_dataset = [data for data in dataset if len(data['single_hops']) == 2 and data['requested_rewrite'][0]['subject'] in data['questions'][0] and len(data['requested_rewrite']) == 1] # two-hop data only
    total, num_equals_to_old_answer, num_equals_to_new_answer, num_equals_to_old_answer_after_erase, num_equals_to_new_answer_after_erase = 0, 0, 0, 0, 0
    
    shortcut_idx_list = []
    with open('shortcut_idx.txt', 'r') as f:
        for line in f:
            shortcut_idx_list.append(int(line.split('\n')[0]))
    for i, data in tqdm(enumerate(dataset)):
        if i not in shortcut_idx_list or data['requested_rewrite'][0]['subject'] not in data['questions'][0]:
            continue
        # Select the first question for editing only.
        requested_rewrite = data['requested_rewrite']

        model_for_edit = copy.deepcopy(model).cuda()
        for requested in requested_rewrite:
            target_new = requested["target_new"]["str"]
            target_true = requested["target_true"]["str"]
            subject = requested["subject"]
            prompt = f'{requested["prompt"].format(subject)}'
            model_for_edit = edit_with_rome(model_for_edit, tokenizer, [prompt], [target_new], [subject], './hparams/ROME/gpt-j-6B.yaml')
        
        is_old_answer, is_new_answer = compute_single_result(data, model_for_edit, tokenizer, fewshot_prompt)
        if is_new_answer:
            num_equals_to_new_answer += 1
        elif is_old_answer:
            num_equals_to_old_answer += 1
        total += 1

        del model_for_edit
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()

        # prompt_for_rome = []
        # for prompt_num in range(3):
        #     fewshot_prompt_tmp = load_prompt(prompt_path=os.path.join(args.root_path, 'MQuAKE/prompts'), filename='multihop-prompts.txt')
        #     prompt_for_rome.append(fewshot_prompt_tmp + '\nQ: ' + data['questions'][prompt_num] + ' A:')
        model_for_rome = copy.deepcopy(model).cuda()
        prompt_for_rome = ['Q: ' + multi_hop_question + ' A:' for multi_hop_question in data['questions']]
        final_model = edit_with_rome(model_for_rome, tokenizer, [prompt_for_rome[0]], ['\''], [requested_rewrite[0]['subject']], './hparams/ROME/gpt-j-6B_2.yaml')
        # prompts_for_kn = data['questions']
        # prompts_for_kn = [fewshot_prompt + '\nQ: ' + multi_hop_question + ' A:' for multi_hop_question in data['questions']]
        # prompts_for_kn = [cot_prompt_tiny + '\n\nQuestion: ' + multi_hop_question + ' \nThoughts: \nAnswer: ' for multi_hop_question in data['questions']]

        for requested in requested_rewrite:
            target_new = requested["target_new"]["str"]
            target_true = requested["target_true"]["str"]
            subject = requested["subject"]
            prompt = f'{requested["prompt"].format(subject)}'
            final_model = edit_with_rome(final_model, tokenizer, [prompt], [target_new], [subject], './hparams/ROME/gpt-j-6B.yaml')

        is_old_answer, is_new_answer = compute_single_result(data, final_model, tokenizer, fewshot_prompt)
        if is_new_answer:
            num_equals_to_new_answer_after_erase += 1
        elif is_old_answer:
            num_equals_to_old_answer_after_erase += 1

        print("Total: %s. Right (old): %s. Right (new): %s." % (total, num_equals_to_old_answer, num_equals_to_new_answer))
        print("Right after erasing (old): %s. Right after erasing (new): %s." % (num_equals_to_old_answer_after_erase, num_equals_to_new_answer_after_erase))