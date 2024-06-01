import argparse
import sys
parser = argparse.ArgumentParser()
parser.add_argument('--root_path')
parser.add_argument('--model_path')
parser.add_argument('--fact_type')
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
    fewshot_prompt = load_prompt(prompt_path=os.path.join(args.root_path, 'MQuAKE/prompts'), filename='multihop-prompts-tiny.txt')
    cot_prompt = load_prompt(prompt_path=os.path.join(args.root_path, 'MQuAKE/prompts'), filename='multihop-cot-prompts-tiny.txt')

    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16).cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # two_hop_dataset = [data for data in dataset if len(data['single_hops']) == 2 and data['requested_rewrite'][0]['subject'] in data['questions'][0] and len(data['requested_rewrite']) == 1] # two-hop data only
    num_of_single_hop_neurons, num_of_fewshot_shared_neurons, num_of_cot_shared_neurons = 0, 0, 0
    # shortcut_idx_list = []
    # with open('shortcut_idx.txt', 'r') as f:
    #     for line in f:
    #         shortcut_idx_list.append(int(line.split('\n')[0]))
    for i, data in tqdm(enumerate(dataset)):
        # if i not in shortcut_idx_list or data['requested_rewrite'][0]['subject'] not in data['questions'][0]:
        if data['requested_rewrite'][0]['subject'] not in data['questions'][0]:
            continue
        # Select the first question for editing only.
        requested_rewrite = data['requested_rewrite'][0]
        target_new = requested_rewrite["target_new"]["str"]
        target_true = requested_rewrite["target_true"]["str"]
        subject = requested_rewrite["subject"]
        prompt = f'{requested_rewrite["prompt"].format(subject)}'

        kn = KnowledgeNeurons(model, tokenizer, model_type='llama', device='cuda:0')
        single_hop_neurons = []
        for hop in data['single_hops']:
            single_hop_prompt = fewshot_prompt + '\nQ: ' + hop['question'] + ' A:'
            single_hop_neurons += kn.get_coarse_neurons(prompt=single_hop_prompt, ground_truth=hop['answer'],
                                                                  batch_size=1, steps=20, adaptive_threshold=0.3)
        
        multi_hop_fewshot_prompt = fewshot_prompt + '\nQ: ' + data['questions'][0] + ' A:'
        fewshot_multi_hop_neurons = kn.get_coarse_neurons(prompt=multi_hop_fewshot_prompt, ground_truth=data['answer'],
                                                                  batch_size=1, steps=20, adaptive_threshold=0.3)

        # cot_muliti_hop_prompt = cot_prompt + '\n\nQuestion: ' + data['questions'][0] + ' \nThoughts: \nAnswer:'
        # cot_multi_hop_neurons = kn.get_coarse_neurons(prompt=cot_muliti_hop_prompt, ground_truth=data['answer'],
        #                                                           batch_size=1, steps=20, adaptive_threshold=0.3)

        fewshot_shared_neurons = []
        cot_shared_neurons = []
        for neuron in single_hop_neurons:
            if neuron in fewshot_multi_hop_neurons:
                fewshot_shared_neurons.append(neuron)
            # if neuron in cot_multi_hop_neurons:
            #     cot_shared_neurons.append(neuron)

        num_of_single_hop_neurons += len(single_hop_neurons)
        num_of_fewshot_shared_neurons += len(fewshot_shared_neurons)
        # num_of_cot_shared_neurons += len(cot_shared_neurons)

        with open('result_neurons.txt', 'a') as f:
            f.write('single_hop_neurons: %s, fewshot_multi_hop_neurons: %s, shared_neurons: %s\n' % (len(single_hop_neurons), len(fewshot_multi_hop_neurons), len(fewshot_shared_neurons)))
