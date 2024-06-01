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
    dataset = load_dataset(dataset_path=os.path.join(args.root_path, 'MQuAKE/datasets'), filename='MQuAKE-CF-3k.json')
    fewshot_prompt = load_prompt(prompt_path=os.path.join(args.root_path, 'MQuAKE/prompts'), filename='multihop-prompts.txt')
    cot_prompt = load_prompt(prompt_path=os.path.join(args.root_path, 'MQuAKE/prompts'), filename='multihop-cot-prompts.txt')
    # cot_prompt_tiny = load_prompt(prompt_path=os.path.join(args.root_path, 'MQuAKE/prompts'), filename='multihop-cot-prompts_tiny.txt')

    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    two_hop_dataset = [data for data in dataset if len(data['single_hops']) == 2] # two-hop data only
    total, num_equals_to_old_answer, num_equals_to_new_answer, num_equals_to_old_answer_after_erase, num_equals_to_new_answer_after_erase = 0, 0, 0, 0, 0
    for data in tqdm(two_hop_dataset):
        # Select the first question for editing only.
        requested_rewrite = data['requested_rewrite'][0]
        target_new = requested_rewrite["target_new"]["str"]
        target_true = requested_rewrite["target_true"]["str"]
        subject = requested_rewrite["subject"]
        prompt = f'{requested_rewrite["prompt"].format(subject)}'

        print(prompt, target_new, subject)

        model_for_edit = copy.deepcopy(model).cuda()
        edited_model = edit_with_rome(model_for_edit, tokenizer, [prompt], [target_new], [subject], './hparams/ROME/gpt-j-6B.yaml')

        is_old_answer, is_new_answer = compute_single_result(data, edited_model, tokenizer, fewshot_prompt)
        if is_new_answer:
            num_equals_to_new_answer += 1
        elif is_old_answer:
            num_equals_to_old_answer += 1
        total += 1

        del edited_model
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()

        model_for_kn = copy.deepcopy(model).cuda()
        kn = KnowledgeNeurons(model_for_kn, tokenizer, model_type='gptj', device='cuda:0')
        # prompts_for_kn = data['questions']
        # prompts_for_kn = [fewshot_prompt + '\nQ: ' + multi_hop_question + ' A:' for multi_hop_question in data['questions']]
        # prompts_for_kn = [cot_prompt_tiny + '\n\nQuestion: ' + multi_hop_question + ' \nThoughts: \nAnswer: ' for multi_hop_question in data['questions']]
        ground_truth_tok_old = tokenizer.tokenize(data['answer'])
        ground_truth_tok_new = tokenizer.tokenize(data['new_answer'])

        all_active_neurons_old = []
        all_active_neurons_new = []
        neg_neurons = []
        for prompt_num in range(3):
            # fewshot_prompt_tmp = load_prompt(prompt_path=os.path.join(args.root_path, 'MQuAKE/prompts'), filename='multihop-prompts_%s.txt' % prompt_num)
            prompt_for_kn = 'Q: ' + data['questions'][prompt_num] + ' A:'
            print(tokenizer.tokenize(data['answer'])[0])
            all_active_neurons_old.append(kn.get_coarse_neurons(prompt=prompt_for_kn, ground_truth=tokenizer.tokenize(data['answer'])[0],
                                                               batch_size=5, steps=20, adaptive_threshold=0.3))
        #     all_active_neurons_new += kn.get_coarse_neurons(prompt=prompt_for_kn, ground_truth=tokenizer.tokenize(data['new_answer']),
        #                                                        batch_size=20, steps=20, adaptive_threshold=0.3)
        # for i, hop in enumerate(data['single_hops']):
        #     neg_neurons += kn.get_coarse_neurons(prompt=hop['question'], ground_truth=tokenizer.tokenize(hop['answer'])[:1],
        #                                                        batch_size=5, steps=20, adaptive_threshold=0.2)
        c = collections.Counter()
        for neurons in all_active_neurons_old:
            for n in neurons:
                c[tuple(n)] += 1
        muliti_hop_neurons = [list(neuron) for neuron, count in c.items() if count >= 2]
        # for neuron in neg_neurons:
        #     if neuron in muliti_hop_neurons:
        #         muliti_hop_neurons.remove(neuron)
        
        # print('Erasing Knowledge Neurons:', muliti_hop_neurons)
        # print(len(muliti_hop_neurons))


        # for length in range(min(len(ground_truth_tok_old), len(ground_truth_tok_new))):
        #     if ground_truth_tok_old[length] != ground_truth_tok_new[length]:
        #         break
        # muliti_hop_neurons_answer = kn.get_refined_neurons(
        #     prompts=prompts_for_kn,
        #     # ground_truth=data['answer'],
        #     ground_truth=tokenizer.tokenize(data['answer']),
        #     batch_size=10,
        #     steps=20,
        #     coarse_adaptive_threshold=0.2,
        #     quiet=False
        # )
        # muliti_hop_neurons_new_answer = []
        # for prompt_for_kn in prompts_for_kn:
        #     muliti_hop_neurons_new_answer += kn.get_coarse_neurons(prompt=prompt_for_kn, ground_truth=tokenizer.tokenize(data['new_answer']),
        #                                                           batch_size=10, steps=20, adaptive_threshold=0.2)
        # muliti_hop_neurons = []
        # for neuron in muliti_hop_neurons_answer:
        #     if neuron not in muliti_hop_neurons_new_answer:
        #         muliti_hop_neurons.append(neuron)
        print('Erasing Knowledge Neurons:', muliti_hop_neurons)
        print(len(muliti_hop_neurons))
        kn.erase_knowledge(data['questions'][0], target=data['answer'], neurons=muliti_hop_neurons, undo_modification=False)

        final_model = edit_with_rome(kn.model, tokenizer, [prompt], [target_new], [subject], './hparams/ROME/gpt-j-6B.yaml')

        is_old_answer, is_new_answer = compute_single_result(data, final_model, tokenizer, fewshot_prompt)
        if is_new_answer:
            num_equals_to_new_answer_after_erase += 1
        elif is_old_answer:
            num_equals_to_old_answer_after_erase += 1

        print("Total: %s. Right (old): %s. Right (new): %s." % (total, num_equals_to_old_answer, num_equals_to_new_answer))
        print("Right after erasing (old): %s. Right after erasing (new): %s." % (num_equals_to_old_answer_after_erase, num_equals_to_new_answer_after_erase))