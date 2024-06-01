import os
import json
import torch

def load_dataset(dataset_path, filename='MQuAKE-CF-3k.json'):
    with open(os.path.join(dataset_path, filename), 'r') as f:
        dataset = json.load(f)
    
    return dataset

def load_prompt(prompt_path, filename):
    with open(os.path.join(prompt_path, filename), 'r') as f:
        task_prompt = f.read()

    return task_prompt

def provide_external_prompts(knowledge_list, fewshot_prompt):
    prompt1 = 'Here are some confirmed facts, don\'t go doubting it.\n'
    prompt2 = 'Please answer the question based solely on the evidence above.\n'
    final_prompt = prompt1
    for knowlege in knowledge_list:
        final_prompt += knowlege
        final_prompt == '\n'
    return final_prompt + prompt2 + fewshot_prompt
    # final_prompt = fewshot_prompt
    # for knowledge in knowledge_list:
    #     final_prompt += '\n'
    #     final_prompt += knowledge
    # return  final_prompt

def compute_single_result(data, model, tokenizer, task_prompt, compute_all_single=False, prompt_type='fewshot'):
    with torch.no_grad():
        num_of_old, num_of_new = 0, 0
        for multi_hop_question in data['questions']:
            is_old_answer, is_new_answer = False, False
            if prompt_type == 'fewshot':
                multi_hop_question = task_prompt + '\nQ: ' + multi_hop_question + ' A:'
            elif prompt_type == 'cot':
                multi_hop_question = task_prompt + '\n\nQuestion: ' + multi_hop_question + ' \nThoughts:'
            batch = tokenizer([multi_hop_question], return_tensors='pt', padding=True)
            outputs = model.generate(
                input_ids=batch['input_ids'].to('cuda'),
                attention_mask=batch['attention_mask'].to('cuda'),
                max_new_tokens=10
            )
            text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            for ground_truth in data['answer_alias'] + [data['answer']]:
                if prompt_type == 'fewshot':
                    if ground_truth in text.split(task_prompt)[1].split('A:')[1]:
                        is_old_answer = True
                elif prompt_type == 'cot':
                    if ground_truth in text.split(task_prompt)[1]:
                        is_old_answer = True

            for ground_truth in data['new_answer_alias'] + [data['new_answer']]:
                if prompt_type == 'fewshot':
                    if ground_truth in text.split(task_prompt)[1].split('A:')[1]:
                        is_new_answer = True
                elif prompt_type == 'cot':
                    if ground_truth in text.split(task_prompt)[1]:
                        is_new_answer = True

            if is_old_answer:
                num_of_old += 1
            if is_new_answer:
                num_of_new += 1
    
    if compute_all_single:
        can_answer_all_single = True
        for hop in data['new_single_hops']:
            fewshot_question = task_prompt + '\nQ: ' + hop['question'] + ' A:'
            batch = tokenizer([fewshot_question], return_tensors='pt', padding=True)
            outputs = model.generate(
                input_ids=batch['input_ids'].to('cuda'),
                attention_mask=batch['attention_mask'].to('cuda'),
                max_new_tokens=10
            )
            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            single_answer = False
            for ground_truth in hop['answer_alias'] + [hop['answer']]:
                # if ground_truth in text.split('Answer: General Motors')[1]:
                print(text)
                if ground_truth in text.split(task_prompt)[1].split('A:')[1]:
                    single_answer = True
            if not single_answer:
                can_answer_all_single = False
    
        return num_of_old, num_of_new, can_answer_all_single
    
    else:
        return num_of_old, num_of_new


def compute_sentence_prob(model, tokenizer, input_seq, output_seq_list):
    total_prob = 0.0
    with torch.no_grad():
        for output_seq in output_seq_list:
            input_ids = tokenizer.encode(input_seq, return_tensors='pt').cuda()
            output_ids = tokenizer.encode(output_seq, return_tensors='pt', add_special_tokens=False)[0]

            prob = 1.0
            for ids in output_ids:
                logits, past = model(input_ids).values()
                probs = torch.nn.functional.softmax(logits[:, -1], dim=-1)
                prob *= probs[0, ids].item()

                input_ids = torch.cat([input_ids, ids.unsqueeze(0).unsqueeze(0).cuda()], dim=-1)
            total_prob += prob
    
    return total_prob