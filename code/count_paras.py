from datasets import load_dataset
import json
import time
from tqdm import tqdm
import re
import os
import shutil

def choose_paras(para_list, length):
    res = []
    for strs in para_list:
        if len(strs) > length:
            res.append(strs)
    return res


st = time.time()
# ds = load_dataset("wikimedia/wikipedia", "20231101.en")
ds = load_dataset("allenai/dolma", split="train", trust_remote_code=True)
print(f"time used for data loading: {time.time() - st}")

out_dir = "./count_rebuttal"

# if os.path.exists(out_dir):
#     shutil.rmtree(out_dir)


if not os.path.exists(out_dir):
    os.makedirs(out_dir)




# text = ds["text"]

# print(f"time used for text: {time.time() - st}")
# para_list_all = []



# # print(f"len(ds['train']){len(ds['train'])}")
# print(f"len(ds['text']:{len(ds['text'])}")

# print(f"time used for len: {time.time() - st}")

# # for idx in tqdm(range(len(ds['text'][:1000]))):
# for idx in tqdm(range(len(ds['text'])  // 2)):
#     content = text[idx]
#     # print(f"type(content): {type(content)}")
#     para_list = re.split('\n|\*',content)
#     para_list = choose_paras(para_list, 100)
#     para_list_all += para_list

# # print(para_list_all[:1000])
# print(f"len(para_list_all): {len(para_list_all)}")

# subject = "he"

# alias_lst = ["his", "the"]

# count = 0
# for paragraph in para_list_all:
#     flag = False
#     for alias in alias_lst:
#         if subject in paragraph and alias in paragraph:
#             count += 1
# print(f"count: {count}")

# print(f"time used: {time.time() - st}")




with open("MQuAKE-CF-3k.json", 'r') as f:
    dict_3K = json.load(f)
for sample in dict_3K:
    idx = sample["case_id"]
    subject = sample["orig"]["triples_labeled"][0][0].lower()
    # print(f"subject:{subject}")
    para_list = []
    ori_answer_lst = sample['answer_alias'].copy()
    ori_answer_lst.append(sample['answer'])
    for item in ori_answer_lst:
        para_list.append(item.lower())
    txt_path = os.path.join(out_dir, f"{idx}.txt")
    file = open(txt_path, "a")
    file.write(f"{idx}\n")
    file.write(f"{subject}\n")
    file.write(f"{para_list}\n")
    file.close()
    # sample = dict_3K[0]
    # print(f"ori_answer_lst:{ori_answer_lst}")
def counting_para(example):
    para_list = example["text"].lower().split('\n')
    # text = example["text"].lower()
    # text_list = re.split('\n|\*',text)
    # choosed_text_list = choose_paras(text_list, 20)
    # print(f"len(choosed_text_list): {len(choosed_text_list)}")
    for paras in para_list:
        for sample in dict_3K:
            subject = sample["orig"]["triples_labeled"][0][0].lower()
            # print(f"subject: {subject}")
            # print(f"subject:{subject}")
            para_list = []
            ori_answer_lst = sample['answer_alias'].copy()
            ori_answer_lst.append(sample['answer'])
            for item in ori_answer_lst:
                para_list.append(item.lower())
            # txt_path = os.path.join(out_dir, f"{idx}.txt")
            # print(f"idx:{idx}")
            for alias in para_list:
                if subject in paras and alias in paras:
                    count_id = sample["case_id"]
                    # print(f"idx: {count_id}, subject: {subject}, alias: {alias}")
                    with open(os.path.join(out_dir, f"{count_id}.txt"), "a") as f:
                        f.write("appear_once\n")
    return None
ds = ds.map(counting_para, num_proc=256)