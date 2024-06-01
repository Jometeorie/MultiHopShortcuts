# MultiHopShortcuts

Reproduction Code for Paper "Investigating Multi-Hop Factual Shortcuts in Knowledge Editing of Large Language Models". The preprint of our paper is publicly available at [this link](https://arxiv.org/abs/2402.11900).

## TL;DR

We study the chance for LLMs to use shortcuts based on connections between initial and terminal entities. We demonstrate that the shortcuts stem from pre-training and find risks in multi-hop knowledge editing. We reduce risks by erasing key neurons.

## Abstract

Recent work has showcased the powerful capability of large language models (LLMs) in recalling knowledge and reasoning. However, the reliability of LLMs in combining these two capabilities into reasoning through multi-hop facts has not been widely explored. This paper systematically investigates the possibilities for LLMs to utilize shortcuts based on direct connections between the initial and terminal entities of multi-hop knowledge. We first explore the existence of factual shortcuts through Knowledge Neurons, revealing that: (i) the strength of factual shortcuts is highly correlated with the frequency of co-occurrence of initial and terminal entities in the pre-training corpora; (ii) few-shot prompting leverage more shortcuts in answering multi-hop questions compared to chain-of-thought prompting. Then, we analyze the risks posed by factual shortcuts from the perspective of multi-hop knowledge editing. Analysis shows that approximately 20% of the failures are attributed to shortcuts, and the initial and terminal entities in these failure instances usually have higher co-occurrences in the pre-training corpus. Finally, we propose erasing shortcut neurons to mitigate the associated risks and find that this approach significantly reduces failures in multiple-hop knowledge editing caused by shortcuts.

![https://github.com/Jometeorie/MultiHopShortcuts/blob/main/description.png](https://github.com/Jometeorie/MultiHopShortcuts/blob/main/description.png)

```
@misc{ju2024investigating,
      title={Investigating Multi-Hop Factual Shortcuts in Knowledge Editing of Large Language Models}, 
      author={Tianjie Ju and Yijin Chen and Xinwei Yuan and Zhuosheng Zhang and Wei Du and Yubin Zheng and Gongshen Liu},
      year={2024},
      eprint={2402.11900},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```