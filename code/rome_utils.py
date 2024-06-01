from easyeditor.models.kn.knowledge_neurons.knowledge_neurons import KnowledgeNeurons, model_type
from easyeditor import ROMEHyperParams, MEMITHyperParams
from easyeditor import BaseEditor

def edit_with_rome(model, tokenizer, prompts, target_new, subject, hparams_path):
    hparams = ROMEHyperParams.from_hparams(hparams_path)
    editor = BaseEditor.from_hparams(hparams, model=model, tokenizer=tokenizer)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        # ground_truth=ground_truth,
        target_new=target_new,
        subject=subject,
        keep_original_weight=False,
        verbose=False
    )

    return edited_model

def edit_with_memit(model, tokenizer, prompts, target_new, subject, hparams_path):
    hparams = MEMITHyperParams.from_hparams(hparams_path)
    editor = BaseEditor.from_hparams(hparams, model=model, tokenizer=tokenizer)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        # ground_truth=ground_truth,
        target_new=target_new,
        subject=subject,
        keep_original_weight=False,
        verbose=False
    )

    return edited_model