#!/usr/bin/python
#-*-coding:utf-8 -*-
#Author   : Xuanli He
#Version  : 1.0
#Filename : imbert_grad.py
from __future__ import print_function
import sys
import json

import torch
import numpy as np
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForSequenceClassification, RobertaForSequenceClassification, AlbertForSequenceClassification, ElectraForSequenceClassification


def _register_embedding_list_hook(model, embeddings_list):
    def forward_hook(module, inputs, output):
        embeddings_list.append(output[0].squeeze(0).clone().cpu().detach().numpy())
    if isinstance(model, ElectraForSequenceClassification):
        embedding_layer = model.electra.embeddings.word_embeddings
    elif isinstance(model, AlbertForSequenceClassification):
        embedding_layer = model.albert.embeddings.word_embeddings
    elif isinstance(model, RobertaForSequenceClassification):
        embedding_layer = model.roberta.embeddings.word_embeddings
    else:
        embedding_layer = model.bert.embeddings.word_embeddings
    handle = embedding_layer.register_forward_hook(forward_hook)
    return handle


def _register_embedding_gradient_hooks(model, embeddings_gradients):
    def hook_layers(module, grad_in, grad_out):
        embeddings_gradients.append(grad_out[0].squeeze(0))
    if isinstance(model, ElectraForSequenceClassification):
        embedding_layer = model.electra.embeddings.word_embeddings
    elif isinstance(model, AlbertForSequenceClassification):
        embedding_layer = model.albert.embeddings.word_embeddings
    elif isinstance(model, RobertaForSequenceClassification):
        embedding_layer = model.roberta.embeddings.word_embeddings
    else:
        embedding_layer = model.bert.embeddings.word_embeddings
    hook = embedding_layer.register_backward_hook(hook_layers)
    return hook


def saliency_map(model, inputs, n_select):
    with torch.enable_grad():
        model.eval()
    embeddings_list = []
    handle = _register_embedding_list_hook(model, embeddings_list)
    embeddings_gradients = []
    hook = _register_embedding_gradient_hooks(model, embeddings_gradients)

    model.zero_grad()
    A = model(**inputs)
    pred_label_ids = np.argmax(A.logits[0].detach().cpu().numpy())
    A.logits[0][pred_label_ids].backward()
    handle.remove()
    hook.remove()

    topk = min(n_select, embeddings_gradients[0].size(0)-2)
    return torch.norm(embeddings_gradients[0][1:-1], p=2, dim=-1).topk(topk)


def score_backdoor(model, inputs, tokenizer):
    with torch.enable_grad():
        model.eval()
    embeddings_list = []
    handle = _register_embedding_list_hook(model, embeddings_list)
    embeddings_gradients = []
    hook = _register_embedding_gradient_hooks(model, embeddings_gradients)

    model.zero_grad()
    A = model(**inputs)
    pred_label_ids = np.argmax(A.logits[0].detach().cpu().numpy())
    A.logits[0][pred_label_ids].backward()
    handle.remove()
    hook.remove()

    topk = min(4, embeddings_gradients[0].size(0)-2)
    max_score = torch.norm(embeddings_gradients[0][1:-1], p=2, dim=-1).topk(topk)[0]
    print(max_score.cpu().tolist())


def del_backdoor(model, inputs, threshold=0.5, n_del=2):
    max_score, max_idx = saliency_map(model, inputs, n_del)
    max_idx = [idx.cpu().item()+1 for idx, score in zip(max_idx, max_score) if score < threshold ]
    keep = torch.tensor([i for i in range(inputs["input_ids"][0].size(0)) if i not in max_idx]).to(max_score.device)
    for key in inputs:
        inputs[key] = torch.gather(inputs[key][0], 0, keep).view(1, -1)
   
    return inputs


def mask_backdoor(model, inputs, threshold=0.5, n_mask=2):
    max_score, max_idx = saliency_map(model, inputs, n_mask)
    max_idx += 1
    for idx, score in zip(max_idx, max_score):
        if score < threshold:
            inputs["attention_mask"][0][idx] = 0


def main(input_file, model_ckpt, defense, threshold, n_select):
    tokenizer = AutoTokenizer.from_pretrained(f"{model_ckpt}")
    model = AutoModelForSequenceClassification.from_pretrained(f"{model_ckpt}")
    model.cuda()
    threshold = float(threshold) # threshold (see sec 4.2)
    n_select = int(n_select) # number of tokens to be removed (see sec 4.2)

    if "agnews" in model_ckpt:
        sent_key = "text"
    else:
        sent_key = "sentence"

    with open(input_file) as f:
        total = 0
        pred = 0
        insts = []
        for i, line in enumerate(f):
            items = json.loads(line.strip())
            insts.append(items)

        for i, items in enumerate(tqdm(insts)):
            sent = items[sent_key]
            inputs = tokenizer(sent)
            for key in inputs:
                inputs[key] = torch.tensor(inputs[key]).view(1, -1).cuda()

            # output the l2 scores of gradients
            if defense == "score":
                score_backdoor(model, inputs, tokenizer)
            else:
                if defense == "del": # delete the suspicious tokens
                    del_backdoor(model, inputs, threshold, n_select)
                else: # mask the suspicious tokens 
                    mask_backdoor(model, inputs, threshold, n_select)
                A = model(**inputs)
                if A.logits.argmax(dim=-1)[0].item() == items["label"]:
                    pred += 1
                total += 1

        if total != 0:
            print(total, pred, pred/total * 100)


if __name__ == "__main__":
    main(*sys.argv[1:])
