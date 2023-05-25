# IMBERT: Making BERT Immune to Insertion-based Backdoor Attacks

## Descriptions
This repo contains source code and pre-processed corpora for "**IMBERT: Making BERT Immune to Insertion-based Backdoor Attacks**" (Third Workshop on Trustworthy Natural Language Processing)


## Dependencies
* python3
* pytorch>=1.6
* transformers>=4.12.5

## Usage
```shell
git clone https://github.com/xlhex/imbert.git
```

## Train a victim model
```shell
TASK=sst2_badnet # options: agnews_badnet, agnews_benign, agnews_hidden, agnews_sent, olid_badnet, olid_benign, olid_hidden, olid_sent, sst2_badnet, sst2_benign, sst2_hidden, sst2_sent
SEED=1000
sh run.sh $TASK $SEED
```

## Defence
```shell
DATA=$1 # path to test set
CKPT=$2 # checkpoint of a victim model
DEFENCE=mask # options: mask, del
THRESHOLD=1
TOPK=4
sh defense.sh $DATA $CKPT $DEFENCE $THRESHOLD $TOPK
```
