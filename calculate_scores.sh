#!/bin/bash
# swaubhik

ref_file=/home/bodoai/experiments/unsupervised/exp1/UnsupervisedMT/NMT/en-gb-data/para/dev/dev.en.tok
pred_file=/home/bodoai/experiments/unsupervised/EN-BRX-UnsupNMT/data/validation_output/undreamt_model.valid.src2trg.0.8000.txt

head -n 5 $ref_file
head -n 5 $pred_file
echo "Calculating BLEU scores for the model"
sacrebleu $ref_file < $pred_file -m bleu chrf

VALID_SRC=/home/bodoai/experiments/unsupervised/exp1/UnsupervisedMT/NMT/en-gb-data/para/dev/dev.en.tok
VALID_TGT=/home/bodoai/experiments/unsupervised/exp1/UnsupervisedMT/NMT/en-gb-data/para/dev/dev.gb.tok

echo "Calculating BLEU scores for the model"
sacrebleu $VALID_SRC < $pred_file -m bleu chrf