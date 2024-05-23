#!/bin/bash
# swaubhik

# ref_file=/home/bodoai/experiments/unsupervised/EN-BRX-UnsupNMT/dataset/eng-brx/dev.tok.brx
pred_file=/home/bodoai/experiments/unsupervised/EN-BRX-UnsupNMT/data/validation_output/undreamt_model.valid.src2trg.0.6000.txt

# head -n 5 $ref_file
# head -n 5 $pred_file
# echo "Calculating BLEU scores for the model"
# sacrebleu $ref_file < $pred_file -m bleu chrf

VALID_SRC=/home/bodoai/experiments/unsupervised/EN-BRX-UnsupNMT/dataset/eng-brx/dev.tok.eng
VALID_TGT=/home/bodoai/experiments/unsupervised/EN-BRX-UnsupNMT/dataset/eng-brx/dev.tok.brx

echo "Calculating BLEU scores for the model"
sacrebleu $VALID_TGT < $pred_file -m bleu chrf --force