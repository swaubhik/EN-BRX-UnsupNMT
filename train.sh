# conda activate undreamt
SRC=eng
TGT=brx

DATA_PATH=$PWD/dataset/eng-brx

SRC_TOK=$DATA_PATH/train.tok.$SRC
TGT_TOK=$DATA_PATH/train.tok.$TGT

VALID_SRC=$DATA_PATH/dev.tok.$SRC
VALID_TGT=$DATA_PATH/dev.tok.$TGT

UMT_PATH=$PWD
TOOLS_PATH=$PWD/tools

SRC_EMBEDDINGS=$DATA_PATH/eng.map
TGT_EMBEDDINGS=$DATA_PATH/brx.map

UNDREAMT_PATH=$TOOLS_PATH/undreamt
mkdir -p data/models
mkdir -p data/validation_output
# train the model
CUDA_VISIBLE_DEVICES=1 python tools/undreamt/train.py \
    --batch 50 \
    --src $SRC_TOK \
    --trg $TGT_TOK \
    --src_embeddings $SRC_EMBEDDINGS \
    --trg_embeddings $TGT_EMBEDDINGS \
    --save data/models/undreamt_model \
    --save en_brx \
    --cuda \
    --validation $VALID_SRC $VALID_TGT \
    --validation_output data/validation_output/undreamt_model.valid \
    --save_interval 10000 \
  
# save tmux log to a file
echo "Training completed. Saving log to train.log"
tmux capture-pane -t 0 -p -S -1000000 > train.log
