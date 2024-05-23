#!/bin/bash 
# swaubhik 2024

set -e

N_THREADS=128     # number of threads in data preprocessing
N_EPOCHS=10      # number of fastText epochs

# create dataset folder
# main paths
UMT_PATH=$PWD
TOOLS_PATH=$PWD/tools
DATA_PATH=$PWD/dataset/eng-brx

# create paths
mkdir -p $TOOLS_PATH

# moses
MOSES=$TOOLS_PATH/mosesdecoder
TOKENIZER=$MOSES/scripts/tokenizer/tokenizer.perl
NORM_PUNC=$MOSES/scripts/tokenizer/normalize-punctuation.perl
INPUT_FROM_SGM=$MOSES/scripts/ems/support/input-from-sgm.perl
REM_NON_PRINT_CHAR=$MOSES/scripts/tokenizer/remove-non-printing-char.perl

# fastText
FASTTEXT_DIR=$TOOLS_PATH/fastText
FASTTEXT=$FASTTEXT_DIR/fasttext

# undreamt
UNDREAMT_DIR=$TOOLS_PATH/undreamt

# langs
SRC=eng
TGT=brx

# files full paths
SRC_RAW=$DATA_PATH/train/train.$SRC
TGT_RAW=$DATA_PATH/train/train.$TGT
SRC_TOK=$DATA_PATH/train.tok.$SRC
TGT_TOK=$DATA_PATH/train.tok.$TGT
SRC_VALID=$DATA_PATH/dev/dev.$SRC
TGT_VALID=$DATA_PATH/dev/dev.$TGT
SRC_TEST=$DATA_PATH/test/test.$SRC
TGT_TEST=$DATA_PATH/test/test.$TGT
SRC_VALID_TOK=$DATA_PATH/dev.tok.$SRC
TGT_VALID_TOK=$DATA_PATH/dev.tok.$TGT
SRC_TEST_TOK=$DATA_PATH/test.tok.$SRC
TGT_TEST_TOK=$DATA_PATH/test.tok.$TGT


#
# Download and install tools
#

# Download Moses
cd $TOOLS_PATH
if [ ! -d "$MOSES" ]; then
  echo "Cloning Moses from GitHub repository..."
  git clone https://github.com/moses-smt/mosesdecoder.git
fi
echo "Moses found in: $MOSES"

# Download undreamt
cd $TOOLS_PATH
if [ ! -d "$UNDREAMT_DIR" ]; then
  echo "Cloning undreamt from GitHub repository..."
  git clone https://github.com/swaubhik/undreamt.git
fi

echo "undreamt found in: $UNDREAMT_DIR"

# Download fastText
cd $TOOLS_PATH
if [ ! -d "$FASTTEXT_DIR" ]; then
  echo "Cloning fastText from GitHub repository..."
  git clone https://github.com/facebookresearch/fastText.git
fi
echo "fastText found in: $FASTTEXT_DIR"

# Compile fastText
cd $TOOLS_PATH
if [ ! -f "$FASTTEXT" ]; then
  echo "Compiling fastText..."
  cd $FASTTEXT_DIR
  make
fi
echo "fastText compiled in: $FASTTEXT"

# download vecmap
cd $TOOLS_PATH
if [ ! -d "vecmap" ]; then
  echo "Cloning vecmap from GitHub repository..."
  git clone https://github.com/artetxem/vecmap.git
fi
echo "vecmap found in: $TOOLS_PATH/vecmap"

# tokenize data
if ! [[ -f "$SRC_TOK" && -f "$TGT_TOK" ]]; then
  echo "Tokenize monolingual data..."
  cat $SRC_RAW | $NORM_PUNC -l en | $TOKENIZER -l en -no-escape -threads $N_THREADS > $SRC_TOK
  cat $TGT_RAW | $NORM_PUNC -l hi | $TOKENIZER -l hi -no-escape -threads $N_THREADS > $TGT_TOK
fi
echo "EN monolingual data tokenized in: $SRC_TOK"
echo "FR monolingual data tokenized in: $TGT_TOK"

echo "Tokenizing valid and test data..."
cat $SRC_VALID | $NORM_PUNC -l en | $TOKENIZER -l en -no-escape -threads $N_THREADS > $SRC_VALID_TOK
cat $TGT_VALID | $NORM_PUNC -l hi | $TOKENIZER -l hi -no-escape -threads $N_THREADS > $TGT_VALID_TOK

cat $SRC_TEST | $NORM_PUNC -l en | $TOKENIZER -l en -no-escape -threads $N_THREADS > $SRC_TEST_TOK
cat $TGT_TEST | $NORM_PUNC -l hi | $TOKENIZER -l hi -no-escape -threads $N_THREADS > $TGT_TEST_TOK

#
# Summary
#
echo ""
echo "===== Data summary"
echo "Monolingual data:"
echo "    ENG: $(wc -l < $SRC_RAW) lines"
echo "    BRX: $(wc -l < $TGT_RAW) lines"
echo "Tokenized monolingual data:"
echo "    ENG: $(wc -l < $SRC_TOK) lines"
echo "    BRX: $(wc -l < $TGT_TOK) lines"
echo "Valid data:"
echo "    ENG: $(wc -l < $SRC_VALID) lines"
echo "    BRX: $(wc -l < $TGT_VALID) lines"
echo "Test data:"
echo "    ENG: $(wc -l < $SRC_TEST) lines"
echo "    BRX: $(wc -l < $TGT_TEST) lines"
echo "===== End of data summary"
echo ""

SRC_EMBEDDINGS=$DATA_PATH/eng
TGT_EMBEDDINGS=$DATA_PATH/brx

# train fastText embeddings
if ! [[ -f "$SRC_EMBEDDINGS" && -f "$TGT_EMBEDDINGS" ]]; then
  echo "Training fastText embeddings..."
  $FASTTEXT skipgram -input $SRC_TOK -output $SRC_EMBEDDINGS -minCount 5 -thread $N_THREADS -epoch $N_EPOCHS
  $FASTTEXT skipgram -input $TGT_TOK -output $TGT_EMBEDDINGS -minCount 5 -thread $N_THREADS -epoch $N_EPOCHS
fi

echo "EN embeddings trained in: $SRC_EMBEDDINGS.vec"
echo "FR embeddings trained in: $TGT_EMBEDDINGS.vec"

SRC_MAPPED=$DATA_PATH/eng.map
TGT_MAPPED=$DATA_PATH/brx.map

# train the cross-lingual embeddings using vecmap
CUDA_VISIBLE_DEVICES=1 python3 $TOOLS_PATH/vecmap/map_embeddings.py \
    --unsupervised $SRC_EMBEDDINGS.vec $TGT_EMBEDDINGS.vec $SRC_MAPPED $TGT_MAPPED\
    --cuda
