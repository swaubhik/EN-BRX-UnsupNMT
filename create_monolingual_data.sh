# !/bin/bash
# swaubhik 2024
SRC=eng
TGT=brx
# create dataset folder
mkdir -p dataset/eng-brx
# create brx monolingual data
INDICTRANS=/home/bodoai/IndicTrans2-data/BPCC

cat $INDICTRANS/daily/eng_Latn-brx_Deva/train.brx_Deva $INDICTRANS/ilci/eng_Latn-brx_Deva/train.brx_Deva $INDICTRANS/wiki/eng_Latn-brx_Deva/train.brx_Deva >> dataset/eng-brx/train.brx_Deva

wget -c -O "dataset/eng-brx/temp.$TGT" "https://objectstore.e2enetworks.net/ai4b-public-nlu-nlg/indic-corp-frozen-for-the-paper-oct-2022/bd.txt"

SANGRAHA=/home/bodoai/experiments/unsupervised/exp1/UnsupervisedMT/en-brx/indicllm/bd.txt

INDICTRANS_BT=/home/bodoai/IndicTrans2-data/BT_data/indic_synthetic/brx_Deva-eng_Latn/train.brx_Deva

# get 100k lines from the INDICTRANS_BT
head -1000000 $INDICTRANS_BT > dataset/eng-brx/train.brx_Deva_BT

cat $SANGRAHA dataset/eng-brx/temp.$TGT dataset/eng-brx/train.brx_Deva dataset/eng-brx/train.brx_Deva_BT > dataset/eng-brx/train.brx

rm dataset/eng-brx/temp.$TGT
rm dataset/eng-brx/train.brx_Deva
rm dataset/eng-brx/train.brx_Deva_BT

# remove duplicates
awk '!seen[$0]++' dataset/eng-brx/train.brx > dataset/eng-brx/train.brx.tmp
mv dataset/eng-brx/train.brx.tmp dataset/eng-brx/train.brx

# suffle the data
shuf dataset/eng-brx/train.brx > dataset/eng-brx/train.brx.tmp
mv dataset/eng-brx/train.brx.tmp dataset/eng-brx/train.brx

# remove empty lines
sed -i '/^$/d' dataset/eng-brx/train.brx

# count number of lines and save it on a variable
echo "processing train.$SRC"

NUMBER_OF_LINES=$(wc -l < dataset/eng-brx/train.brx)

# get the eng monolingual data
head -$NUMBER_OF_LINES /home/bodoai/experiments/unsupervised/exp1/UnsupervisedMT/en-brx/en.txt > train.$SRC

# print the number of lines
echo "Number of lines in train.$SRC: $NUMBER_OF_LINES"

# create valid data
VALID_SRC=/home/bodoai/IndicTrans2-data/flores-22_dev/all/eng_Latn-brx_Deva/dev.eng_Latn
VALID_TGT=/home/bodoai/IndicTrans2-data/flores-22_dev/all/eng_Latn-brx_Deva/dev.brx_Deva

cat $VALID_SRC > dataset/eng-brx/dev/dev.$SRC
cat $VALID_TGT > dataset/eng-brx/dev/dev.$TGT


