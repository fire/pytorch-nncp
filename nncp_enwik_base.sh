#!/bin/bash
#
# NNCP v2 enwik8/enwik9 compression and decompression
#
cmd="$1"
file="$2"

if [ '(' "$cmd" != "c" -a "$cmd" != "d" ')' -o \
     '(' "$file" != "enwik8" -a "$file" != "enwik9" ')' ]; then
    echo "usage: nncp_enwik.sh [c|d] [enwik8|enwik9]"
    echo "NNCP v2 enwik8/enwik9 compression test"
    echo "c   compress 'file' to 'file.bin' and 'file.voc'"
    echo "d   decompress 'file.bin' and 'file.voc' to 'file.out'"
    echo "    with file=enwik8 or enwik9"
    exit 1
fi

infile=${file}
vocfile=${file}.voc
ppfile=${file}.pp
cmpfile=${file}.bin
ppoutfile=${file}.ppout
outfile=${file}.out

if [ "$cmd" == "c" ]; then
  if [ "$file" == "enwik8" ]; then
      ./preprocess c $vocfile $infile $ppfile 16384 64
  else
      ./preprocess c $vocfile $infile $ppfile 16384 512
  fi
fi

if [ "$cmd" == "c" ]; then
    nncp_args="--input $ppfile --output $cmpfile"
else
    nncp_args="--decompress --input $cmpfile --output $ppoutfile"
fi

# needed for deterministic PyTorch
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python nncp.py $nncp_args \
       --debug \
       --cuda \
       --fp16 \
       --vocab_size 16388  \
       --block_len 500000 \
       --n_layer 12 \
       --d_model 512 \
       --n_head 8 \
       --d_head 64 \
       --d_inner 2048 \
       --attn_type 1 \
       --tied_r_bias \
       --dropout 0.25 \
       --dropatt 0.0 \
       --lr 7.9e-5,341105,1.6e-5,3134681,5.0e-6 \
       --tgt_len 1 \
       --ext_tgt_len 31 \
       --mem_len 160 \
       --batch_size 64 \
       --init_std 0.013 \
       --adam_eps 1e-9 \
       --clip 0.25 \
       --retrain_period 500000 \
       --retrain_len 10000000 \
       --retrain_tgt_len 64 \
       --retrain_mem_len 128 \
       --retrain_batch_size 32 \
       --retrain_lr 4.0e-4,13000,2.0e-4,93000,1.0e-4,163000,5.0e-5,1911300,1.6e-5 \
       --gelu


if [ "$cmd" == "d" ]; then
  ./preprocess d $vocfile $ppoutfile $outfile
fi
