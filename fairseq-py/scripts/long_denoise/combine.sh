#!/bin/bash


# echo "*** Split into train shards and valid ***"
# split_data() {
#     get_seeded_random() {
#         seed="$1"; openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt </dev/zero 2>/dev/null
#     };
#     NLINES=`wc -l $1  | awk -F " " '{print $1}'`;
    
#     NTRAIN=$((NLINES - 1000000));
#     NSHARD=$((NTRAIN/5));
#     NBLOCK=$NSHARD;

#     for SHARD in {0..4}
#     do
#         echo "writing lines $((NBLOCK-NSHARD)) - $NBLOCK into $2/shard${SHARD}.txt";
#         shuf --random-source=<(get_seeded_random 42) $1 | head -$NBLOCK | tail -$NSHARD  > $2/shard${SHARD}.txt;
#         NBLOCK=$((NBLOCK + NSHARD));
#     done
#     shuf --random-source=<(get_seeded_random 42) $1 | tail -1000000                > $2/valid.txt;
# }

# split_data /fsx/xwhan/data/pretrain_corpus/realnews/realnews.bpe.filtered /fsx/xwhan/data/pretrain_corpus/realnews

LONGFORMER_DIR=/fsx/xwhan/data/pretrain_corpus/longformer_bookwiki_stories_realnes-bin


# STORIES 1591.0 + 1591.7 + 1594.6 + 1590.4 + 1590.4 = 7958.1

ln -sf $LONGFORMER_DIR/stories-mmap2-bin/shard0/train.bin train.bin
ln -sf $LONGFORMER_DIR/stories-mmap2-bin/shard0/train.idx train.idx

for SHARD in {1..4};
do
    ln -sf $LONGFORMER_DIR/stories-mmap2-bin/shard${SHARD}/train.bin train$SHARD.bin
    ln -sf $LONGFORMER_DIR/stories-mmap2-bin/shard${SHARD}/train.idx train$SHARD.idx
done


# BookWiki 799.6 + 799.7 + 790.8 + 799.1 + 801.5 = 3990.7
for SHARD in {0..4};
do
    ln -sf $LONGFORMER_DIR/bookwiki_aml-mmap2-bin/shard${SHARD}/train.bin train$(( 5+SHARD )).bin
    ln -sf $LONGFORMER_DIR/bookwiki_aml-mmap2-bin/shard${SHARD}/train.idx train$(( 5+SHARD )).idx
done

# RealNews 1709.0 + 1708.3 + 1708.3 + 1708.2 = 6833.8
for SHARD in {0..4};
do
    ln -sf $LONGFORMER_DIR/realnews-filtered-bin/shard${SHARD}/train.bin train$(( 10+SHARD )).bin
    ln -sf $LONGFORMER_DIR/realnews-filtered-bin/shard${SHARD}/train.idx train$(( 10+SHARD )).idx
done

## 18781.9

# PG19 873.2 + 978.11 + 994.5 = 2845.8
PG19_DIR=/fsx/xwhan/data/pretrain_corpus/pg19/bin
for SHARD in {0..2};
do
    ln -sf $PG19_DIR/$SHARD/train.bin train$(( 15+SHARD )).bin
    ln -sf $PG19_DIR/$SHARD/train.idx train$(( 15+SHARD )).idx;
done

# Books3 2629.3 * 10 = 26293
BOOKS3_DIR=/fsx/xwhan/data/pretrain_corpus/books3/bin
for SHARD in {0..9};
do
    ln -sf $BOOKS3_DIR/$SHARD/train.bin train$(( 18+SHARD )).bin
    ln -sf $BOOKS3_DIR/$SHARD/train.idx train$(( 18+SHARD )).idx;
done


# dialogue datasets 973.6 + 3838.7
MEDIASUM_DIR=/fsx/xwhan/data/pretrain_corpus/dialogue/mediasum/bin
ln -sf $MEDIASUM_DIR/train.bin train28.bin
ln -sf $MEDIASUM_DIR/train.idx train28.idx;

OPENSUM_DIR=/fsx/xwhan/data/pretrain_corpus/dialogue/opensub/bin
ln -sf $OPENSUM_DIR/train.bin train29.bin
ln -sf $OPENSUM_DIR/train.idx train29.idx;

# 2/10 of c4 data 17502.3 each
C4_DIR=/fsx/xwhan/data/pretrain_corpus/c4/bin

for SHARD in {0..1};
do
    ln -sf $C4_DIR/train$SHARD/train.bin train$(( 30+SHARD )).bin
    ln -sf $C4_DIR/train$SHARD/train.idx train$(( 30+SHARD )).idx;
done

ln -sf $LONGFORMER_DIR/stories-mmap2-bin/valid/valid.bin valid.bin
ln -sf $LONGFORMER_DIR/stories-mmap2-bin/valid/valid.idx valid.idx


ln -sf $LONGFORMER_DIR/realnews-filtered-bin/valid/valid.bin valid1.bin
ln -sf $LONGFORMER_DIR/realnews-filtered-bin/valid/valid.idx valid1.idx


ln -sf $BOOKS3_DIR/valid/train.bin valid2.bin
ln -sf $BOOKS3_DIR/valid/train.idx valid2.idx;

# ln -s $LONGFORMER_DIR/bookwiki_aml-mmap2-bin/valid/valid.bin valid1.bin
# ln -s $LONGFORMER_DIR/bookwiki_aml-mmap2-bin/valid/valid.idx valid1.idx


ln -sf $C4_DIR/valid/valid.bin valid3.bin
ln -sf $C4_DIR/valid/valid.idx valid3.idx;


ln -sf $OPENSUM_DIR/valid.bin valid4.bin
ln -sf $OPENSUM_DIR/valid.idx valid4.idx;

ln -sf $MEDIASUM_DIR/valid.bin valid5.bin
ln -sf $MEDIASUM_DIR/valid.idx valid5.idx;


ln -sf /dataset
