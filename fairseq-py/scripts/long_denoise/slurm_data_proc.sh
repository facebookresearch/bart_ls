#!/bin/bash

JOBSCRIPTS=slurm_scripts
mkdir -p ${JOBSCRIPTS}
JOB=binarize_mediasum_

for shard_id in {0..9}
    do

    # for split_id in {0..4}
    # do

    # JNAME=${JOB}_${shard_id}_${split_id}
    JNAME=${JOB}_${shard_id}
    mkdir -p /fsx/xwhan/checkpoints/sjobs/${JOB}
    SCRIPT=${JOBSCRIPTS}/run.${JNAME}.sh
    SLURM=${JOBSCRIPTS}/run.${JNAME}.slrm

    echo "#!/bin/sh" > ${SCRIPT}
    echo "#!/bin/sh" > ${SLURM}
    echo "#SBATCH --job-name=$JNAME" >> ${SLURM}
    echo "#SBATCH --output=/fsx/xwhan/checkpoints/sjobs/${JOB}/${JNAME}.%j.out" >> ${SLURM}
    echo "#SBATCH --error=/fsx/xwhan/checkpoints/sjobs/${JOB}/${JNAME}.%j.err" >> ${SLURM}
    echo "#SBATCH --partition=a100" >> ${SLURM}
    echo "#SBATCH --gres=gpu:2" >> ${SLURM}
    echo "#SBATCH --time=4320" >> ${SLURM}
    echo "#SBATCH --nodes=1" >> ${SLURM}
    echo "#SBATCH --cpus-per-task=10" >> ${SLURM}
    echo "srun sh ${SCRIPT}" >> ${SLURM}
    echo "echo \$SLURM_JOB_ID >> jobs" >> ${SCRIPT}
    echo "{ " >> ${SCRIPT}

    # echo "cat /fsx/xwhan/data/pretrain_corpus/assembled_c4_top2k/shards/${shard_id}/c_*.txt > /fsx/xwhan/data/pretrain_corpus/assembled_c4_top2k/shards/shard${shard_id}.txt" >> ${SCRIPT}
    # echo "cd /fsx/xwhan/data/pretrain_corpus/c4/long/assembled/shards" >> ${SCRIPT}

    # echo "python -m examples.roberta.multiprocessing_bpe_encoder --encoder-json gpt2_bpe/encoder.json --vocab-bpe gpt2_bpe/vocab.bpe --inputs /fsx/xwhan/data/pretrain_corpus/assembled_c4_top2k/shards/shard${shard_id}_0${split_id} --outputs /fsx/xwhan/data/pretrain_corpus/assembled_c4_top2k/bpe/${shard_id}_${split_id}.bpe --keep-empty --workers 20" >> ${SCRIPT}

    echo "fairseq-preprocess --only-source --trainpref /fsx/xwhan/data/pretrain_corpus/dialogue/mediasum/bpe/shard0${shard_id} --destdir /fsx/xwhan/data/pretrain_corpus/dialogue/mediasum/bin/shard${shard_id}/train${split_id} --workers 20 --srcdict gpt2_bpe/dict.txt" >> ${SCRIPT}

    # echo "fairseq-preprocess --only-source --trainpref /fsx/xwhan/data/pretrain_corpus/realnews/shard.bpe${shard_id} --destdir /fsx/xwhan/data/pretrain_corpus/realnews/bin/${split_id} --workers 30 --srcdict gpt2_bpe/dict.txt" >> ${SCRIPT}

    # echo "cd /fsx/xwhan/data/pretrain_corpus/assembled_c4_top2k/bpe" >> ${SCRIPT}

    # echo "cat ${shard_id}_0.bpe ${shard_id}_1.bpe > shard${shard_id}_0.bpe" >> ${SCRIPT}

    # echo "cat ${shard_id}_2.bpe ${shard_id}_3.bpe > shard${shard_id}_1.bpe" >> ${SCRIPT}

    # echo "cat ${shard_id}_4.bpe ${shard_id}_5.bpe > shard${shard_id}_2.bpe" >> ${SCRIPT}

    # echo "cat ${shard_id}_6.bpe ${shard_id}_7.bpe > shard${shard_id}_3.bpe" >> ${SCRIPT}

    # echo "cat ${shard_id}_8.bpe ${shard_id}_9.bpe > shard${shard_id}_4.bpe" >> ${SCRIPT}

    # echo "rm ${shard_id}_*" >> ${SCRIPT}

    # echo "cd /fsx/xwhan/data/pretrain_corpus/assembled_c4_top2k/shards" >> ${SCRIPT}
    # echo "rm shard${shard_id}_*" >> ${SCRIPT}
    # echo "split --number=l/10 --numeric-suffixes=0 shard${shard_id}.txt shard${shard_id}_" >> ${SCRIPT}
    # echo "rm shard${shard_id}.txt" >> ${SCRIPT}


    # echo "fairseq-preprocess --only-source --trainpref /fsx/xwhan/data/pretrain_corpus/realnews/shard.bpe${shard_id} --destdir /fsx/xwhan/data/pretrain_corpus/realnews/bin/${split_id} --workers 30 --srcdict gpt2_bpe/dict.txt" >> ${SCRIPT}

    # echo "rm ${shard_id}_*.bpe" >> ${SCRIPT}

    # echo python /data/home/xwhan/fairseq-py/scripts/long_denoise/make_c4_longer.py $shard_id  >> ${SCRIPT}

    # echo python -m pyserini.index.lucene --collection JsonCollection --input /data/home/xwhan/data/long_c4/jsonl_files/shard${shard_id} --index /data/home/xwhan/data/long_c4/indexes/shard${shard_id} --generator DefaultLuceneDocumentGenerator --threads 30 --storePositions >> ${SCRIPT}

    echo "kill -9 \$\$" >> ${SCRIPT}
    echo "} & " >> ${SCRIPT}
    echo "child_pid=\$!" >> ${SCRIPT}
    echo "trap \"echo 'TERM Signal received';\" TERM" >> ${SCRIPT}
    echo "trap \"echo 'Signal received'; if [ \"\$SLURM_PROCID\" -eq \"0\" ]; then sbatch ${SLURM}; fi; kill -9 \$child_pid; \" USR1" >> ${SCRIPT}
    echo "while true; do     sleep 1; done" >> ${SCRIPT}

    # done
done

for shard_id in {0..9}
    do
    # for split_id in {0..4}
    # do
        sbatch ./slurm_scripts/run.${JOB}_${shard_id}.slrm &
    # done
done