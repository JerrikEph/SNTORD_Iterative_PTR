#!/bin/bash
emb=(100)
enc=(200)
dec=(200)
rep=(200)
if [ "$1" == "" ]
then
    echo "specify gpu number"
    exit
fi
gpu_num=$1
root_dir=.
for emb_sz in ${emb[*]}
do
    for h_enc_sz in ${enc[*]}
    do
        for h_dec_sz in ${dec[*]}
        do
            for h_rep_sz in ${rep[*]}
            do
                pad_step=emb${emb_sz}_enc${h_enc_sz}_dec${h_dec_sz}_rep${h_rep_sz}
                tmplt_config=${root_dir}/savings/config.tmplt
                save_dir=${root_dir}/savings/lstm_${pad_step}
                rm -rf ${save_dir}
                mkdir ${save_dir}
                awk '{f=0}                                             \
                    /embed_size =/{f=1; $3="'${emb_sz}'"; print $0}    \
                    /h_enc_sz =/  {f=2; $3="'${h_enc_sz}'"; print $0}  \
                    /h_dec_sz =/  {f=3; $3="'${h_dec_sz}'"; print $0}  \
                    /h_rep_sz =/  {f=4; $3="'${h_rep_sz}'"; print $0}  \
                    f==0{print $0}                                     \
                    ' ${tmplt_config} > ${save_dir}/config
                CUDA_VISIBLE_DEVICES=${gpu_num} python model.py --load-config --weight-path ${save_dir}
                CUDA_VISIBLE_DEVICES=${gpu_num} python model.py --load-config --weight-path ${save_dir} --train-test test
            done
        done
    done
done

