export http_proxy=http://oversea-squid1.jp.txyun:11080 https_proxy=http://oversea-squid1.jp.txyun:11080 no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com
CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --category=Beauty \
    --lr=0.01 \
    --temperature=0.03 \
    --n_codebook=32 \
    --num_beams=20 \
    --n_edges=200 \
    --propagation_steps=3