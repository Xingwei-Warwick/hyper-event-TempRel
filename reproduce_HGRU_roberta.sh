python exp_HGRU.py --hid_dim 128 --nn_hid_dim 64 --step_size 10 --max_epoch 15 --gamma 0.3 --common_sense_emb_dim 32 --granularity 0.5 --bigramstats_dim 1 --weight_decay 1e-1 --sd 102 --batch 20 --expname HGRU_cse_gran5_sd102_wd01 --lr 0.001 --context roberta-base --testsetname matres --cuda 1

python exp_HGRU.py --hid_dim 128 --nn_hid_dim 64 --step_size 10 --max_epoch 15 --gamma 0.3 --common_sense_emb_dim 32 --granularity 0.5 --bigramstats_dim 1 --weight_decay 1e-1 --sd 102 --batch 20 --expname HGRU_cse_gran5_sd102_wd01 --lr 0.001 --context roberta-base --testsetname tcr --cuda 1
