python exp_eventPoincare_static.py --precision double --nn_hid_dim 2 --step_size 10 --max_epoch 20 --weight_decay 1e-2 --batch 300 --context roberta-base --expname mypoincare_sd102_2d_neg1 --lr 0.001 --testsetname matres --cuda 1 --skiptraining

python exp_eventPoincare_static.py --precision double --nn_hid_dim 2 --step_size 10 --max_epoch 20 --weight_decay 1e-2 --batch 300 --context roberta-base --expname mypoincare_sd102_2d_neg1 --lr 0.001 --testsetname tcr --cuda 1  --skiptraining
