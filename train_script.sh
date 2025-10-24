CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7  deepspeed --num_gpus 8 run_train.py --ds_config ds_config_zero2.json
