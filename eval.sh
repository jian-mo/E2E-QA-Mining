module load StdEnv/2020     gcc/9.3.0  cuda/11.4 scipy-stack arrow/5.0.0
source /home/mojians/projects/def-lulam50/mojians/tqEnv/bin/activate
#-m torch.distributed.launch --nproc_per_node=2
model=$1
deepspeed=$2
do_sample=$3

if [ $deepspeed ]; then
   deepspeed --num_gpus=1 eval.py  --model_name_or_path t5-$model-aeqg-hl --valid_file_path data/valid_data_aeqg_prepend_qg_format_t5.pt  --model_type t5 --output_dir output/hypothesis-"${model}".txt --max_decoding_length 1000 --do_sample $do_sample --deepspeed ds_config.json

else
  export CUDA_VISIBLE_DEVICES=0,1,2,3

  CUDA_LAUNCH_BLOCKING=1 python -m torch.distributed.launch --nproc_per_node=2 run_qg.py --model_name_or_path t5-$model --resume_from_checkpoint t5-$model-aeqg-hl --model_type t5 --tokenizer_name_or_path t5_qg_tokenizer --output_dir t5-$model-aeqg-hl --train_file_path data/train_data_aeqg_prepend_qg_format_t5.pt --valid_file_path data/valid_data_aeqg_prepend_qg_format_t5.pt --per_device_train_batch_size 4 --per_device_eval_batch_size 1 --gradient_accumulation_steps 8 --learning_rate 1e-4 --num_train_epochs 3 --seed 42 --do_train --do_eval --logging_steps 1000    --do_eval True --do_train True --sharded_ddp simple --fp16
fi
