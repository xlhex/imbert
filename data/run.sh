TASK=$1
SEED=$2
CKPT=ckpt/bert/backdoor/$TASK/$SEED
TRAIN=data/$TASK/train.json
VALID=data/$TASK/test_clean.json
TEST=data/$TASK/test_poison.json
MODEL=bert-base-uncased
LOG=log/bert/backdoor/$TASK/$SEED
if [ ! -d $LOG ];then
    mkdir -p $LOG
fi

# Training victim model
log=$LOG/train_log.txt
python run_glue.py --seed $SEED --model_name_or_path $MODEL --output_dir ${CKPT} --save_total_limit 1 --do_train --do_eval --train_file ${TRAIN} --validation_file ${VALID} --max_seq_length 128 --per_device_train_batch_size 32  --learning_rate 2e-5 --num_train_epochs 3 --per_device_eval_batch_size 32 > $log 2>&1
rm -rf ${CKPT}/checkpoint-*

# Test victim model on clean set
VALID=data/$TASK/test_clean.json
log=$LOG/clean_log.txt
python run_glue.py --model_name_or_path ${CKPT} --output_dir ${CKPT} --do_eval --train_file ${VALID} --validation_file ${VALID} --max_seq_length 128 --per_device_train_batch_size 32  --learning_rate 2e-5 --num_train_epochs 3 --per_device_eval_batch_size 32 > $log 2>&1

# Test victim model on poisoning set
TEST=data/$TASK/test_poison.json
log=$LOG/backdoor_log.txt
python run_glue.py --model_name_or_path ${CKPT} --output_dir ${CKPT} --do_eval --train_file ${VALID} --validation_file ${TEST} --max_seq_length 128 --per_device_train_batch_size 32  --learning_rate 2e-5 --num_train_epochs 3 --per_device_eval_batch_size 32 > $log 2>&1
