model_name=Llama-2-7b-chat-hf
task=$1

exp_name=$2

node_num=4  # number of GPUs
num_workers=4   # number of inference workers

model_path=$3 # path to the original LLM
save_dir=$4    # checkpoint save path

# Part 1: SFT stage
sft_data_path="data/${task}_sft.json"
batch_size=64
micro_batch_size=4
accumulation_step=$((${batch_size}/${node_num}/${micro_batch_size}))

sft_model_name=${exp_name}-${model_name}-${task}-sft

python -m torch.distributed.run --nproc_per_node=${node_num} --master_port=20001 fastchat/train/train.py \
    --model_name_or_path ${model_path}${model_name} \
    --data_path ${sft_data_path} \
    --bf16 True \
    --output_dir ${save_dir}${sft_model_name} \
    --num_train_epochs 3 \
    --per_device_train_batch_size ${micro_batch_size} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps ${accumulation_step} \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 5 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --lazy_preprocess False

# if failed, exit
if [ $? -ne 0 ]; then
    echo "SFT training failed"
    exit 1
fi

# launch the FastChat controller
python -u -m fastchat.serve.controller >> logs/${exp_name}-controller.log 2>&1 &
fs_controller_pid=$!

# Evaluate the base agent
fs_worker_port=21002
CUDA_VISIBLE_DEVICES=0 python -u -m fastchat.serve.model_worker --model-path ${save_dir}${sft_model_name} --port ${fs_worker_port} --worker-address http://localhost:${fs_worker_port} >> logs/${exp_name}-model_worker.log 2>&1 &

fs_worker_pid=$!
sleep 60

# evaluate on the test set
python -m eval_agent.main --agent_config fastchat --model_name ${sft_model_name} --exp_config ${task} --split test

# if failed, exit
if [ $? -ne 0 ]; then
    echo "base agent evaluation failed"
    kill -9 $fs_worker_pid
    exit 1
fi

# kill the model worker
kill -9 $fs_worker_pid

# The ETO iteration
cur_model_name=${sft_model_name}
for i in {1..3}; do
    # Part 2: base agent explore stage
    # launch multiple fastchat model workers
    explore_model_name=${cur_model_name}-explore

    for ((j=0;j<${num_workers};j=j+1)); do
        if [ -d "${save_dir}${explore_model_name}-${j}" ]; then
            echo "Link to model exists"
        else
            ln -s ${save_dir}${cur_model_name} ${save_dir}${explore_model_name}-${j}
        fi
    done
    if [ -f "logs/${exp_name}-worker_pid.txt" ]; then
        rm logs/${exp_name}-worker_pid.txt
    fi

    fs_worker_port=21002
    worker_idx=0
    for ((j=0;j<${num_workers};j=j+1)); do
        echo "Launch the model worker on port ${fs_worker_port}"
        CUDA_VISIBLE_DEVICES=$((${worker_idx} % ${node_num})) python -u -m fastchat.serve.model_worker \
            --model-path ${save_dir}${explore_model_name}-${j} \
            --port ${fs_worker_port} \
            --worker-address http://localhost:${fs_worker_port} >> logs/${exp_name}-model_worker-${j}.log 2>&1 &
        echo $! >> logs/${exp_name}-worker_pid.txt
        fs_worker_port=$(($fs_worker_port+1))
        worker_idx=$(($worker_idx+1))
        sleep 15
    done
    
    sleep 60

    # start explore on the same sft data
    echo "Base agent starts exploring"
    if [ -f "logs/${exp_name}-eval_pid.txt" ]; then
        rm logs/${exp_name}-eval_pid.txt
    fi
    for ((j=0;j<${num_workers};j=j+1)); do
        python -m eval_agent.main --agent_config fastchat --model_name ${explore_model_name}-${j} --exp_config ${task} --split train --part_num ${num_workers} --part_idx ${j} &
        echo $! >> logs/${exp_name}-eval_pid.txt
    done

    wait $(cat logs/${exp_name}-eval_pid.txt)
    rm logs/${exp_name}-eval_pid.txt
    echo "Base agent has finished exploring"

    # if failed, exit
    if [ $? -ne 0 ]; then
        echo "base agent exploration failed"
        kill -9 $(cat logs/${exp_name}-worker_pid.txt)
        rm logs/${exp_name}-worker_pid.txt
        exit 1
    fi

    # kill the model worker
    echo "Kill the model workers"
    kill -9 $(cat logs/${exp_name}-worker_pid.txt)
    rm logs/${exp_name}-worker_pid.txt

    # build trajectory preference data
    echo "Build preference data"
    pm_data_path=data_pm/${task}_pm_${exp_name}_${i}.json
    python construct_preference.py --model ${explore_model_name} --task $task --golden_traj_path $sft_data_path --output_path $pm_data_path

    # Part 3: preference model training stage
    batch_size=32
    micro_batch_size=2
    accumulation_step=$((${batch_size}/${node_num}/${micro_batch_size}))
    if [ ${i} -eq 1 ]; then
        beta=0.1
        lr=1e-6
    else
        beta=0.5
        lr=5e-7
    fi

    dpo_model_name=${exp_name}-${model_name}-${task}-dpo-iter-${i}

    python -m torch.distributed.run --nproc_per_node=${node_num} --master_port=20001 fastchat/train/train_dpo.py \
        --model_name_or_path ${save_dir}${cur_model_name} \
        --ref_model_name_or_path ${save_dir}${cur_model_name} \
        --data_path ${pm_data_path} \
        --bf16 True \
        --output_dir ${save_dir}${dpo_model_name} \
        --num_train_epochs 3 \
        --per_device_train_batch_size ${micro_batch_size} \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps ${accumulation_step} \
        --evaluation_strategy "no" \
        --save_strategy "no" \
        --save_total_limit 5 \
        --beta ${beta} \
        --learning_rate ${lr} \
        --weight_decay 0. \
        --warmup_ratio 0.1 \
        --lr_scheduler_type "constant_with_warmup" \
        --logging_steps 5 \
        --fsdp "full_shard auto_wrap" \
        --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
        --tf32 True \
        --model_max_length 4096 \
        --max_prompt_length 512 \
        --max_target_length 3072 \
        --gradient_checkpointing True \
        --lazy_preprocess False

    # if failed, exit
    if [ $? -ne 0 ]; then
        echo "Preference model training failed"
        exit 1
    fi

    # Part 4: Evaluate the agent
    fs_worker_port=21002
    CUDA_VISIBLE_DEVICES=0 python -u -m fastchat.serve.model_worker --model-path ${save_dir}${dpo_model_name} --port ${fs_worker_port} --worker-address http://localhost:${fs_worker_port} >> logs/model_worker.log 2>&1 &

    fs_worker_pid=$!
    sleep 60

    # evaluate on the test set
    python -m eval_agent.main --agent_config fastchat --model_name ${dpo_model_name} --exp_config ${task} --split test

    # if failed, exit
    if [ $? -ne 0 ]; then
        echo "base agent evaluation failed"
        kill -9 $fs_worker_pid
        exit 1
    fi

    # kill the model worker
    kill -9 $fs_worker_pid

    cur_model_name=${dpo_model_name}
done

# kill the controller
kill -9 $fs_controller_pid
