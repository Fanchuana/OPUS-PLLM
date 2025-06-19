#!/bin/bash
#SBATCH --job-name=eval_loop_llama3# 作业名称
#SBATCH --output=eval_loop_llama3.out  # 输出文件
#SBATCH --error=eval_loop_llama3.err  # 错误输出文件
#SBATCH --ntasks=1                        # 总任务数量
#SBATCH --cpus-per-task=4                 # 每个任务的CPU核数
#SBATCH --gres=gpu:4                     # 申请 4 个 GPU
#SBATCH --mem=32G                         # 分配的内存
#SBATCH --time=10-00:00:00                # 最长运行时间，格式为 HH:MM:SS
#SBATCH --partition=normal                # 使用的分区（指定为 normal）
#SBATCH --nodelist=g01n05                 # 指定节点为 g01n01

# 加载 CUDA 11.8 模块
module load cuda11.8

# 激活你的 Conda 虚拟环境
source activate OpusPLLM
export PYTHONPATH="/work/home/cryoem666/xyf/temp/pycharm/OPUS-PLLM-GithubTest/"
# 切换到工作目录
cd "/work/home/cryoem666/xyf/temp/pycharm/OPUS-PLLM-GithubTest/multi_modality_model/multi_modality_v1/eval/"

current_date=$(date +"%Y-%m-%d")

output_dir="./downstream/ablation_study/("$current_date")"
mkdir -p "$output_dir"

json_paths=(
    "./downstream/subcellular_localization/OPI_localization_test.json"
    "test_localization.json"
    "./downstream/Uniprot2024/uniprot2024_localization_test_less2500.json"
    "./downstream/keywords_prediction/GO/OPI_CASPSimilarGO_Test_unique.json"
    "./downstream/keywords_prediction/GO/OPI_IDFilterGO_Test_unique.json"
    "./downstream/keywords_prediction/GO/OPI_UniProtGO_Test_unique.json"
    "./downstream/Uniprot2024/uniprot2024_go_test_less2500.json"
    "./downstream/keywords_prediction/Keywords/OPI_CASPSimilarSeq_keywords_test_unique.json"
   "./downstream/keywords_prediction/Keywords/OPI_IDFilterSeq_keywords_test_unique.json"
   "./downstream/keywords_prediction/Keywords/OPI_UniProtSeq_keywords_test_unique.json"
   "./downstream/Uniprot2024/uniprot2024_keywords_test_less2500.json"
   "./downstream/function_description/OPI_CASPSimilarSeq_function_test_unique.json"
   "./downstream/function_description/OPI_IDFilterSeq_function_test_unique.json"
   "./downstream/function_description/OPI_UniProtSeq_function_test_unique.json"
   "./downstream/Uniprot2024/uniprot2024_function_test_less2500.json"
   "./downstream/keywords_prediction/EC_Number/EC_number_New392_with_Name.json"
   "./downstream/keywords_prediction/EC_Number/EC_number_Price149_with_Name.json"
)
number_list=(
1
2
3
4
5
)
# 定义模型路径
adapter_paths=(
   "/work/home/cryoem666/xyf/temp/pycharm/huggingface/Opus_PLLM_Llama3-8B-Base/"
)

# 运行训练指令
for json_path in "${json_paths[@]}"; do
    for adapter_path in "${adapter_paths[@]}"; do
        for number in "${number_list[@]}"; do
          echo "Running task with json_path=$json_path and adapter_path=$adapter_path"
          save_path="$output_dir/$(basename "$json_path" .json)_$(basename "$adapter_path")_$(basename "$number").json"
          # 运行任务
          accelerate launch run_opus_ddp.py \
              --model-base-path "/work/home/cryoem666/xyf/temp/pycharm/model/llama3/" \
              --opus-pllm-weights-path "$adapter_path" \
              --input_path "$json_path" \
              --save_path "$save_path" 
              #--use_cstp  
          done
    done
done
#"/work/home/cryoem666/xyf/temp/pycharm/OPUS-BioLLM-CSTP/multi_modality_model/multi_modality_v1/train/checkpoints/v1.5/PT/Qwen2.5_3B/protein_caption_2.23M_pretrain/switch_projector.bin"
#"/work/home/cryoem666/xyf/temp/pycharm/model/Qwen2.5/Qwen2.5-1.5B-Instruct/"
#"/work/home/cryoem666/xyf/temp/pycharm/model/Qwen2.5/Qwen2.5-0.5B-Instruct/"
#"/work/home/cryoem666/xyf/temp/pycharm/model/Qwen2.5/Qwen2.5-3B-Instruct/"
#"/work/home/cryoem666/xyf/temp/pycharm/model/Qwen2.5/Qwen2.5-7B-Instruct/"
#"/work/home/cryoem666/xyf/temp/pycharm/model/Galactica_EvoInstruct/"
#"/work/home/cryoem666/xyf/temp/pycharm/model/llama3.2/llama3.2_1B_instruct/"
#"/work/home/cryoem666/xyf/temp/pycharm/model/llama3.2/llama3.2_3B_instruct/"
#"/work/home/cryoem666/xyf/temp/pycharm/model/Galactica_125M/" Base Model
#"/work/home/cryoem666/xyf/temp/pycharm/model/Galactica_1.3B/"
#"/work/home/cryoem666/xyf/temp/pycharm/model/Galactica/"
#"/work/home/cryoem666/xyf/temp/pycharm/model/llama3/"
# ./downstream/keywords_prediction/GO/OPI_CASPSimilarGO_Test_unique.json
# ./downstream/keywords_prediction/GO/OPI_IDFilterGO_Test_unique.json
# ./downstream/keywords_prediction/GO/OPI_UniProtGO_Test_unique.json
# ./downstream/Uniprot2024/uniprot2024_go_test_less2500.json