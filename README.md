# Opus-PLLM: Advancing Generative Large Language Models Toward Discriminative Performance in Protein Function Prediction

Here is the official codebase for OPUS-PLLM: Advancing Generative Large Language Models Toward Discriminative Performance in Protein Function Prediction.



## Dependencies

1.First, Create a new virtual python3.10 enviroment and activate it. We recommend you to deploy this project in cuda11.8.

```shell
conda create -n OpusPLLM python=3.10
conda activate OpusPLLM
```

2.Install most packages required through requirements.txt.

```shell
pip install -r requirements.txt
```

3.Install **PyTorch with cuda-11.8** using pip following the instructions in [link](https://pytorch.org/get-started/locally/). In this project, we employ **torch==2.4.0** and its corresponding dependencies so you can download it with command:

```shell
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
```

And download xformers:

```shell
pip install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu118
```

4.Detect whether your conda support CXXABI1.3.9 using command 

```shell
strings $CONDA_PREFIX/lib/libstdc++.so.6 | grep CXXABI
```

If CXXABI_1.3.9 is not returned, using command following to download libstdcxx-ng to support  **bitsandbytes**

```shell
conda install -c conda-forge libstdcxx-ng
```

5.Finally, setup the project using:

```shell
export PYTHONPATH=/path/to/OPUS_PLLM/
```



## Dataset and Model Weight

As mentioned in the article, we have open-sourced two versions of the dataset for training:  OPUS-InstructionCorpus and OPUS-InstructionCorpus-Evol.

Both two versions of training data are available for download through our Huggingface training dataset repos [OPUS-InstructionCorpus](https://huggingface.co/datasets/YifanXu24/OPUS-InstructionCorpus) 👈🤗 and [OPUS-InstructionCorpus-Evol](https://huggingface.co/datasets/YifanXu24/OPUS-InstructionCorpus-Evol) 👈🤗.  

Also, 17 test datasets of our benchmark are all open-sourced [here](https://huggingface.co/YifanXu24/OPUS-InstructionCorpus-Benchmark) 👈.  We gratefully acknowledge the teams that contributed some parts of the original test sets(OPI-team, Clean-team, Deeploc-team).

## Steps to Reproduce the Inference Result 

- At first, please ensure you have prepared all the prerequisite environments as specified in the **Dependencies** section. If not, please follow the instructions step by step in the **Dependencies** section.
- Download the corresponding test sets from our 🤗  [Hugging Face Repo](https://huggingface.co/YifanXu24/OPUS-InstructionCorpus-Benchmark)  and maintain their original name. The naming of the test dataset will affect how the automatic metrics are calculated. Specifically, test sets containing **"GO"** in their names will be processed through a dedicated pipeline that calculates **precision**, **recall**, and **F1 score** between the generated text and ground truth. The same applies to the others.

If you wish to experience OPUS-PLLM's efficient and accurate protein annotation capabilities, or to evaluate its performance on our provided functional annotation benchmark, please follow these steps:

- Download the Base Model Weights and its corresponding Lora weights, Switch Projector Weights and Cross-Modal Adapter Weights. Base Model Weights can be downloaded from their official 🤗 Hugging Face Repo. Others can be downloaded on our 🤗 Hugging Face Repo. We have provided the corresponding repository URLs in [Model Zoo](#model-zoo)  🦒🏛️Section, please select the appropriate pairing scheme.
- For example, if you choose to validate annotation model with llama3-8B🐪, you can download llama3-8B-Base from [here](https://huggingface.co/meta-llama/Meta-Llama-3-8B) 👈. Then, download its corresponding Lora weights, Switch Projector Weights and Cross-Modal Adapter Weights from [here](https://huggingface.co/YifanXu24/OPUS-PLLM-Llama3-8B-Base) 👈.
- Navigate to the inference path: `OPUS-PLLM/multi_modality_model/multi_modality_v1/eval/`, and you can reproduce results by running:



#### For Batch Annotation:

```shell
accelerate launch  run_opus_ddp.py  \
--model-base-path /path/to/model/ \
--adapter-path /path/to/adapter \
--is_json True \
--json_path /path/to/file \
--save_path /path/to/save \
--load-4bit True \
--cstp_path /path/to/cstp
```

- In this case, **model-base-path** represents the path to the llama3-8B-Base, **adapter-path** represents the path to the corresponding Lora weights and Switch Projector Weights, **json_path** represents the path to a certain test set, **save_path** represents the path to the saving file including inferencing results and ground truth and cstp_path represents the path to the Cross-Modal Adapter.

#### For Online Inference:

```shell
python run_opus_online.py  \
--model-base-path /path/to/model/  \
--adapter-path /path/to/adapter \
--load-4bit True \
--cstp_path /path/to/cstp
```

- In this case, you will use OPUS-PLLM for protein function annotation in an interactive, single-turn dialogue-like manner directly in the terminal. Specifically, terminal will hint you **"Enter your instruction:"** that requires one instruction and **"Enter the protein sequence (or leave empty to skip):"** that requires one protein sequence as input. For example, we can use case for instruction like **"Given a protein sequence, predict the corresponding Gene Ontology term that describes its molecular function, biological process, and cellular component."** and protein sequence like "MPYFAQRLYNTCKASFSSDGPITEDALEKVRNVLEKIKPSDVGIEQDAQLARSRSGPLNERNGSNQSPPAIKYLHLHECDSFSIGIFCMPPSSMIPLHNHPGMTVLSKLVYGSMHVKSYDWLEPQLTEPEDPSQARPAKLVKDTEMTAQSPVTTLYPKSGGNIHCFKAITHCAILDILAPPYSSEHDRHCTYFRKSRREDLPGELEVDGEVVTDVTWLEEFQPPDDFVIRRIPYRGPVIRT". The model will output the result with **"cytosol; nucleus; cysteine dioxygenase activity; iron ion binding; cellular response to hypoxia; detection of hypoxia; response to hypoxia"**.

If you want to experience the OPUS-PLLM-Evol's ability to solve protein-sequence-centered interactive capabilities in conversational mode, or verify its performance on our provided MCQ benchmark, please first select the model in **[Model Zoo](#model-zoo🦒🏛️)** and download the corresponding module weights using the same method described above.

- For example, if you choose to experience the OPUS-PLLM-Evol with llama3-8B-Instruct🐪, you can download llama3-8B-Instruct from [here](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) 👈. Then, download its corresponding Lora weights, Switch Projector Weights and Cross-Modal Adapter Weights from [here](https://huggingface.co/YifanXu24/OPUS-PLLM-Llama3-8B-Evol) 👈.
- Navigate to the inference path: `OPUS-PLLM/multi_modality_model/multi_modality_v1/eval/`, and you can reproduce results by running:

#### For Batch MCQ Inference:

```shell
accelerate launch  eval_run_multichoice.py  \
--model-base-path /path/to/model/ \
--adapter-path /path/to/adapter \
--json_path /path/to/file \
--save_path /path/to/save \
--load-4bit True \
--cstp_path /path/to/cstp
```

- In this case, **model-base-path** represents the path to the llama3-8B-Instruct, **adapter-path** represents the path to the corresponding Lora weights and Switch Projector Weights, **json_path** represents the path to MCQ benchmark, **save_path** represents the path to the saving file including inferencing results for MCQ questions and correct options and cstp_path represents the path to the Cross-Modal Adapter.

#### For Online Inference：

```shell
python eval_run_online.py  \
--model-base-path /path/to/model/ \
--adapter-path /path/to/adapter \
--load-4bit True \
--cstp_path /path/to/cstp
```

- Similar to Online Inference script for annotation model, you also need to enter any instruction and protein sequence you want in the terminal and the model will response with diverse and professional response for you. 



## Model Zoo🦒🏛️

Here is the list of different module versions of OPUS-PLLM. The deployment of OPUS-PLLM requires weight files from four components: **Base Model Weight**, **corresponding LoRA weights**, **Switch Projector Weights**, and **Cross-Modal Adapter Weights**. Among these, the **Cross-Modal Adapter Weights** are universal and can be downloaded [here](https://huggingface.co/YifanXu24/OPUS-PLLM-CSTP) 👈, while the other three must be mutually compatible.

We provide four sets of model weights:

- **Llama3-8B-Base**, **Galactica-6.7B**, and **Galactica-1.3B** serve as the foundation for **OPUS-PLLM**, specializing in **annotation tasks**.
- **Llama3-8B-Instruct** acts as the base for **OPUS-PLLM-Evol**, designed for **diverse and complex daily interactions** with enhanced **instruction-following capabilities**.

|        Model name        | Model_Type |                          Base Model                          |                  LoRA and Switch Projector                   |
| :----------------------: | :--------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| OPUS-PLLM-Llama3-8B-Base |    Base    | [Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)🐪 | [Link🤗](https://huggingface.co/YifanXu24/OPUS-PLLM-Llama3-8B-Base) |
| OPUS-PLLM-Galactica-1.3B |    Base    | [Galactica-1.3B](https://huggingface.co/facebook/galactica-1.3b)🌌 | [Link🤗](https://huggingface.co/YifanXu24/OPUS-PLLM-Galactica-1.3B-Base) |
| OPUS-PLLM-Galactica-6.7B |    Base    | [Galactica-6.7B](https://huggingface.co/facebook/galactica-6.7b)🌌 | [Link🤗](https://huggingface.co/YifanXu24/OPUS-PLLM-Galactica-6.7B-Base) |
| OPUS-PLLM-Llama3-8B-Evol |    Evol    | [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)🐪 | [Link🤗](https://huggingface.co/YifanXu24/OPUS-PLLM-Llama3-8B-Evol) |



## Acknowledgements





## Citing OPUS-PLLM
