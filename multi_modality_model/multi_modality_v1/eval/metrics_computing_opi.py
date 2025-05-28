import json
import os
import tqdm
from rouge_score import rouge_scorer
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from sklearn.preprocessing import MultiLabelBinarizer
import nltk
import sys
import evaluate

def truncate_sequences(sequences, tokenizer, max_length=500): ##biobert only supports max token length 512
    truncated_sequences = []
    length_list = []
    for seq in sequences:
        tokenized = tokenizer.tokenize(seq)
        length_list.append(len(tokenized))
        truncated = tokenizer.convert_tokens_to_string(tokenized[:max_length])
        truncated_sequences.append(truncated)
    #print(length_list)
    return truncated_sequences


def calculate_metrics(output, target):
    # Convert to binary format
    mlb = MultiLabelBinarizer(classes=sorted(set(output + target)))
    y_true = mlb.fit_transform([target])
    y_pred = mlb.transform([output])

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='micro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='micro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    return accuracy, precision, recall, f1


def calculate_bleu(output, target):
    bleu = evaluate.load("./metrics/bleu")
    print('Load BLEU')
    print('Compute BleuScore...')
    bleu_results = bleu.compute(predictions=output, references=target)
    print(f"Bleu Result:{bleu_results}")
    return bleu_results['bleu']


def calculate_rouge_scores(output, target):
    rouge = evaluate.load("./metrics/rouge")
    print('Load ROUGE')
    print('Compute RougeScore...')
    rouge_results = rouge.compute(predictions=output, references=target)
    return rouge_results


def calculate_bertscore(output, target):
    from transformers import AutoTokenizer
    bertscore = evaluate.load("./metrics/bertscore")
    print('Load BERTSCORE')
    print('Compute BertScore...')
    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-large-cased-v1.1")
    output = truncate_sequences(output, tokenizer)
    target = truncate_sequences(target, tokenizer)
    bert_results = bertscore.compute(predictions=output, references=target,
                                  model_type="dmis-lab/biobert-large-cased-v1.1", num_layers=24)

    return {
        "precision": sum(bert_results["precision"]) / len(bert_results["precision"]),
        "recall": sum(bert_results["recall"]) / len(bert_results["recall"]),
        "f1": sum(bert_results["f1"]) / len(bert_results["f1"])
    }


def calculate_pub_bert_score(output, target):
    from transformers import AutoTokenizer
    bertscore = evaluate.load("./metrics/bertscore")
    print('Load BERTSCORE with PubMedBERT')

    tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")  # [2,3](@ref)

    output = truncate_sequences(output, tokenizer)
    target = truncate_sequences(target, tokenizer)


    bert_results = bertscore.compute(
        predictions=output,
        references=target,
        model_type="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",  # [3](@ref)
        num_layers=12,
        lang="en"
    )

    return {
        "precision": sum(bert_results["precision"]) / len(bert_results["precision"]),
        "recall": sum(bert_results["recall"]) / len(bert_results["recall"]),
        "f1": sum(bert_results["f1"]) / len(bert_results["f1"])
    }


def calculate_meteor(output, target):
    meteor = evaluate.load('./metrics/meteor')
    print('Load METEOR')
    print('Compute MeteorScore')
    meteor_results = meteor.compute(predictions=output, references=target)
    #print(meteor_results)
    return meteor_results['meteor']


def process_data(data, json_file_path):
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    total_scores = []

    for entry in tqdm.tqdm(data):
        output = entry.get("generated", entry.get("predict", []))
        target = entry.get("ground_truth", [])

        # Ensure both output and target are lists
        if isinstance(output, str):
            if any(keyword.lower() in json_file_path.lower() for keyword in
                   ['ec_number', 'go', 'keywords']):
                output = [item.lower().strip() for item in output.strip('.').split(';')]
            elif any(keyword in json_file_path.lower() for keyword in ['function', 'localization']):
                output = [output.split('\n')[0].lower().strip('.')]
        if isinstance(target, str):
            if any(keyword.lower() in json_file_path.lower() for keyword in
                   ['ec_number', 'go', 'keywords']):
                target = [item.lower().strip() for item in target.split(';')]
            elif any(keyword in json_file_path.lower() for keyword in ['function', 'localization']):
                target = [target.split('\n')[0].lower().strip('.')]

        # Compute metrics based on the file path
        if 'function' in json_file_path.lower():
            continue
        elif 'localization' in json_file_path.lower():
            # Calculate accuracy
            accuracy, _, _, _ = calculate_metrics(output, target)
            accuracies.append(accuracy)
        elif any(keyword.lower() in json_file_path.lower() for keyword in ['ec_number', 'go', 'keywords']):
            # Calculate precision, recall, F1
            _, precision, recall, f1 = calculate_metrics(output, target)
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
            total_scores.append((precision, recall, f1))
    # Aggregate results
    results = {}
    if 'function' in json_file_path.lower():
        generated_list = [item['generated'] for item in data]
        ground_truth_list = [item['ground_truth'] for item in data]
        rouge_score = calculate_rouge_scores(generated_list, ground_truth_list)
        bleu_score = calculate_bleu(generated_list, ground_truth_list)
        bert_score = calculate_bertscore(generated_list, ground_truth_list)
        meteor_score = calculate_meteor(generated_list, ground_truth_list)
        #pub_bertscore = calculate_pub_bert_score(generated_list, ground_truth_list)
        pub_bertscore = None


        results.update({
            'ROUGEScore': {'rouge1':round(rouge_score['rouge1'], 4),
                           'rouge2':round(rouge_score['rouge2'],4),
                           'rougel':round(rouge_score['rougeL'],4),
                           'rougeLsum':round(rouge_score['rougeLsum'],4)} if rouge_score!=None else None,
            'BLEU': round(bleu_score, 4) if bleu_score!=None else None,
            'BERTScore': {
                'precision': round(bert_score['precision'], 4),
                'recall': round(bert_score['recall'], 4),
                'f1': round(bert_score['f1'] , 4),
            } if bert_score!=None else None,
            'PubBertScore':{
                'precision': round(pub_bertscore['precision'], 4),
                'recall': round(pub_bertscore['recall'], 4),
                'f1': round(pub_bertscore['f1'] , 4),
            } if pub_bertscore!=None else None,
            'METEOR': round(meteor_score, 4) if meteor_score!=None else None,
        })
    if accuracies:
        results['Accuracy'] = round(sum(accuracies) / len(accuracies), 4)
    if precisions or recalls or f1_scores:
        results.update({
            'Precision': round(sum(precisions) / len(precisions), 4) if precisions else None,
            'Recall': round(sum(recalls) / len(recalls), 4) if recalls else None,
            'F1 Score': round(sum(f1_scores) / len(f1_scores), 4) if f1_scores else None,
        })

    return results


def return_opi_metrics(original_result, file_path, input_model=None):
    deeploc_label = {
        0: 'Cell.membrane',  # 细胞膜
        1: 'Cytoplasm',  # 细胞质
        2: 'Endoplasmic.reticulum',  # 内质网
        3: 'Golgi.apparatus',  # 高尔基体
        4: 'Lysosome/Vacuole',  # 溶酶体/液泡
        5: 'Mitochondrion',  # 线粒体
        6: 'Nucleus',  # 细胞核
        7: 'Peroxisome',  # 过氧化物酶体
        8: 'Plastid',  # 质体
        9: 'Extracellular'  # 细胞外
    }
    opi_label = {
        0: 'membrane',
        1: 'Cytoplasm',
        2: 'reticulum',
        3: 'apparatus',
        4: 'Lysosome/Vacuole',
        5: 'Mitochondrion',
        6: 'Nucleus',
        7: 'Peroxisome',
        8: 'Plastid',
        9: 'Extracellular'
    }
    instruct_protein_dict = {0: "plasma membrane",
                             1: "cytoplasm",
                             2: "endoplasmic reticulum",
                             3: "golgi",
                             4: "vacuole",
                             5: "mitochondrion",
                             6: "nucleus",
                             7: "peroxisome",
                             8: "chloroplast",
                             9: "extracellular"}
    instructprotein2opi = dict(zip(instruct_protein_dict.values(), opi_label.values()))
    if input_model == 'InstructProtein':
        original_result = [{'generated': instructprotein2opi.get(item['generated'], item['generated']),
                            'ground_truth': item['ground_truth']} for item in original_result]
    deeploc2opi = dict(zip(deeploc_label.values(), opi_label.values()))
    if set(deeploc_label.values()) == set([item['ground_truth'] for item in original_result]):
        original_result = [{'generated': item['generated'],
                            'ground_truth': deeploc2opi[item['ground_truth']]} for item in original_result]

    metrics = process_data(original_result, file_path)
    print(metrics)



