import json

def calculate_precision_recall(generated, ground_truth):
    generated_labels = set([word.lower().strip().replace('.','') for word in generated.split(';')])
    ground_truth_labels = set([word.lower().strip().replace('.','') for word in ground_truth.split(';')])
    exactly_match = 0
    sub_match = 0
    tp = len(generated_labels & ground_truth_labels)
    fp = len(generated_labels - ground_truth_labels)
    fn = len(ground_truth_labels - generated_labels)
    if fp == 0:
        if fn == 0:
            exactly_match = 1
        sub_match = 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall/(precision + recall) if (precision + recall)>0 else 0

    return precision, recall, f1, tp, fp, fn, exactly_match, sub_match

def generate_label_prob(result):
    label_set = list(set([item['ground_truth'] for item in result]))
    label_acc = dict(zip(label_set,[0]*len(label_set)))
    label_count = dict(zip(label_set,[0]*len(label_set)))
    for item in result:
        if item['ground_truth'] in label_count.keys():
            label_count[item['ground_truth']]+=1
        else:
            label_count[item['ground_truth']]=1
        #if item['ground_truth'].lower().strip('.') in item['generated'].lower().strip('.') or item['generated'].lower().strip('.') in item['ground_truth'].lower().strip('.'):
        if item['ground_truth'].lower().strip('.') == item['generated'].lower().strip('.'):
            if item['ground_truth'] in label_acc.keys():
                label_acc[item['ground_truth']]+=1
            else:
                label_acc[item['ground_truth']]=1
    label_acc = {label:value/label_count[label] for label, value in label_acc.items()}
    return label_acc, label_count, {'accuracy':len([item for item in result if item['ground_truth'].lower().strip('.') == item['generated'].lower().strip('.')])/len(result)}

def return_our_metrics(original_result, json_path, input_model=None):
    if 'go' in json_path.lower() or 'ec_number' in json_path.lower() or 'keywords' in json_path.lower():
        results = [calculate_precision_recall(item['generated'], item['ground_truth']) for item in original_result]
        precisions = [result[0] for result in results]
        recalls = [result[1] for result in results]
        f1 = [result[2] for result in results]
        exactly_match = sum([result[6] for result in results]) / len(results)
        sub_match = sum([result[7] for result in results]) / len(results)

        # 宏平均
        macro_precision = sum(precisions) / len(precisions)
        macro_recall = sum(recalls) / len(recalls)
        macro_f1_score = sum(f1)/len(f1)
        total_tp = sum(result[3] for result in results)
        total_fp = sum(result[4] for result in results)
        total_fn = sum(result[5] for result in results)
        # 微平均
        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        micro_f1_score = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        print(f"Micro Precision: {micro_precision}")
        print(f"Micro Recall: {micro_recall}")
        print(f"Micro F1 Score: {micro_f1_score}")
        print(f"Macro Precision: {macro_precision}")
        print(f"Macro Recall: {macro_recall}")
        print(f"Macro F1 Score: {macro_f1_score}")
        print(f"Exact Match: {exactly_match}")
        print(f"Sub Match: {sub_match}")
    elif 'location' in json_path or 'localization' in json_path:
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
            0: 'membrane',  # 细胞膜
            1: 'Cytoplasm',  # 细胞质
            2: 'reticulum',  # 内质网
            3: 'apparatus',  # 高尔基体
            4: 'Lysosome/Vacuole',  # 溶酶体/液泡
            5: 'Mitochondrion',  # 线粒体
            6: 'Nucleus',  # 细胞核
            7: 'Peroxisome',  # 过氧化物酶体
            8: 'Plastid',  # 质体
            9: 'Extracellular'  # 细胞外
        }
        instruct_protein_dict = {0:"plasma membrane",
                                               1:"cytoplasm",
                                               2:"endoplasmic reticulum",
                                               3:"golgi",
                                               4:"vacuole",
                                               5:"mitochondrion",
                                               6:"nucleus",
                                               7:"peroxisome",
                                               8:"chloroplast",
                                               9:"extracellular"}
        instructprotein2opi = dict(zip(instruct_protein_dict.values(), opi_label.values()))
        if input_model == 'InstructProtein':
            original_result = [{'generated': instructprotein2opi[
                item['generated']] if item['generated'] in instructprotein2opi.keys() else item['generated'],
                                'ground_truth': item['ground_truth']} for item in
                               original_result]  # 测试的是InstructProtein
        deeploc2opi = dict(zip(deeploc_label.values(), opi_label.values()))
        if set(deeploc_label.values())==set([item['ground_truth'] for item in original_result]): #测试的是Deeploc Test
            original_result = [{'generated': item['generated'],
                                 'ground_truth': deeploc2opi[
                                     item['ground_truth']]} for item in original_result]
        print(generate_label_prob(original_result))