import pickle
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ast
class UniProtQADataset(Dataset):
    def __init__(self, dsets_path):
        with open(dsets_path, 'rb') as file:
            self.data = pickle.load(file)
        self.seq_text_pairs = self.update_dict(self.data)

    def check_and_update_dict(self, my_dict):
        required_keys = ['Description', 'Accession', 'Name', 'Similarity', 'Sequence', 'Subcellular_Location']
        for key in required_keys:
            if key not in my_dict:
                my_dict[key] = 'None'
        return my_dict

    def update_dict(self, data):
        seq_text_pairs = []
        i = 0
        for key, value in data.items():
            if(len(value['Sequence'])>2500):
                continue
            updated_value = self.check_and_update_dict(value)
            #if (len(updated_value['Name'])>1):
            #    print(type(updated_value['Sequence']),updated_value['Sequence'])
            #    print('Name', type(updated_value['Name']),updated_value['Name'])
            #    print('Accession',type(updated_value['Accession']),updated_value['Accession'])
            #    print('Similarity',type(updated_value['Similarity']),updated_value['Similarity'])
            #    print('Subcellular_Location',type(updated_value['Subcellular_Location']),updated_value['Subcellular_Location'])
            name_str = ", ".join(updated_value['Name'])
            accession_str = ", ".join(updated_value['Accession'])
            seq_text_pairs.append({
                'sequence': updated_value['Sequence'],
                'text': f'''The name of protein is {name_str} '''.replace('.', '')+'. '
                       +f'''Accession: {accession_str} '''.replace('.', '')+'. '
                       +f'''Similarity: {updated_value['Similarity']} '''.replace('.', '')+'. '
                       +f'''Subcellular_Location: {updated_value['Subcellular_Location']}'''.replace('.', '')+'. '
            })
            
            #if(len(updated_value['Sequence'])>2500):
            #    i = i+1
            #    print(i, len(updated_value['Sequence']))
        #for x in seq_text_pairs:
        #    if len(x['sequence'])>2500:
        #        print(len(x['sequence']))
        #print(len(data))
        #print(len(seq_text_pairs))
        #print(len(data)-len(seq_text_pairs))
        return seq_text_pairs

    def __len__(self):
        return len(self.seq_text_pairs)

    def __getitem__(self, idx):
        return self.seq_text_pairs[idx]

class BinaryLocalizationDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        self.binaryLocalizationData = data
    def __len__(self):
        return len(self.binaryLocalizationData)

    def __getitem__(self, idx):
        return self.binaryLocalizationData[idx]
    
class SubcellularlizationDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        self.subcellularlizationData = data
    def __len__(self):
        return len(self.subcellularlizationData)

    def __getitem__(self, idx):
        return self.subcellularlizationData[idx]

class ECDataset(Dataset):
    def __init__(self, file_path, split_key):
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        self.ECData = data[split_key]
    def __len__(self):
        return len(self.ECData)

    def __getitem__(self, idx):
        dict_EC = self.ECData[idx]
        return dict_EC['seq'], dict_EC['seq_embedding'],dict_EC['label']

class GoDataset(Dataset):
    def __init__(self, file_path, split_key):
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        self.ECData = data[split_key]
    def __len__(self):
        return len(self.ECData)

    def __getitem__(self, idx):
        dict_EC = self.ECData[idx]
        return dict_EC['seq'], dict_EC['seq_embedding'],dict_EC['label']

class UniProtQAVecDataset(Dataset):
    def __init__(self, dsets_path):
        with open(dsets_path, 'rb') as file:
            self.seq_text_pairs_vec = pickle.load(file)
    def __len__(self):
        return len(self.seq_text_pairs_vec)

    def __getitem__(self, idx):
        return self.seq_text_pairs_vec[idx]

class AAVDatasetMaxMin(Dataset):
    def __init__(self, csv_path, global_min_target,global_max_target, split='train'):
        """
        csv_path: CSV文件路径
        split: 指定加载的数据集分割，可以是'train'、'test'或'validation'
        """
        # 读取CSV文件
        full_data = pd.read_csv(csv_path)
        self.global_min_target = global_min_target
        self.global_max_target = global_max_target
        if split in ['train', 'test']:
            # 对于训练集和测试集，排除validation为True的数据
            self.dataframe = full_data[(full_data['set'] == split) & (full_data['validation'] != True)]
        elif split == 'validation':
            # 对于验证集，选取set为train且validation为True的数据
            self.dataframe = full_data[(full_data['set'] == 'train') & (full_data['validation'] == True)]
        else:
            raise ValueError("Invalid split name. Expected 'train', 'test', or 'validation'.")
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        """
        返回指定索引处的数据项，包括蛋白质序列（sequence）和目标值（target）
        """
        data_row = self.dataframe.iloc[idx]
        sequence = data_row['sequence']
        target = data_row['target']
        max_min_target = (target - self.global_min_target) / (self.global_max_target - self.global_min_target)
        return sequence, target, max_min_target

class AAVDataset(Dataset):
    def __init__(self, csv_path, split='train',label = 'target'):
        """
        csv_path: CSV文件路径
        split: 指定加载的数据集分割，可以是'train'、'test'或'validation'
        """
        # 读取CSV文件
        full_data = pd.read_csv(csv_path)
        self.label = label
        
        if split in ['train', 'test']:
            # 对于训练集和测试集，排除validation为True的数据
            self.dataframe = full_data[(full_data['set'] == split) & (full_data['validation'] != True)]
        elif split == 'validation':
            # 对于验证集，选取set为train且validation为True的数据
            self.dataframe = full_data[(full_data['set'] == 'train') & (full_data['validation'] == True)]
        else:
            raise ValueError("Invalid split name. Expected 'train', 'test', or 'validation'.")
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        """
        返回指定索引处的数据项，包括蛋白质序列（sequence）和目标值（target）
        """
        data_row = self.dataframe.iloc[idx]
        sequence = data_row['sequence']
        #seq_embedding = data_row['seq_embedding']
        seq_embedding = torch.tensor(ast.literal_eval(data_row['seq_embedding']))
        target = data_row[self.label]
        return sequence, seq_embedding, target
    
class Beta_LacDataset(Dataset):
    def __init__(self, csv_path, split='train',label = 'scaled_effect1'):
        """
        csv_path: CSV文件路径
        split: 指定加载的数据集分割，可以是'train'、'test'或'validation'
        """
        # 读取CSV文件
        full_data = pd.read_csv(csv_path)
        self.label = label
        
        if split in ['train', 'test']:
            # 对于训练集和测试集，排除validation为True的数据
            self.dataframe = full_data[(full_data['set'] == split) & (full_data['validation'] != True)]
        elif split == 'validation':
            # 对于验证集，选取set为train且validation为True的数据
            self.dataframe = full_data[(full_data['set'] == 'train') & (full_data['validation'] == True)]
        else:
            raise ValueError("Invalid split name. Expected 'train', 'test', or 'validation'.")
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        """
        返回指定索引处的数据项，包括蛋白质序列（sequence）和目标值（target）
        """
        data_row = self.dataframe.iloc[idx]
        sequence = data_row['sequence']
        #seq_embedding = data_row['seq_embedding']
        seq_embedding = torch.tensor(ast.literal_eval(data_row['seq_embedding']))
        target = data_row[self.label]
        #target = data_row['z_score_target']
        #target = data_row['min_max_target']
        return sequence, seq_embedding, target

class FluoreDataset(Dataset):
    def __init__(self, csv_path, split='train',label = 'log_fluorescence'):
        """
        csv_path: CSV文件路径
        split: 指定加载的数据集分割，可以是'train'、'test'或'validation'
        """
        # 读取CSV文件
        full_data = pd.read_csv(csv_path)
        self.label = label
        if split in ['train', 'test']:
            # 对于训练集和测试集，排除validation为True的数据
            self.dataframe = full_data[(full_data['set'] == split) & (full_data['validation'] != True)]
        elif split == 'validation':
            # 对于验证集，选取set为train且validation为True的数据
            self.dataframe = full_data[(full_data['set'] == 'train') & (full_data['validation'] == True)]
        else:
            raise ValueError("Invalid split name. Expected 'train', 'test', or 'validation'.")
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        """
        返回指定索引处的数据项，包括蛋白质序列（sequence）和目标值（target）
        """
        data_row = self.dataframe.iloc[idx]
        sequence = data_row['sequence']
       # seq_embedding = data_row['seq_embedding']
        seq_embedding = torch.tensor(ast.literal_eval(data_row['seq_embedding']))
        target = data_row[self.label]
        return sequence, seq_embedding, target

class StabilityDataset(Dataset):
    def __init__(self, csv_path, split='train', label = 'stability_score'):
        """
        csv_path: CSV文件路径
        split: 指定加载的数据集分割，可以是'train'、'test'或'validation'
        """
        # 读取CSV文件
        print(csv_path)
        full_data = pd.read_csv(csv_path)
        self.label = label
        if split in ['train', 'test']:
            # 对于训练集和测试集，排除validation为True的数据
            self.dataframe = full_data[(full_data['set'] == split) & (full_data['validation'] != True)]
        elif split == 'validation':
            # 对于验证集，选取set为train且validation为True的数据
            self.dataframe = full_data[(full_data['set'] == 'train') & (full_data['validation'] == True)]
        else:
            raise ValueError("Invalid split name. Expected 'train', 'test', or 'validation'.")
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        """
        返回指定索引处的数据项，包括蛋白质序列（sequence）和目标值（target）
        """
        data_row = self.dataframe.iloc[idx]
        sequence = data_row['sequence']
        #seq_embedding = data_row['seq_embedding']
        seq_embedding = torch.tensor(ast.literal_eval(data_row['seq_embedding']))
        #target = data_row['stability_score']
        target = data_row[self.label]
        return sequence, seq_embedding, target

class ThermoDataset(Dataset):
    def __init__(self, csv_path, split='train',label = 'z_score_target'):
        """
        csv_path: CSV文件路径
        split: 指定加载的数据集分割，可以是'train'、'test'或'validation'
        """
        self.label = label
        # 读取CSV文件
        full_data = pd.read_csv(csv_path)
        full_data = full_data[full_data['sequence'].str.len() < 3000]
        for seq in full_data['sequence']:
            if len(seq)>3000:
               print(len(seq))
        if split in ['train', 'test']:
            # 对于训练集和测试集，排除validation为True的数据
            self.dataframe = full_data[(full_data['set'] == split) & (full_data['validation'] != True)]
        elif split == 'validation':
            # 对于验证集，选取set为train且validation为True的数据
            self.dataframe = full_data[(full_data['set'] == 'train') & (full_data['validation'] == True)]
        else:
            raise ValueError("Invalid split name. Expected 'train', 'test', or 'validation'.")
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        """
        返回指定索引处的数据项，包括蛋白质序列（sequence）和目标值（target）
        """
        data_row = self.dataframe.iloc[idx]
        sequence = data_row['sequence']
        #seq_embedding = data_row['seq_embedding']
        seq_embedding = torch.tensor(ast.literal_eval(data_row['seq_embedding']))
        #target = data_row['target']
       #target = data_row['min_max_target']
       # target = data_row['z_score_target']
        target = data_row[self.label]
        return sequence, seq_embedding, target
    



def show_data(protein_dataset):

    sequence_lengths = []
    for x in protein_dataset:
        sequence_lengths.append(len(x['sequence']))

    # 设置区间和区间宽度
    bin_width = 50
    bins = np.arange(0, max(sequence_lengths) + bin_width, bin_width)

    # 统计每个区间中的序列数量
    hist, _ = np.histogram(sequence_lengths, bins=bins)

    # 绘制长度分布的直方图
    plt.bar(bins[:-1], hist, width=bin_width, color='blue', edgecolor='black')
    plt.xlabel('Sequence Length')
    plt.ylabel('Frequency')
    plt.title('Protein Sequence Length Distribution after Filtering')
    plt.savefig('Protein_Sequence_Length_Distribution_after_Filtering.png')
    plt.show()

if __name__ == '__main__':
    
   # dsets_path = "/mnt/petrelfs/lvying/LLM/OPUS-BioLLM/dataset/clip_data_embedding/clip_seq_text_embedding/uniprot_st_pairs_embedding_part_1.pickle"
   # uniport_dataset = UniProtQAVecDataset(dsets_path)   
   # with open(dsets_path, 'rb') as file:
   #     data = pickle.load(file)
   # print(data)
   # for i, item in enumerate(uniport_dataset):
   #     print(f"Protein {i}: {item}")
   #     if i >= 50:  
   #         break
    #dsets_path = "/mnt/petrelfs/lvying/LLM/OPUS-BioLLM/dataset/binary_localization/gt_subcellular_localization.pkl"
   # dict_path = "/mnt/petrelfs/lvying/LLM/OPUS-BioLLM/dataset/subcellular_localization/gt_subcellular_localization_dict.pkl"
   # dict_path = "/mnt/petrelfs/lvying/LLM/OPUS-BioLLM/dataset/binary_localization/gt_binary_localization_dict.pkl"

    #df = pd.read_pickle(dsets_path)
    #data_dict = df.to_dict(orient='records')
    #with open(dict_path, 'wb') as file:
    #     pickle.dump(data_dict, file)
    #uniport_dataset = SubcellularlizationDataset(dict_path)    
    #for i, item in enumerate(uniport_dataset):
    #    print(f"Protein {i}: {item}")
    #    if i >= 50:  
    #        break
    #show_data(uniport_dataset)
    #data_loader = DataLoader(uniport_dataset, batch_size=32, shuffle=True)
    #for batch in data_loader:
    #    for entry in batch:
    #        protein_name, sequence = entry
    #        sequence_length = len(sequence)
    #        if(sequence_length>2500):
    #            print(f"{protein_name}: {sequence_length}")
    dpath = '/mnt/petrelfs/lvying/LLM/OPUS-BioLLM/dataset/fitness_landscape_predicetion/thermostability/human_cell.csv'
    train_dataset = ThermoDataset(dpath, 'train')
    test_dataset = ThermoDataset(dpath, 'test')
    validation_dataset = ThermoDataset(dpath, 'validation')
    print("train",len(train_dataset))
    print("test",len(test_dataset))
    print("validation",len(validation_dataset))
    print(train_dataset[0])
    