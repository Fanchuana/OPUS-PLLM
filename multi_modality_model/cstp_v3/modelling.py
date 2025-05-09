import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import torch
import torch.nn as nn
import torch.nn.functional as F
import esm
import time
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
)
import pytorch_lightning as pl
from multi_modality_model.cstp_v3.evidence_loss import ce_loss
from torch.amp import autocast

#Contrastive Protein Sequence-Text Pretraining, Mapping Protein Sequence to text space
class ProteinSeqEmbeddingExtractor():
    def __init__(self, ckpt=None):
        # Load ESM-2 model
        self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        if ckpt!=None:
            # 加载 checkpoint
            print(f"======checkpoint {ckpt}被调用=====")
            checkpoint = torch.load(ckpt, map_location="cpu")['model']
            all_params = list(checkpoint.keys())
            # 筛选出属于 ESM2 模型的参数（即以 'protein_model.model.' 开头的参数）
            esm2_params = {param.replace('protein_model.model.', ''): checkpoint[param] for param in all_params if
                           param.startswith('protein_model.model.')}
            self.model.load_state_dict(esm2_params, strict=False)
        else:
            print("======ESM2被调用=====")
            self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model.eval().cuda() # Disable dropout
        #self.model = torch.nn.DataParallel(self.model)
    def get_protein_seq_embeddings(self, data):
        # Convert data to model input format
        data = [(f"protein{i}", seq) for i, seq in enumerate(data)]
        #for entry in data:
            #protein_name, sequence = entry
            #sequence_length = len(sequence)
            #print(f"{protein_name}: {sequence_length}")
        _, _, batch_tokens = self.batch_converter(data)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
        # Extract per-residue representations
        with torch.no_grad():
            results = self.model(batch_tokens.cuda(), repr_layers=[33], return_contacts=False)
        #token_representations = results["representations"][33]

        # Generate per-sequence representations via averaging
        sequence_representations = []
        for i, tokens_len in enumerate(batch_lens):
            sequence_representations.append(results["representations"][33][i, 1: tokens_len - 1].mean(0))
        sequence_representations = torch.stack(sequence_representations).float()
        torch.cuda.empty_cache()
        return sequence_representations



    def get_amino_acid_embeddings(self, data):
        # Convert data to model input format
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)

        # Extract per-residue representations
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33]

        # Extract per-amino acid representations
        amino_acid_representations = []
        for i, tokens_len in enumerate(batch_lens):
            # Extract representations for each amino acid, excluding start/end tokens
            amino_acid_representations.append(token_representations[i, 1: tokens_len - 1])
        
        return amino_acid_representations
              
class TextEmbeddingExtractor():
    def __init__(self, model_path, config_kwargs, tokenizer_max_length=4096):
        self.model_path = model_path
        self.tokenizer_max_length = tokenizer_max_length

        # Configure and load the model
        config = AutoConfig.from_pretrained(model_path, **config_kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            device_map = 'auto',
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
            revision='main',
           
        )
        self.model.eval()

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
    def get_text_embeddings(self, text):
         # Tokenize and encode the text
        tokens = self.tokenizer.batch_encode_plus(
            text,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=self.tokenizer_max_length,
            return_tensors='pt'
        )   
        input_ids = tokens["input_ids"].to('cuda')
        attention_mask = tokens['attention_mask'].to('cuda')

         # Obtain text embeddings
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.hidden_states
            last_hidden_states = hidden_states[-1]
            first_hidden_states = hidden_states[0]
            text_representations = torch.mean(first_hidden_states + last_hidden_states, dim=1)

            #embedding = list(outputs.hidden_states)
            #last_hidden_states = embedding[-1].cpu().numpy()
            #first_hidden_states = embedding[0].cpu().numpy()
            #last_hidden_states = np.squeeze(last_hidden_states)
            #first_hidden_states = np.squeeze(first_hidden_states)
            #text_representations = np.mean(first_hidden_states + last_hidden_states, axis=0)
            
        #text_representations = torch.Tensor(text_representations).to("cuda")
        return text_representations
 
 
 
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadCrossAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads

        self.w_qs = nn.Linear(d_model, d_model)
        self.w_kt = nn.Linear(d_model, d_model)
        self.w_vt = nn.Linear(d_model, d_model)

        self.fc = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        # Linear projections
        #q = self.w_qs(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        #k = self.w_kt(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        #v = self.w_vt(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        q = self.w_qs(q).view(batch_size, self.num_heads, self.d_k)
        k = self.w_kt(k).view(batch_size, self.num_heads, self.d_k)
        v = self.w_vt(v).view(batch_size, self.num_heads, self.d_k)
        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)

        # Apply attention to values
        #output = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        #output = self.fc(output)
        output = torch.matmul(attn, v)
        output = output.view(batch_size, -1)  # 将多头结果合并
        output = self.fc(output)
        return output

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadCrossAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Linear(d_model, d_model)

    def forward(self, seq, text):
        attn_output = self.mha(seq, text, text)
        out1 = self.norm1(seq + attn_output)
        ff_output = self.ff(out1)
        out2 = self.norm2(out1 + ff_output)
        return out2

class ProteinAdapterLayer_v2(nn.Module):
    def __init__(self, d_model, n_heads=4, n_layers=3):
        super(ProteinAdapterLayer_v2, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads) for _ in range(n_layers)])

    def forward(self, seq, text):
        for layer in self.layers:
            seq = layer(seq, text)
        #print("seq",seq.shape)
        return seq

class ProteinProjectionLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ProteinProjectionLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
  #      self.linear_1 = nn.Linear(output_dim, output_dim)
        
        #self.relu = nn.ReLU()
        #self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = self.linear(x)
 #       x = self.linear_1(x)
        #x = self.relu(x)
        #x = self.dropout(x)
        return x
    
class TextProjectionLayer_Temp(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TextProjectionLayer_Temp, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
 #       self.linear_1 = nn.Linear(output_dim, output_dim)

    
    def forward(self, x):
        x = self.linear(x)
   #     x = self.linear_1(x)
      
        return x

class TextProjectionLayer(nn.Module):
    def __init__(self, feature_dim, intermediate_dim, alpha=0.8):
        """
        This employs residual connections, utilizing two constant values, 
        α (alpha) and β (beta), as "residual ratios" to 
        help adjust the extent to which the original knowledge is retained
        Initialize TextProjectionLayer.
        Parameters:
        - feature_dim (int): Dimension of input and output features.
        - intermediate_dim (int): Dimension of the intermediate layer.
        - alpha (float): Proportional parameter α for the residual connection.
        
        """
        super(TextProjectionLayer, self).__init__()
        self.alpha = alpha
        self.linear1 = nn.Linear(feature_dim, intermediate_dim)
        self.linear2 = nn.Linear(intermediate_dim, feature_dim)

    def forward(self, f):
        """
        Forward propagation method.
        Parameters:
        - f (Tensor): Input feature, shape (batch_size, feature_dim).
        Returns:
        - Tensor: Output feature, shape (batch_size, feature_dim).
        """
        # Av(f) = ReLU(f^T Wv1) Wv2
        av_f = self.linear2(F.relu(self.linear1(f)))

        # f' = α Av(f)^T + (1 - α) f
        return self.alpha * av_f + (1 - self.alpha) * f

class ProteinAdapterLayer(nn.Module): #****
    # mapping protein sequence to text space
    def __init__(self, input_dim, output_dim):
        super(ProteinAdapterLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

    
class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.0007):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, preds, targets):
        """
        Calculate the InfoNCE loss.
        Parameters:
        - preds (dict): Predicted similarity score matrix.
        - targets (dict): Target indices.
        """
        logits = preds / self.temperature
        return F.cross_entropy(logits, targets)
    
class InfoNCELoss_h(nn.Module):
    def __init__(self, temperature=0.007):
        super(InfoNCELoss_h, self).__init__()
        self.temperature = temperature

    def forward(self, preds, targets):
        """
        Calculate the InfoNCE loss.
        Parameters:
        - preds (dict): Predicted similarity score matrix.
        - targets (dict): Target indices.
        """
        logits_q2t = preds['protein2text'] / self.temperature
        loss_q2t = F.cross_entropy(logits_q2t, targets['protein2text'])
        
        logits_t2q  = preds['text2protein'] / self.temperature
        loss_t2q = F.cross_entropy(logits_t2q, targets['text2protein'])
        loss_total = (loss_t2q+loss_q2t)/2
        return loss_total
    
class TrustEvidenceLoss(nn.Module):
    def __init__(self, temperature=0.0007):
        super(TrustEvidenceLoss, self).__init__()
        self.temperature = temperature

    def forward(self, preds, targets):
        """
        Calculate the InfoNCE loss.
        Parameters:
        - preds (dict): Predicted similarity score matrix.
        - targets (dict): Target indices.
        """
        logits_q2t = preds['protein2text'] / self.temperature
        
        loss_q2t = ce_loss(logits_q2t, targets['protein2text'], global_step=1000)
       # loss_q2t = F.cross_entropy(logits_q2t, targets['protein2text'])
        
        logits_t2q  = preds['text2protein'] / self.temperature
        loss_t2q = ce_loss(logits_t2q, targets['text2protein'], global_step=1000)
        #loss_t2q = F.cross_entropy(logits_t2q, targets['text2protein'])
        loss_total = (loss_t2q+loss_q2t)/2
        return loss_total


class CSTPBase(nn.Module):
    """
    Contrastive Protein Sequence-Text Pretraining
    Mapping Protein Sequence to text space
    """
    def __init__(self, 
                 protein_projection_input_dim, 
                 protein_projection_output_dim,
                 text_projection_input_dim, 
                 text_projection_output_dim,
                 n_heads, 
                 n_layers,
                 alpha=0.5):
        """
        Initialize the CSTP model.
        Parameters:
        - protein_input_embedding (int): Dimension of the input features for protein sequences.
        - text_input_dim (int): Dimension of the input features for text.
        - intermediate_dim (int): Dimension of the intermediate layer for text projection.
        - output_dim (int): Output dimension for the protein and text projection layers.
        - alpha (float): Residual connection proportion parameter for the TextProjectionLayer.
        """
        super(CSTPBase, self).__init__()

        # Protein sequence embedding extractor
        #self.protein_seq_embedder = ProteinSeqEmbeddingExtractor()

        # Text embedding extractor
        #self.text_embedder = TextEmbeddingExtractor(llama_model_path, llama_config_kwargs)  # Specify model_path and config_kwargs
        #for param in self.protein_seq_embedder.parameters():
        #    param.requires_grad = False
        #for param in self.text_embedder.parameters():
        #    param.requires_grad = False
        # Projection layers
        
        self.protein_projection = ProteinProjectionLayer(protein_projection_input_dim, protein_projection_output_dim)
        self.text_projection = TextProjectionLayer_Temp(text_projection_input_dim, text_projection_output_dim)
        #self.protein_adapter = ProteinAdapterLayer(protein_projection_output_dim,text_projection_output_dim)
        #self.protein_adapter = ProteinAdapterLayer_v2(protein_projection_output_dim, n_heads, n_layers)
        #self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.007))
        self.train_epoch_counter = 0
        self.train_epoch_last_time = time.time()
        
    def forward(self, protein_embeddings,text_embeddings):
        # Extract embeddings
        #protein_embeddings = self.protein_seq_embedder.get_protrein_seq_embeddings(protein_seq)
        #text_embeddings = self.text_embedder.get_text_embeddings(text)

        # Apply projection layers
        protein_embeddings = F.normalize(protein_embeddings, dim=-1)
        text_embeddings = F.normalize(text_embeddings, dim=-1)
        projected_protein = self.protein_projection(protein_embeddings)
        projected_text = self.text_projection(text_embeddings)
        #print('projected_protein',projected_protein.shape)
        #print('projected_text',projected_text.shape)
        # Adapt protein representation to text space
        #adapted_protein = self.protein_adapter(projected_protein)
        #adapted_protein = self.protein_adapter(projected_protein,projected_text)
        #print('adapted_protein',adapted_protein.shape)
        return projected_protein, projected_text

    def protein_forward(self, protein_embeddings):
        with autocast(device_type='cuda'):
            protein_embeddings = F.normalize(protein_embeddings, dim=-1)
            projected_protein = self.protein_projection(protein_embeddings)
            return projected_protein

    def alignment_seq_and_text(self, adapted_protein, projected_text):
        # Normalize the features
        #adapted_protein = adapted_protein / adapted_protein.norm(dim=-1, keepdim=True)
        #projected_text = projected_text / projected_text.norm(dim=-1, keepdim=True)
        adapted_protein = F.normalize(adapted_protein, dim=-1)
        projected_text = F.normalize(projected_text, dim=-1)
        # Compute the logit scale factor
        #logit_scale = self.logit_scale.exp()

        # Calculate the predicted similarity from protein sequences to text and from text to protein sequences
        #pred_protein2text = logit_scale * adapted_protein @ projected_text.t()
        pred_protein2text = adapted_protein @ projected_text.t()
        #pred_text2protein = logit_scale * projected_text @ adapted_protein.t()

        # Calculate the target indices
        target_protein2text = torch.arange(adapted_protein.shape[0], device=pred_protein2text.device)
        #target_text2protein = torch.arange(projected_text.shape[0], device=pred_text2protein.device)

        # P/mnt/petrelfs/lvying/LLM/OPUS-BioLLM/multi_modality_model/downstream_tasks/resultsackage predictions and targets
        #preds = {"protein2text": pred_protein2text, "text2protein": pred_text2protein}
        #targets = {"protein2text": target_protein2text, "text2protein": target_text2protein}
        preds = {"protein2text": pred_protein2text}
        targets = {"protein2text": target_protein2text}
        #print("preds similary", pred_protein2text)
        #print("preds",preds,preds['protein2text'].shape)
        #print("targets",targets,targets['protein2text'].shape)
       # print(preds['protein2text'].shape)
       # print(targets['protein2text'].shape)
         
        return preds, targets
    def alignment_seq_and_text_h(self, adapted_protein, projected_text):
        # Normalize the features
        #adapted_protein = adapted_protein / adapted_protein.norm(dim=-1, keepdim=True)
        #projected_text = projected_text / projected_text.norm(dim=-1, keepdim=True)
        #adapted_protein = F.normalize(adapted_protein, dim=-1)
        #projected_text = F.normalize(projected_text, dim=-1)
        # Compute the logit scale factor
       # logit_scale = self.logit_scale.exp()

        # Calculate the predicted similarity from protein sequences to text and from text to protein sequences
        #pred_protein2text = logit_scale * adapted_protein @ projected_text.t()
        #pred_text2protein = logit_scale * projected_text @ adapted_protein.t()

        pred_protein2text =  adapted_protein @ projected_text.t()
        pred_text2protein =  projected_text @ adapted_protein.t()
        # Calculate the target indices
       # print("==============================")
       # print("logit_scale",logit_scale)
       # print("pred_protein2text",pred_protein2text)
        #print("no ligits_scale pred_protein2text",adapted_protein @ projected_text.t())
        #print("no ligits_scale pred_text2protein",projected_text @ adapted_protein.t())
        #print("pred_text2protein",pred_text2protein)
       # print("==============================")
        target_protein2text = torch.arange(adapted_protein.shape[0], device=pred_protein2text.device)
        target_text2protein = torch.arange(projected_text.shape[0], device=pred_text2protein.device)

        # P/mnt/petrelfs/lvying/LLM/OPUS-BioLLM/multi_modality_model/downstream_tasks/resultsackage predictions and targets
        preds = {"protein2text": pred_protein2text, "text2protein": pred_text2protein}
        targets = {"protein2text": target_protein2text, "text2protein": target_text2protein}
        #preds = {"protein2text": pred_protein2text}
        #targets = {"protein2text": target_protein2text}
        #print("preds similary", pred_protein2text)
        #print("preds",preds,preds['protein2text'].shape)
        #print("targets",targets,targets['protein2text'].shape)
       # print(preds['protein2text'].shape)
       # print(targets['protein2text'].shape)
         
        return preds, targets


class CSTPLightning(CSTPBase, pl.LightningModule):
    def __init__(self, *args, **kwargs):
        CSTPBase.__init__(self, *args, **kwargs)
        #pl.LightningModule.__init__(self)

        # Define the InfoNCE Loss function
        #self.loss_fn = InfoNCELoss_h()
        self.loss_fn = TrustEvidenceLoss()
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def training_step(self, batch, batch_idx):
        # Single training step during the training process
        protein_seq, text = batch['seq_embedding'], batch['text_embedding'] #记得这里需要处理
       # print(torch.sum(protein_seq,-1))
       # print(torch.sum(text,-1))
        #protein_seq = protein_seq / protein_seq.norm(dim=-1, keepdim=True)
        #text = text / text.norm(dim=-1, keepdim=True)
       
      #  print(torch.sum(protein_seq,-1))
      #  print(torch.sum(text,-1))
        adapted_protein, projected_text = self.forward(protein_seq, text)
        #print(adapted_protein.shape)
        #print(projected_text.shape)
        preds, targets = self.alignment_seq_and_text_h(adapted_protein, projected_text)
       # loss = self.loss_fn(preds["protein2text"], targets["protein2text"])
        loss = self.loss_fn(preds, targets)

        output = {'train_loss': loss}
        #print('batch_train_loss',loss)
        self.training_step_outputs.append(output)
        self.log_dict({'train_loss':loss}, batch_size=len(protein_seq),sync_dist=True)
        #pl.utilities.rank_zero_info(
        #    f"Train loss at batch {batch_idx}: {loss:.4f}"
        #)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Single validation step during the validation process
        protein_seq, text = batch['seq_embedding'], batch['text_embedding']
        #print(protein_seq.shape)
        #print(text.shape) 
        adapted_protein, projected_text = self.forward(protein_seq, text)
        #print(adapted_protein.shape)
        #print(projected_text.shape)
        preds, targets = self.alignment_seq_and_text_h(adapted_protein, projected_text)
        #loss = self.loss_fn(preds["protein2text"], targets["protein2text"])
        loss = self.loss_fn(preds, targets)
        output = {"val_loss": loss}
        #print('batch_val_loss',loss)
        max_values_p, max_indices_p = torch.max(preds['protein2text'], dim=1)
        correct_predictions_p = torch.sum(max_indices_p == targets['protein2text'], dim=0)
        accuracy_p = correct_predictions_p.float()/targets['protein2text'].shape[0]
        print("accuracy_p=",accuracy_p)
        max_values_t, max_indices_t = torch.max(preds['text2protein'], dim=1)
        correct_predictions_t = torch.sum(max_indices_t == targets['text2protein'], dim=0)
        accuracy_t = correct_predictions_t.float()/targets['text2protein'].shape[0]
        print("accuracy_t=",accuracy_t)
        
        self.validation_step_outputs.append(output)
        self.log_dict({"val_loss": loss}, batch_size=len(protein_seq),sync_dist=True)
        #pl.utilities.rank_zero_info(
        #    f"Val loss at batch {batch_idx}: {loss:.4f}"
        #)
        return  {"val_loss": loss}
    
    def on_train_epoch_end(self) -> None:
        losses = torch.stack([o["train_loss"] for o in self.training_step_outputs])
        mean_loss = torch.mean(losses)
        t_delta = time.time() - self.train_epoch_last_time
        pl.utilities.rank_zero_info(
            f"Train loss at epoch {self.train_epoch_counter} end: {mean_loss:.4f} ({t_delta:.2f} seconds)"
        )
        self.training_step_outputs.clear()
        # Increment the epoch counter and update the time
        self.train_epoch_counter += 1
        self.train_epoch_last_time = time.time()
    
    def on_validation_epoch_end(self) -> None:
        """Log the average validation loss over the epoch"""
        # Aggregate the losses
        val_losses = torch.stack([x['val_loss'] for x in self.validation_step_outputs])
        avg_val_loss = torch.mean(val_losses)
        # Log the average validation loss
        self.log('avg_val_loss', avg_val_loss, on_epoch=True, prog_bar=True,sync_dist=True)
        self.validation_step_outputs.clear()
    
    def test_step(self, batch, batch_idx):
        protein_seq, text = batch['seq_embedding'], batch['text_embedding']
        #print(protein_seq,labels)
        
        adapted_protein, projected_text = self.forward(protein_seq, text)
        preds, targets = self.alignment_seq_and_text_h(adapted_protein, projected_text)
        loss = self.loss_fn(preds, targets)
        output = {"val_loss": loss}
        max_values_p, max_indices_p = torch.max(preds['protein2text'], dim=1)
        correct_predictions_p = torch.sum(max_indices_p == targets['protein2text'], dim=0)
        accuracy_p = correct_predictions_p.float()/targets['protein2text'].shape[0]
        print("accuracy=",accuracy_p)
        max_values_t, max_indices_t = torch.max(preds['text2protein'], dim=1)
        correct_predictions_t = torch.sum(max_indices_t == targets['text2protein'], dim=0)
        accuracy_t = correct_predictions_t.float()/targets['text2protein'].shape[0]
        print("accuracy=",accuracy_t)
            
        self.log('test_loss', loss)
        self.log('test_acc_p', accuracy_p)
        self.log('test_acc_t', accuracy_t)
        self.test_step_outputs.append({'test_loss': torch.tensor(loss), 'test_p_acc': torch.tensor(accuracy_p), 'test_t_acc': torch.tensor(accuracy_t)})

        return {'test_loss': loss, 'test_p_acc':accuracy_p,'test_t_acc': accuracy_t }
    
    def on_test_epoch_end(self):
        avg_loss = torch.stack([x['test_loss'] for x in self.test_step_outputs]).mean()
        avg_acc_p = torch.stack([x['test_p_acc'] for x in self.test_step_outputs]).mean()
        avg_acc_t = torch.stack([x['test_t_acc'] for x in self.test_step_outputs]).mean()
        print(f'Test Loss: {avg_loss}, Test Accuracy Protein: {avg_acc_p},Test Accuracy Tet: {avg_acc_t}')

        return {'test_loss': torch.tensor(avg_loss), 'test_acc_p': torch.tensor(avg_acc_p), 'test_acc_t': torch.tensor(avg_acc_t)}

    
    
    def configure_optimizers(self):
        # Define the optimizer
        #optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=0.0005,weight_decay=0.001)
       # optimizer = torch.optim.Adam(self.parameters(), lr=0.0001,weight_decay=0.001)
      #  optimizer = torch.optim.AdamW(self.parameters(), lr=0.0001,weight_decay=0.001) #针对InfoNCELoss_h用的优化器设置
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.05,weight_decay=0.0001) #针对evidence loss
       # optimizer = torch.optim.AdamW(self.parameters(), lr=0.01) #针对evidence loss
        return optimizer
    
    #def configure_ddp(self, model, device_ids):
    #    ddp = DistributedDataParallel(
    #        model,
    #        device_ids=device_ids,
    #        find_unused_parameters=True  # 设置find_unused_parameters=True
    #    )
    #    return ddp 

#class CTSP():
"""
Contrastive Text-Sequence of Protein Pretraining 
Mapping text to protein sequence space

"""

    


























