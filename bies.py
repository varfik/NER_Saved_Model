import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from transformers import AutoModel, AutoTokenizer, AutoConfig, BertConfig
from torchcrf import CRF
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler  # Added WeightedRandomSampler
import networkx as nx
import matplotlib.pyplot as plt
import random
import os
import json
from collections import defaultdict
from tqdm.auto import tqdm
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence
import numpy as np

ENTITY_TYPES = {
    'PERSON': 1,
    'PROFESSION': 2,
    'ORGANIZATION': 3,
    'FAMILY': 4
}

# BIOES labels: O, B, I, E, S для каждого типа сущности (5 labels на тип)
# Формат: O, B-PER, I-PER, E-PER, S-PER, B-PROF, I-PROF, E-PROF, S-PROF, ...
NUM_BIOES_LABELS = 1 + 4 * len(ENTITY_TYPES)  # O + 4 labels * 4 entity types

RELATION_TYPES = {
    'WORKS_AS': 1,
    'MEMBER_OF': 2,
    'FOUNDED_BY': 3,
    'SPOUSE': 4,
    'PARENT_OF': 5,
    'SIBLING': 6

}

class ContextAwareAttention(nn.Module):
    """Механизм внимания между эмбеддингами сущностей и контекстом документа"""
    def __init__(self, hidden_size):
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, entity_embeddings, context_embeddings, attention_mask=None):
        """
        entity_embeddings: [num_entities, hidden_size]
        context_embeddings: [seq_len, hidden_size]
        attention_mask: [seq_len]
        """
        # Project to query, key, value
        q = self.query(entity_embeddings)  # [num_entities, hidden_size]
        k = self.key(context_embeddings)   # [seq_len, hidden_size]
        v = self.value(context_embeddings) # [seq_len, hidden_size]
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(0, 1))  # [num_entities, seq_len]
        scores = scores / (q.size(-1) ** 0.5)
        
        # Apply mask if provided
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask.unsqueeze(0) == 0, -1e9)
        
        # Attention weights
        attn_weights = self.softmax(scores)  # [num_entities, seq_len]
        
        # Weighted sum of values
        output = torch.matmul(attn_weights, v)  # [num_entities, hidden_size]
        
        return output, attn_weights

class NERRelationModel(nn.Module):
    def __init__(self, model_name="DeepPavlov/rubert-base-cased", num_ner_labels=NUM_BIOES_LABELS, num_rel_labels=7):
        super().__init__()
        self.num_ner_labels = num_ner_labels # O, B-PER, I-PER, B-PROF, I-PROF, B-ORG, I-ORG, B-FAM, I-FAM
        self.num_rel_labels = num_rel_labels
        
        # BERT encoder
        self.bert = AutoModel.from_pretrained(model_name)
        
        # NER Head with CRF
        self.ner_classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_ner_labels)
        )
        self.crf = CRF(num_ner_labels, batch_first=True)
        
        # Context-aware attention
        self.context_attention = ContextAwareAttention(self.bert.config.hidden_size)

        # Graph attention network components (GAT)
        self.gat1 = GATConv(self.bert.config.hidden_size, 128, heads=4, dropout=0.3)
        self.gat2 = GATConv(128*4, 64, heads=1, dropout=0.3)
        # Concatenate heads from first layer
    
        # Relation classifiers
        self.rel_classifiers = nn.ModuleDict({
            'WORKS_AS': self._build_relation_classifier(),
            'MEMBER_OF': self._build_relation_classifier(),
            'FOUNDED_BY': self._build_relation_classifier(),
            'SPOUSE': self._build_relation_classifier(),
            'PARENT_OF': self._build_relation_classifier(),
            'SIBLING': self._build_relation_classifier()
        })

        # Инициализация весов
        self._init_weights()

    def _build_relation_classifier(self):
        return nn.Sequential(
            nn.Linear(64 * 2, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def _init_weights(self):
        for module in [self.ner_classifier, *self.rel_classifiers.values()]:
            if isinstance(module, nn.Sequential):
                for submodule in module:
                    if isinstance(submodule, nn.Linear):
                        nn.init.xavier_uniform_(submodule.weight)
                        nn.init.constant_(submodule.bias, 0)

    def forward(self, input_ids, attention_mask, ner_labels=None, rel_data=None):
        device = input_ids.device
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # NER prediction with CRF
        ner_logits = self.ner_classifier(sequence_output)
        total_loss = 0
        
        # NER loss
        if ner_labels is not None:
            mask = attention_mask.bool()
            ner_loss = -self.crf(ner_logits, ner_labels, mask=mask, reduction='mean')
            total_loss += ner_loss

        # Relation extraction
        rel_probs = {}
        rel_targets = {}

        if rel_data and self.training:
            total_rel_loss = 0
            rel_correct = 0
            rel_total = 0
            # Process each sample in batch
            for batch_idx, sample in enumerate(rel_data):
                if 'pairs' not in sample or len(sample['pairs']) == 0:
                    continue
                    
                # Get valid entities
                valid_entities = [
                    e for e in sample['entities'] 
                    if isinstance(e, dict) and 'start' in e and 'end' in e
                ]
                
                if len(valid_entities) < 2:
                    continue
                    
                # Create initial entity embeddings (mean over tokens)
                entity_embeddings = []
                for e in valid_entities:
                    start = min(e['start'], sequence_output.size(1)-1)
                    end = min(e['end'], sequence_output.size(1)-1)
                    entity_embed = sequence_output[batch_idx, start:end+1].mean(dim=0)
                    entity_embeddings.append(entity_embed)
                
                # Apply context-aware attention
                entity_embeddings = torch.stack(entity_embeddings).to(device)
                context_emb = sequence_output[batch_idx]  # [seq_len, hidden_size]
                attended_entities, _ = self.context_attention(
                    entity_embeddings, 
                    context_emb,
                    attention_mask[batch_idx]
                )
                entity_embeddings = entity_embeddings + attended_entities  # Residual connection

                # Build complete graph
                edge_index = torch.tensor([
                    [i, j] for i in range(len(valid_entities)) 
                    for j in range(len(valid_entities)) if i != j
                ], dtype=torch.long).t().contiguous().to(device)
                
                # Apply GAT
                x = self.gat1(entity_embeddings, edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=0.3, training=self.training)
                x = self.gat2(x, edge_index)
                
                # Process each relation type
                for rel_type in RELATION_TYPES:
                    rel_probs[rel_type] = []
                    rel_targets[rel_type] = []
                    entity_indices = {e['id']: i for i, e in enumerate(valid_entities)}
                    # Process positive pairs first
                    pos_pairs = 0
                    for (e1_idx, e2_idx), label in zip(sample['pairs'], sample['labels']):
                        if e1_idx in entity_indices and e2_idx in entity_indices:
                            i = entity_indices[e1_idx]
                            j = entity_indices[e2_idx]
                            e1_type = valid_entities[i]['type']
                            e2_type = valid_entities[j]['type']
                            
                            if self._is_valid_pair(e1_type, e2_type, rel_type):
                                pair_features = torch.cat([x[i], x[j]])
                                rel_probs[rel_type].append(self.rel_classifiers[rel_type](pair_features))
                                rel_targets[rel_type].append(label)
                                pos_pairs += 1

                    # Generate balanced negative examples
                    neg_pairs = min(pos_pairs * 3, len(valid_entities)**2 - pos_pairs)  # Max 3:1 ratio
                    neg_count = 0
                    
                    # Shuffle all possible negative pairs
                    all_pairs = [(i,j) for i in range(len(valid_entities)) 
                                for j in range(len(valid_entities)) if i != j]
                    random.shuffle(all_pairs)
                    
                    for i, j in all_pairs:
                        if neg_count >= neg_pairs:
                            break
                        e1 = valid_entities[i]
                        e2 = valid_entities[j]
                        if (e1['id'], e2['id']) not in pos_indices and self._is_valid_pair(e1['type'], e2['type'], rel_type):
                            pair_features = torch.cat([x[i], x[j]])
                            rel_probs[rel_type].append(self.rel_classifiers[rel_type](pair_features))
                            rel_targets[rel_type].append(0.0)
                            neg_count += 1
                    
                    if rel_probs[rel_type]:
                        probs_tensor = torch.cat(rel_probs[rel_type]).view(-1)
                        targets_tensor = torch.tensor(rel_targets[rel_type], dtype=torch.float, device=device)

                        # Ensure same length
                        min_len = min(len(probs_tensor), len(targets_tensor))
                        if min_len > 0:
                            rel_loss = nn.BCEWithLogitsLoss()(
                                probs_tensor[:min_len], 
                                targets_tensor[:min_len]
                            )
                            total_loss += rel_loss

                        preds = (torch.sigmoid(probs_tensor) > 0.5).float()
                        rel_correct += (preds == targets_tensor).sum().item()
                        rel_total += targets_tensor.size(0)

        return {
            'ner_logits': ner_logits,
            'rel_probs': rel_probs,
            'loss': total_loss if total_loss != 0 else None
        }

    def _is_valid_pair(self, e1_type, e2_type, rel_type):
        relation_rules = {
            'WORKS_AS': [('PERSON', 'PROFESSION')],
            'MEMBER_OF': [('PERSON', 'ORGANIZATION'), ('ORGANIZATION', 'ORGANIZATION')],
            'FOUNDED_BY': [('ORGANIZATION', 'PERSON')],
            'SPOUSE': [('PERSON', 'PERSON')],
            'PARENT_OF': [('PERSON', 'PERSON'), ('PERSON', 'FAMILY')],
            'SIBLING': [('PERSON', 'PERSON')]
        }
        print(f"VALID: {e1_type} -> {e2_type} for {rel_type}")
        return (e1_type, e2_type) in relation_rules.get(rel_type, [])

    def _generate_negative_examples(self, entity_embeddings, entity_types, rel_type, ratio=0.5):
        device = entity_embeddings.device
        neg_probs = []
        neg_targets = []
        
        # Собираем все возможные валидные пары для этого типа отношения
        possible_pairs = []
        for i, e1_type in enumerate(entity_types):
            for j, e2_type in enumerate(entity_types):
                if i != j and self._is_valid_pair(e1_type, e2_type, rel_type):
                    possible_pairs.append((i, j))

        # Выбираем случайное подмножество пар в качестве отрицательных примеров
        num_neg = max(1, int(len(possible_pairs) * ratio))
        sampled_pairs = random.sample(possible_pairs, min(num_neg, len(possible_pairs)))

        for i, j in sampled_pairs:
            pair_features = torch.cat([entity_embeddings[i], entity_embeddings[j]])
            neg_probs.append(self.rel_classifiers[rel_type](pair_features))
            neg_targets.append(0.0)
        
        if neg_probs:
            return torch.cat(neg_probs).view(-1), torch.tensor(neg_targets, device=device)
        return torch.tensor([], device=device), torch.tensor([], device=device)

    def save_pretrained(self, save_dir, tokenizer=None):
        """Сохраняет модель, конфигурацию и токенизатор в указанную директорию."""
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Сохраняем веса модели
        torch.save(self.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
        
        # 2. Сохраняем конфигурацию модели в формате Hugging Face
        config = {
            "model_type": "bert",  # Указываем тип модели для Hugging Face
            "model_name": self.bert.name_or_path,
            "num_ner_labels": self.num_ner_labels,
            "bert_config": self.bert.config.to_dict()
        }
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
        
        # 3. Сохраняем конфигурацию токенизатора
        if tokenizer is not None:
            tokenizer.save_pretrained(save_dir)
    
    @classmethod
    def from_pretrained(cls, model_dir, device="cuda"):
        """Загружает модель из указанной директории."""
        # 1. Загружаем конфигурацию
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # 2. Инициализируем BERT с сохраненной конфигурацией
        bert_config = BertConfig.from_dict(config["bert_config"])
        bert = AutoModel.from_config(bert_config)
        
        # 3. Создаем экземпляр модели
        model = cls(
            model_name=config["model_name"],
            num_ner_labels=config["num_ner_labels"]
        ).to(device)
        
        # 4. Заменяем BERT на загруженную версию
        model.bert = bert.to(device)
        
        # 5. Загружаем веса модели
        model.load_state_dict(torch.load(
            os.path.join(model_dir, "pytorch_model.bin"), 
            map_location=device
        ))
        
        model.eval()
        return model

class NERELDataset(Dataset):
    def __init__(self, data_dir, tokenizer, max_length=512):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self._load_samples()
        
    def _load_samples(self):
        samples = []
        for txt_file in os.listdir(self.data_dir):
            if not txt_file.endswith('.txt'):
                continue
                
            ann_path = os.path.join(self.data_dir, txt_file.replace('.txt', '.ann'))
            if not os.path.exists(ann_path):
                continue
                
            with open(os.path.join(self.data_dir, txt_file), 'r', encoding='utf-8') as f:
                text = f.read()
            
            entities, relations = self._parse_ann_file(ann_path, text)
            samples.append({'text': text, 'entities': entities, 'relations': relations})
        
        return samples
    
    def _parse_ann_file(self, ann_path, text):
        entities, relations = [], []
        entity_map = {}
        
        with open(ann_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith('T'):
                    parts = line.split('\t')
                    if len(parts) < 3:
                        continue 

                    entity_id = parts[0]
                    type_and_span = parts[1].split()
                    entity_type = type_and_span[0]

                    # Поддерживаемые типы сущностей
                    if entity_type in ['PERSON', 'PROFESSION', 'ORGANIZATION', 'FAMILY']:
                        start = int(type_and_span[1])
                        end = int(type_and_span[-1])
                        entity_text = parts[2]
                        
                        # Verify entity span matches text
                        if text[start:end] != entity_text:
                            # Try to find correct span
                            found_pos = text.find(entity_text)
                            if found_pos != -1:
                                start = found_pos
                                end = found_pos + len(entity_text)
                            else:
                                continue # Пропускать сущности, которые не найдены в тексте
                    
                        entity = {
                            'id': entity_id,
                            'type': entity_type,
                            'start': start,
                            'end': end,
                            'text': entity_text
                        }
                        entities.append(entity)
                        entity_map[entity_id] = entity
                
                elif line.startswith('R'):
                    parts = line.strip().split('\t')
                    if len(parts) < 2:
                        continue
                    
                    rel_type = parts[1].split()[0]
                    arg1 = parts[1].split()[1].split(':')[1]
                    arg2 = parts[1].split()[2].split(':')[1]
                
                    # Проверяем существование сущностей
                    if arg1 not in entity_map or arg2 not in entity_map:
                        continue
                
                    relations.append({
                            'type': rel_type,
                            'arg1': arg1,
                            'arg2': arg2
                    })
        return entities, relations
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample['text']

        # Tokenize with subword information
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_offsets_mapping=True,
            return_tensors='pt'
        )
        
        # Initialize NER labels (0=O, 1=B-PER, 2=I-PER, 3=B-PROF, 4=I-PROF, 5=B-ORGANIZATION, 6=I-ORGANIZATION, 7=B-FAMILY, 8=I-FAMILY)
        ner_labels = torch.zeros(self.max_length, dtype=torch.long)
        token_entities = []

        # Align entities with tokenization
        for entity in sample['entities']:
             # Find token spans for entity
            start_token = end_token = None
            for i, (start, end) in enumerate(encoding['offset_mapping'][0]):
                if start <= entity['start'] < end and start_token is None:
                    start_token = i
                if start < entity['end'] <= end and end_token is None:
                    end_token = i
                if start >= entity['end']:
                    break
            
            if start_token is not None and end_token is not None:
                entity_length = end_token - start_token + 1

                # Определяем тип сущности и соответствующие метки BIOES
                if entity['type'] == 'PERSON':
                    base_label = 1  # B-PER
                elif entity['type'] == 'PROFESSION':
                    base_label = 5  # B-PROF
                elif entity['type'] == 'ORGANIZATION':
                    base_label = 9  # B-ORG
                elif entity['type'] == 'FAMILY':
                    base_label = 13  # B-FAM

                # Устанавливаем метки в соответствии с BIOES-схемой
                if entity_length == 1:
                    # Single token entity - S label
                    ner_labels[start_token] = base_label + 3  # S-*
                else:
                    # Multi-token entity
                    ner_labels[start_token] = base_label  # B-*
                    ner_labels[start_token+1:end_token] = base_label + 1  # I-*
                    ner_labels[end_token] = base_label + 2  # E-*


                token_entities.append({
                    'start': start_token,
                    'end': end_token,
                    'type': entity['type'],
                    'id': entity['id']
                })

        # Prepare relation data
        rel_data = {
            'entities': token_entities,
            'pairs': [],
            'labels': []
        }
        
        token_entity_id_to_idx = {e['id']: i for i, e in enumerate(token_entities)}
        
        for relation in sample['relations']:
            arg1_token_idx = token_entity_id_to_idx.get(relation['arg1'], -1)
            arg2_token_idx = token_entity_id_to_idx.get(relation['arg2'], -1)
            
            if arg1_token_idx != -1 and arg2_token_idx != -1:
                e1_type = token_entities[arg1_token_idx]['type']
                e2_type = token_entities[arg2_token_idx]['type']
                
                # Validate relation type and entity types
                if relation['type'] == 'WORKS_AS' and e1_type == 'PERSON' and e2_type == 'PROFESSION':
                    rel_data['pairs'].append((arg1_token_idx, arg2_token_idx))
                    rel_data['labels'].append(RELATION_TYPES['WORKS_AS'])
                elif relation['type'] == 'MEMBER_OF' and e1_type == 'PERSON' and e2_type == 'ORGANIZATION':
                    rel_data['pairs'].append((arg1_token_idx, arg2_token_idx))
                    rel_data['labels'].append(RELATION_TYPES['MEMBER_OF'])
                elif relation['type'] == 'FOUNDED_BY' and e1_type == 'ORGANIZATION' and e2_type == 'PERSON':
                    rel_data['pairs'].append((arg2_token_idx, arg1_token_idx))  # Reverse order
                    rel_data['labels'].append(RELATION_TYPES['FOUNDED_BY'])
                elif relation['type'] in ['SPOUSE', 'PARENT_OF', 'SIBLING'] and e1_type == 'PERSON' and e2_type == 'PERSON':
                    rel_data['pairs'].append((arg1_token_idx, arg2_token_idx))
                    rel_data['labels'].append(RELATION_TYPES[relation['type']])
                elif relation['type'] == 'PARENT_OF' and e1_type == 'PERSON' and e2_type == 'FAMILY':
                    rel_data['pairs'].append((arg1_token_idx, arg2_token_idx))
                    rel_data['labels'].append(RELATION_TYPES['PARENT_OF'])
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'ner_labels': ner_labels,
            'rel_data': rel_data,
            'text': text,
            'offset_mapping': encoding['offset_mapping'].squeeze(0)
        }

def collate_fn(batch):
    # All elements already padded to max_length
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    ner_labels = torch.stack([item['ner_labels'] for item in batch])
    offset_mapping = torch.stack([item['offset_mapping'] for item in batch])

    device = input_ids.device

    rel_data = []
    # Собираем rel_data как список словарей
    for item in batch:
        rel_entry = {
            'entities': item['rel_data']['entities'],
            'pairs': torch.tensor(item['rel_data']['pairs'], dtype=torch.long) if item['rel_data']['pairs'] else torch.zeros((0, 2), dtype=torch.long),
            'labels': torch.tensor(item['rel_data']['labels'], dtype=torch.float) if item['rel_data']['labels'] else torch.zeros(0, dtype=torch.float)
        }
        rel_data.append(rel_entry)
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'ner_labels': ner_labels,
        'rel_data': rel_data,
        'texts': [item['text'] for item in batch],
        'offset_mapping': offset_mapping
    }

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Инициализация модели и токенизатора
    tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
    model = NERRelationModel().to(device)

    # Загрузка данных
    train_dataset = NERELDataset("NEREL/NEREL-v1.1/train", tokenizer)

    # Create weighted sampler to balance relation examples
    sample_weights = []
    for sample in train_dataset:
        has_relations = len(sample['rel_data']['labels']) > 0
        sample_weights.append(1.0 if has_relations else 0.3)
    
    sampler = WeightedRandomSampler(sample_weights, len(train_dataset), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=8, collate_fn=collate_fn, sampler=sampler)

    # Optimizer with different learning rates
    optimizer = AdamW([
        {'params': model.bert.parameters(), 'lr': 2e-5},
        {'params': model.ner_classifier.parameters(), 'lr': 1e-4},
        {'params': model.crf.parameters(), 'lr': 1e-4},
        {'params': model.gat1.parameters(), 'lr': 1e-3},
        {'params': model.gat2.parameters(), 'lr': 1e-3},
        {'params': model.rel_classifiers.parameters(), 'lr': 1e-3}
    ])
    
    # Training loop
    best_ner_f1 = 0
    # Цикл обучения
    for epoch in range(1):
        model.train()
        epoch_loss = 0
        ner_correct = ner_total = 0
        rel_correct = rel_total = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            
            # Перенос данных на устройство
            input_ids = batch['input_ids'].to(device)
            attention_mask =  batch['attention_mask'].to(device)
            ner_labels = batch['ner_labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                ner_labels=ner_labels,
                rel_data=batch['rel_data'] 
            )
            
            if outputs['loss'] is not None:
                outputs['loss'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += outputs['loss'].item()
            
            # NER metrics
            with torch.no_grad():
                mask = attention_mask.bool()
                ner_preds = model.crf.decode(outputs['ner_logits'], mask=mask)
                
                # Перебираем каждый пример в батче
                for i in range(len(ner_preds)):
                    # Получаем длину последовательности без паддинга
                    seq_len = mask[i].sum().item()
                    # Берем только нужные элементы (без паддинга)
                    pred = torch.tensor(ner_preds[i][:seq_len], device=device)
                    true = ner_labels[i][:seq_len]
                    
                    ner_correct += (pred == true).sum().item()
                    ner_total += seq_len
            
            # Вычисление метрик для отношений
            if outputs['rel_probs']:
                for rel_type, probs in outputs['rel_probs'].items():
                    if len(probs) > 0:
                         # Get predictions and targets for this relation type only
                        probs_tensor = torch.cat(probs).view(-1)
                        preds = (torch.sigmoid(probs_tensor) > 0.5).long()

                        # Get corresponding targets
                        targets_tensor = torch.tensor(rel_targets[rel_type], dtype=torch.float, device=device)
                        
                        # Ensure same length
                        min_len = min(len(preds), len(targets_tensor))
                        if min_len > 0:
                            rel_correct += (preds[:min_len] == targets_tensor[:min_len]).sum().item()
                            rel_total += min_len

        # Evaluation metrics
        ner_acc = ner_correct / ner_total if ner_total > 0 else 0
        rel_acc = rel_correct / rel_total if rel_total > 0 else 0
        
        print(f"\nEpoch {epoch+1} Results:")
        print(f"Loss: {epoch_loss/len(train_loader):.4f}")
        print(f"NER Accuracy: {ner_acc:.2%} ({ner_correct}/{ner_total})")
        print(f"Relation Accuracy: {rel_acc:.2%} ({rel_correct}/{rel_total})")

    save_dir = "saved_model"
    model.save_pretrained(save_dir, tokenizer=tokenizer)
    print(f"Model saved to {save_dir}")
    
    return model, tokenizer

def predict(text, model, tokenizer, device="cuda", relation_threshold=0.5):
    # Tokenize input with offset mapping
    encoding = tokenizer(text, return_tensors="pt", return_offsets_mapping=True, max_length=512,
        truncation=True)
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    offset_mapping = encoding['offset_mapping'][0].cpu().numpy()
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)

    # Decode NER with CRF
    mask = attention_mask.bool()
    ner_preds = model.crf.decode(outputs['ner_logits'], mask=mask)[0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    # Extract entities
    entities = []
    current_entity = None
    entity_id = 0
    
    # Create mapping from label indices to entity types and positions
    label_to_type = {
        1: ('PERSON', 'B'), 2: ('PERSON', 'I'), 3: ('PERSON', 'E'), 4: ('PERSON', 'S'),
        5: ('PROFESSION', 'B'), 6: ('PROFESSION', 'I'), 7: ('PROFESSION', 'E'), 8: ('PROFESSION', 'S'),
        9: ('ORGANIZATION', 'B'), 10: ('ORGANIZATION', 'I'), 11: ('ORGANIZATION', 'E'), 12: ('ORGANIZATION', 'S'),
        13: ('FAMILY', 'B'), 14: ('FAMILY', 'I'), 15: ('FAMILY', 'E'), 16: ('FAMILY', 'S')
    }

    i = 0
    while i < len(ner_preds):
        pred = ner_preds[i]
        token = tokens[i]
        
        if token in ['[CLS]', '[SEP]', '[PAD]']:
            if current_entity:
                entities.append(current_entity)
                current_entity = None
            i += 1
            continue
            
        if pred == 0:  # O
            if current_entity:
                entities.append(current_entity)
                current_entity = None
            i += 1
        else:
            entity_type, position = label_to_type[pred]
            
            if position in ['B', 'S']:
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    'id': f"T{entity_id}",
                    'type': entity_type,
                    'start': i,
                    'end': i,
                    'token_ids': [i],
                    'text': token.replace('##', '')
                }
                entity_id += 1
                i += 1
            elif position == 'I':
                if current_entity and current_entity['type'] == entity_type:
                    current_entity['end'] = i
                    current_entity['token_ids'].append(i)
                i += 1
            elif position == 'E':
                if current_entity and current_entity['type'] == entity_type:
                    current_entity['end'] = i
                    current_entity['token_ids'].append(i)
                    entities.append(current_entity)
                    current_entity = None
                i += 1
    
    # Handle the case where the last entity wasn't added
    if current_entity:
        entities.append(current_entity)

    # Convert token positions to character positions
    for entity in entities:
        start_char = offset_mapping[entity['start']][0]
        end_char = offset_mapping[entity['end']][1]
        entity['text'] = text[start_char:end_char]
        entity['start_char'] = start_char
        entity['end_char'] = end_char

    # Extract relations
    relations = []
    entity_map = {e['id']: e for e in entities}
    if len(entities) >= 2:
        sequence_output = model.bert(input_ids, attention_mask).last_hidden_state

        # Create entity embeddings
        entity_embeddings = torch.stack([
            sequence_output[0, e['start']:e['end']+1].mean(dim=0) 
            for e in entities
        ]).to(device)   
        
        # Build complete graph
        edge_index = torch.tensor([
            [i, j] for i in range(len(entities)) 
            for j in range(len(entities)) if i != j
        ], dtype=torch.long).t().contiguous().to(device)

        # Apply GAT
        x = model.gat1(entity_embeddings, edge_index)
        x = F.relu(x)
        x = model.gat2(x, edge_index)
        
        # Predict all possible relations
        for rel_type in RELATION_TYPES:
            for i, e1 in enumerate(entities):
                for j, e2 in enumerate(entities):
                    if i != j and model._is_valid_pair(e1['type'], e2['type'], rel_type):
                        pair_features = torch.cat([x[i], x[j]])
                        logit = model.rel_classifiers[rel_type](pair_features)
                        prob = torch.sigmoid(logit).item()
                        
                        if prob > relation_threshold:
                            # For FOUNDED_BY we reverse the direction
                            if rel_type == 'FOUNDED_BY':
                                i, j = j, i  # Organization should be first
                            
                            relations.append({
                                'type': rel_type,
                                'arg1_id': e1['id'],
                                'arg2_id': e2['id'],
                                'arg1': e1,
                                'arg2': e2,
                                'confidence': prob,
                                'direction': f"{e1['type']}->{e2['type']}"
                            })
    
    # Remove duplicates and keep highest confidence
    unique_relations = {}
    for rel in relations:
        key = (rel['arg1_id'], rel['arg2_id'], rel['type'])
        if key not in unique_relations or rel['confidence'] > unique_relations[key]['confidence']:
            unique_relations[key] = rel
    
    # Sort by confidence
    sorted_relations = sorted(unique_relations.values(), 
                             key=lambda x: x['confidence'], reverse=True)
    
    return {
        'text': text,
        'entities': entities,
        'relations': sorted_relations
    }

if __name__ == "__main__":
    model, tokenizer = train_model()
    test_texts = [
        "Айрат Мурзагалиев, заместитель начальника управления президента РФ, встретился с главой администрации Уфы.",
        "Иван Петров работает программистом в компании Яндекс.",
        "Доктор Сидоров принял пациентку Ковалеву в городской больнице.",
        "Директор сводного экономического департамента Банка России Надежда Иванова назначена также на должность заместителя председателя ЦБ, сообщил в четверг регулятор."
    ]
    
    for text in test_texts:
        print("\n" + "="*80)
        print(f"Processing text: '{text}'")
        result = predict(text, model, tokenizer)
        print("\nEntities:")
        for e in result['entities']:
            print(f"{e['type']}: {e['text']}")
        print("\nRelations:")
        for r in result['relations']:
            print(f"{r['type']}: {r['arg1']['text']} -> {r['arg2']['text']} (conf: {r['confidence']:.2f})")

    # Для загрузки модели
    loaded_model = NERRelationModel.from_pretrained("saved_model")
    loaded_tokenizer = AutoTokenizer.from_pretrained("saved_model")
    
    # Использование модели
    result = predict("По улице шел красивый человек, его имя было Мефодий. И был он счастлив. Работал этот чувак в яндексе, разработчиком. Или директором. Он пока не определился!", loaded_model, loaded_tokenizer)
    print("Сущности:")
    for e in result['entities']:
        print(f"{e['type']}: {e['text']} (позиция: {e['start_char']}-{e['end_char']})")

    print("\nОтношения:")
    for r in result['relations']:
        print(f"{r['type']}: {r['arg1']['text']} -> {r['arg2']['text']} (confidence: {r['confidence']:.2f})")

