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

import regex
import unicodedata
from termcolor import colored

# Цвета для визуализации разных типов сущностей
ENTITY_COLORS = {
    'PERSON': 'cyan',
    'PROFESSION': 'green',
    'ORGANIZATION': 'yellow',
    'FAMILY': 'magenta',
    'LOCATION': 'blue',
}

# Пороги уверенности для разных типов отношений
RELATION_THRESHOLDS = {
    'WORKS_AS': 0.7,
    'MEMBER_OF': 0.7,
    'FOUNDED_BY': 0.7,
    'SPOUSE': 0.5,
    'PARENT_OF': 0.5,
    'SIBLING': 0.5,
    'PART_OF': 0.7,
    'WORKPLACE': 0.7,
    'RELATIVE': 0.5
}

# Маппинг типов сущностей в числовые индексы
ENTITY_TYPES = {
    'PERSON': 1,
    'PROFESSION': 2,
    'ORGANIZATION': 3,
    'FAMILY': 4,
    'LOCATION': 5
}

# Маппинг типов отношений в числовые индексы
RELATION_TYPES = {
    'WORKS_AS': 1,
    'MEMBER_OF': 2,
    'FOUNDED_BY': 3,
    'SPOUSE': 4,
    'PARENT_OF': 5,
    'SIBLING': 6,
    'PART_OF': 7,    
    'WORKPLACE': 8,     
    'RELATIVE': 9      
}

# Допустимые комбинации типов сущностей для каждого типа отношения
VALID_COMB = {
    'WORKS_AS': [('PERSON', 'PROFESSION')],
    'MEMBER_OF': [('PERSON', 'ORGANIZATION')],
    'FOUNDED_BY': [('ORGANIZATION', 'PERSON'), ('LOCATION', 'PERSON')],
    'SPOUSE': [('PERSON', 'PERSON')],
    'PARENT_OF': [('PERSON', 'PERSON')],
    'SIBLING': [('PERSON', 'PERSON')],
    'PART_OF': [('ORGANIZATION', 'ORGANIZATION'), ('LOCATION', 'LOCATION')],
    'WORKPLACE': [('PERSON', 'ORGANIZATION'), ('PERSON', 'LOCATION')],
    'RELATIVE': [('PERSON', 'PERSON')]
}

# Обратный маппинг индексов в названия отношений
RELATION_TYPES_INV = {v: k for k, v in RELATION_TYPES.items()}

class NERRelationModel(nn.Module):
    def __init__(self, model_name="DeepPavlov/rubert-base-cased", num_ner_labels=len(ENTITY_TYPES)*2+1, num_rel_labels=len(RELATION_TYPES), relation_types=None, entity_types=None):
        super().__init__()
        # Initialize NER labels (0=O, 1=B-PER, 2=I-PER, ..., 9=B-LOC, 10=I-LOC)
        self.entity_type_to_idx = {etype: idx for idx, etype in enumerate(entity_types)}
        self.relation_types = relation_types
        self.entity_types = entity_types
        self.num_ner_labels = num_ner_labels
        self.num_rel_labels = num_rel_labels
        
        # BERT encoder
        self.bert = AutoModel.from_pretrained(model_name) # BERT-модель для эмбеддингов
        hidden_size = self.bert.config.hidden_size
        
        # NER Head with CRF
        # Классификатор для NER
        self.ner_classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_ner_labels)
        )
        self.crf = CRF(num_ner_labels, batch_first=True) # CRF слой для NER

        # Графовые слои внимания (GAT)
        self.gat1 = GATConv(
            self.bert.config.hidden_size, 
            128, 
            heads=2, 
            dropout=0.3,
            concat=True
        )
        self.norm1 = nn.LayerNorm(128 * 2)
        self.gat2 = GATConv(
            128 * 2, 
            64, 
            heads=1, 
            dropout=0.3,
            concat=False
        )
        self.norm2 = nn.LayerNorm(64)
        # Concatenate heads from first layer

        # Эмбеддинги для типов сущностей и отношений
        self.entity_type_emb = nn.Embedding(len(entity_types), 768)
        self.rel_type_emb = nn.Embedding(self.num_rel_labels, 32)

        # Классификатор отношений
        self.rel_classifier = nn.Sequential(
            nn.Linear(64 * 2 + 32 + hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.rel_classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_ids, attention_mask, ner_labels=None, rel_data=None):
        device = input_ids.device
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state # [B, T, H]
        cls_token = sequence_output[:, 0, :]  # [B, H]

        # NER with CRF
        ner_logits = self.ner_classifier(sequence_output)
        loss = 0
        if ner_labels is not None:
            mask = attention_mask.bool()
            ner_loss = -self.crf(ner_logits, ner_labels, mask=mask, reduction='mean')
            loss += ner_loss

        rel_probs = defaultdict(list)

        if rel_data:
            for batch_idx, sample in enumerate(rel_data):
                entities, id_map, x = self._encode_entities(sequence_output, batch_idx, sample, device)
                if x is None or len(id_map) < 2:
                    continue

                x = self._compute_gat(x, device)

                # ======= Положительные пары =======
                pos_indices_by_type = defaultdict(list)
                for (i1, i2), label in zip(sample['pairs'], sample['labels']):
                    if i1 not in id_map or i2 not in id_map:
                        continue
                    idx1, idx2 = id_map[i1], id_map[i2]
                    if self.relation_types[label] == 'FOUNDED_BY':
                        idx1, idx2 = idx2, idx1
                    rel_type_tensor = self.rel_type_emb(torch.tensor(label, device=device))
                    pair_vec = torch.cat([x[idx1], x[idx2], rel_type_tensor, cls_token[batch_idx]])
                    score = self.rel_classifier(pair_vec)
                    rel_probs[label].append((score, 1.0))
                    pos_indices_by_type[label].append((idx1, idx2))

                # ======= Отрицательные пары (только в train) =======
                if self.training:
                    for rel_type_idx, rel_type in enumerate(self.relation_types):
                        negatives = self._get_negatives(x, entities, rel_type_idx, rel_type, pos_indices_by_type,
                                                        device)
                        for i, j in negatives:
                            rel_type_tensor = self.rel_type_emb(torch.tensor(rel_type_idx, device=device))
                            pair_vec = torch.cat([x[i], x[j], rel_type_tensor, cls_token[batch_idx]])
                            score = self.rel_classifier(pair_vec)
                            rel_probs[rel_type_idx].append((score, 0.0))

            # ======= Loss по отношениям (train only) =======
            if self.training:
                for rel_type_idx, pairs in rel_probs.items():
                    if not pairs:
                        continue
                    logits, labels = zip(*pairs)
                    logits = torch.cat(logits).view(-1)
                    labels = torch.tensor(labels, device=device, dtype=torch.float)
                    pos_weight = torch.tensor(
                        [3.0 if self.relation_types[rel_type_idx] in ['SPOUSE', 'SIBLING'] else 1.0],
                        device=device)
                    rel_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(logits, labels)
                    loss += rel_loss

        return {
            'ner_logits': ner_logits,
            'rel_probs': rel_probs,
            'loss': loss if loss != 0 else None
        }

    def _encode_entities(self, sequence_output, batch_idx, sample, device):
        entities = [e for e in sample['entities'] if 'start' in e and 'end' in e]
        if len(entities) < 2:
            if self.training:
                print(f"Пропуск примера {batch_idx}: недостаточно сущностей ({len(entities)})")
            return None, None, None

        entity_embeds, id_map = [], {}
        for i, e in enumerate(entities):
            start, end = e['start'], e['end']
            token_embeds = sequence_output[batch_idx, start:end + 1]
            attention_scores = token_embeds @ token_embeds.mean(dim=0)
            attention_weights = torch.softmax(attention_scores, dim=0)
            pooled = torch.sum(token_embeds * attention_weights.unsqueeze(-1), dim=0)

            try:
                type_idx = self.entity_type_to_idx.get(e['type'], -1)
            except ValueError:
                if self.training:
                    print(f"Неизвестный тип сущности: {e['type']}")
                continue

            type_emb = self.entity_type_emb(torch.tensor(type_idx, device=device))
            entity_embeds.append(pooled + type_emb)
            id_map[e['id']] = len(entity_embeds) - 1

        if len(entity_embeds) < 2:
            return None, None, None

        x = torch.stack(entity_embeds)
        return entities, id_map, x

    def _compute_gat(self, x, device):
        edge_pairs = [[i, j] for i in range(len(x)) for j in range(len(x)) if i != j]
        edge_index = torch.tensor(edge_pairs).t().to(device)

        x = self.gat1(x, edge_index)
        x = self.norm1(x)
        x = F.elu(x)
        x = self.gat2(x, edge_index)
        x = self.norm2(x)
        x = F.elu(x)
        return x

    def _get_negatives(self, x, entities, rel_type_idx, rel_type, pos_indices_by_type, device):
        pos_set = set(pos_indices_by_type[rel_type_idx])
        if len(x) < 2:
            return []

        # Составим матрицу совместимых типов
        valid_combinations = VALID_COMB.get(rel_type, [])
        valid_pairs_mask = torch.zeros((len(x), len(x)), dtype=torch.bool, device=device)

        entity_types = [e['type'] for e in entities]
        for i in range(len(x)):
            for j in range(len(x)):
                if i == j or (i, j) in pos_set:
                    continue
                if (entity_types[i], entity_types[j]) in valid_combinations:
                    valid_pairs_mask[i, j] = True

        valid_indices = valid_pairs_mask.nonzero(as_tuple=False)
        if valid_indices.size(0) == 0:
            return []

        # Построим отрицательные пары
        i_indices = valid_indices[:, 0]
        j_indices = valid_indices[:, 1]
        neg_pairs = torch.stack([i_indices, j_indices], dim=1)

        # Если нет позитивов — возвращаем случайную подвыборку
        if not pos_set:
            sampled_ids = torch.randperm(len(neg_pairs))[:min(30, len(neg_pairs))]
            return [tuple(pair.cpu().tolist()) for pair in neg_pairs[sampled_ids]]

        # Расчёт эмбеддингов для позитивных пар
        pos_embeds = torch.stack([torch.cat([x[i], x[j]]) for i, j in pos_set])
        pos_mean = pos_embeds.mean(dim=0)

        # Расчёт эмбеддингов для всех кандидатов (батчево)
        x_i = x[i_indices]  # [N, H]
        x_j = x[j_indices]  # [N, H]
        pair_embeds = torch.cat([x_i, x_j], dim=1)  # [N, 2H]

        # Cosine similarity
        pos_mean_norm = F.normalize(pos_mean.unsqueeze(0), dim=1)  # [1, 2H]
        pair_embeds_norm = F.normalize(pair_embeds, dim=1)  # [N, 2H]
        similarities = torch.matmul(pair_embeds_norm, pos_mean_norm.T).squeeze(1)  # [N]

        # Top-K наиболее похожих отрицательных примеров
        k = min(3 * len(pos_set), len(similarities))
        topk_indices = torch.topk(similarities, k).indices

        selected_pairs = neg_pairs[topk_indices]
        return [tuple(pair.cpu().tolist()) for pair in selected_pairs]


    def save_pretrained(self, save_dir, tokenizer=None):
        """Сохраняет модель, конфигурацию и токенизатор в указанную директорию."""
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Сохраняем веса модели
        model_path = os.path.join(save_dir, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)
        
        # 2. Сохраняем конфигурацию модели
        config = {
            "model_type": "bert-ner-rel",
            "model_name": getattr(self.bert, "name_or_path", "custom"),
            "num_ner_labels": self.num_ner_labels,
            "num_rel_labels": len(RELATION_TYPES),  # Добавляем
            "bert_config": self.bert.config.to_diff_dict(),  # Более безопасный метод
            "model_config": {  # Добавляем специфичные для модели параметры
                "gat_hidden_size": 64,
                "gat_heads": 4
            }
        }
        
        config_path = os.path.join(save_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        # 3. Сохраняем токенизатор
        if tokenizer is not None:
            tokenizer.save_pretrained(save_dir)
        
    @classmethod
    def from_pretrained(cls, model_dir, device="cuda"):
        """Загружает модель из указанной директории."""
        try:
            device = torch.device(device)
            
            # 1. Загружаем конфигурацию
            config_path = os.path.join(model_dir, "config.json")
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found at {config_path}")
                
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            
            # 2. Инициализируем BERT
            if "bert_config" not in config:
                raise ValueError("Invalid config: missing bert_config")
                
            bert_config = BertConfig.from_dict(config["bert_config"])
            bert = AutoModel.from_pretrained(
                model_dir,
                config=bert_config,
                ignore_mismatched_sizes=True
            )
            
            # 3. Создаем экземпляр модели
            model = cls(
                model_name=config.get("model_name", "DeepPavlov/rubert-base-cased"),
                num_ner_labels=config.get("num_ner_labels", len(ENTITY_TYPES)*2+1),
                num_rel_labels=config.get("num_rel_labels", len(RELATION_TYPES))
            ).to(device)
            
            # 4. Загружаем веса
            model_path = os.path.join(model_dir, "pytorch_model.bin")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model weights not found at {model_path}")
                
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            
            # 5. Загружаем BERT
            model.bert = bert.to(device)
            
            model.eval()
            return model
            
        except Exception as e:
            raise RuntimeError(f"Error loading model from {model_dir}: {str(e)}")


SYMMETRIC_RELATIONS = {'SIBLING', 'SPOUSE', 'RELATIVE'}
class NERELDataset(Dataset):
    def __init__(self, data_dir, tokenizer, max_length=512, include_offsets=False):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_offsets = include_offsets
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


    def _find_best_span(self, entity_text, text, approx_start):
        # Ищем все вхождения entity_text в тексте
        matches = [
            (m.start(), m.end())
            for m in regex.finditer(regex.escape(entity_text), text, overlapped=True)
        ]
        if not matches:
            return None

        # Из всех совпадений выбираем ближайшее к приблизительному `start`
        return min(matches, key=lambda span: abs(span[0] - approx_start))
    
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

                    if entity_type not in ENTITY_TYPES:
                        print(f"Неизвестный тип сущности: {entity_type}")
                        continue
                    try:
                        start = int(type_and_span[1])
                        end = int(type_and_span[-1])
                    except ValueError:
                        continue

                    entity_text = parts[2]

                    extracted_text = text[start:end]
                    if extracted_text != entity_text:
                        print(f"[DEBUG] Misalignment detected:")
                        print(f"  entity_id: {entity_id}")
                        print(f"  expected: '{entity_text}'")
                        print(f"  found:    '{extracted_text}'")
                        print(f"  raw span: '{text[start:end]}'")
                        print(f"  context:  '{text[start - 20:end + 20].replace('\\n', '⏎')}'")
                        best_span = self._find_best_span(entity_text, text, start)
                        if best_span:
                            start, end = best_span
                        else:
                            print(f"[WARN] Entity alignment failed: Entity: '{entity_text}' ({entity_type}), "
                                  f"Span: {start}-{end}, Text: '{text[start - 10:end + 10]}'")
                            continue

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
                    parts = line.split('\t')
                    if len(parts) < 2:
                        continue
                    
                    rel_info = parts[1].split()
                    if len(rel_info) < 3:
                        continue
                    
                    rel_type = rel_info[0]
                    arg1 = rel_info[1].split(':')[1] if ':' in rel_info[1] else None
                    arg2 = rel_info[2].split(':')[1] if ':' in rel_info[2] else None

                    if arg1 and arg2 and arg1 in entity_map and arg2 in entity_map:
                        relations.append({'type': rel_type, 'arg1': arg1, 'arg2': arg2})
        return entities, relations
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample['text']
        text = unicodedata.normalize("NFC", text.replace('\u00A0', ' '))
        # Tokenize with subword information
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_offsets_mapping=True,
            return_tensors='pt'
        )

        ner_labels = torch.zeros(self.max_length, dtype=torch.long)
        offset_mapping = encoding['offset_mapping'][0]
        token_entities = []

        # Align entities with tokenization
        for entity in sample['entities']:
            matched_tokens = []
            for i, (start, end) in enumerate(offset_mapping):
                if start == end:
                    continue  # спецтокены
                if start <= entity['end'] and end >= entity['start']:
                    matched_tokens.append(i)
            if not matched_tokens:
                print(f"[WARN] Entity alignment failed: Entity: '{entity['text']}' ({entity['type']}), "
                      f"Span: {entity['start']}-{entity['end']}, "
                      f"Text segment: '{text[entity['start']:entity['end']]}'")
                continue

            ent_type_id = ENTITY_TYPES[entity['type']]
            b_label = ent_type_id * 2 - 1
            i_label = ent_type_id * 2
            ner_labels[matched_tokens[0]] = b_label
            for idx in matched_tokens[1:]:
                ner_labels[idx] = i_label
            token_entities.append({
                'start': matched_tokens[0],
                'end': matched_tokens[-1],
                'type': entity['type'],
                'id': entity['id']
            })

        rel_data = {
            'entities': token_entities,
            'pairs': [],
            'labels': []
        }
        
        token_entity_id_to_idx = {e['id']: i for i, e in enumerate(token_entities)}
        used_pairs = set()

        # Положительные примеры
        for relation in sample['relations']:
            if relation['type'] not in RELATION_TYPES:
                continue
            idx1 = token_entity_id_to_idx.get(relation['arg1'], -1)
            idx2 = token_entity_id_to_idx.get(relation['arg2'], -1)
            if idx1 == -1 or idx2 == -1:
                continue
            if relation['type'] in SYMMETRIC_RELATIONS:
                idx1, idx2 = sorted([idx1, idx2])
            pair = (idx1, idx2)
            if pair not in used_pairs:
                rel_data['pairs'].append(pair)
                rel_data['labels'].append(RELATION_TYPES[relation['type']])
                used_pairs.add(pair)

        # Отрицательные примеры
        num_entities = len(token_entities)
        for i in range(num_entities):
            for j in range(num_entities):
                if i == j:
                    continue
                pair = (min(i, j), max(i, j))
                if (i, j) in used_pairs:
                    continue
                rel_data['pairs'].append(pair)
                rel_data['labels'].append(0)  # 0 — нет отношения
                used_pairs.add(pair)

        output = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'ner_labels': ner_labels,
            'rel_data': rel_data,
            'text': text,
            'offset_mapping': encoding['offset_mapping'].squeeze(0)
        }

        return output

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
            'labels': torch.tensor(item['rel_data']['labels'], dtype=torch.long) if item['rel_data']['labels'] else torch.zeros(0, dtype=torch.long),
            'rel_types': [RELATION_TYPES_INV.get(l, 'UNK') for l in item['rel_data']['labels']] if item['rel_data']['labels'] else []
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
    entity_types = [etype for etype, _ in sorted(ENTITY_TYPES.items(), key=lambda x: x[1])]
    relation_types = [rtype for rtype, _ in sorted(RELATION_TYPES.items(), key=lambda x: x[1])]
    tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
    model = NERRelationModel(relation_types=relation_types, entity_types=entity_types).to(device)

    # Загрузка данных
    train_dataset = NERELDataset("NEREL/NEREL-v1.1/train", tokenizer)

    # Create weighted sampler to balance relation examples
    sample_weights = []
    for sample in train_dataset:
        has_relations = len(sample['rel_data']['labels']) > 0
        sample_weights.append(3.0 if has_relations else 1.0)
    
    sampler = WeightedRandomSampler(sample_weights, len(train_dataset), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=16, collate_fn=collate_fn, sampler=sampler)

    # Optimizer with different learning rates
    optimizer = AdamW([
    {'params': model.bert.parameters(), 'lr': 1e-5},
    {'params': model.ner_classifier.parameters(), 'lr': 1e-4},
    {'params': model.crf.parameters(), 'lr': 1e-4},
    {'params': model.gat1.parameters(), 'lr': 3e-4},
    {'params': model.gat2.parameters(), 'lr': 3e-4},
    {'params': model.rel_classifier.parameters(), 'lr': 5e-4}
    ], weight_decay=1e-5)
    
    # Training loop
    for epoch in range(2):
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

            print(f"[DEBUG] outputs keys: {outputs.keys()}")

            if outputs['loss'] is None:
                print(f"[WARN] Skipping batch due to missing loss")
                continue

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

            # NEW: Relation metrics calculation
            if 'rel_logits' in outputs:
                rel_logits = outputs['rel_logits']  # shape: (B, N_pairs, num_rel_types)
                for i, rel_sample in enumerate(batch['rel_data']):
                    if 'labels' in rel_sample and len(rel_sample['labels']) > 0:
                        # Get predictions for valid pairs
                        valid_pairs = min(rel_logits[i].shape[0], len(rel_sample['labels']))
                        if valid_pairs == 0:
                            continue

                        logits = rel_logits[i][:valid_pairs]  # shape: (valid_pairs, num_rel_types)
                        preds = logits.argmax(dim=1)  # predicted class per pair
                        targets = torch.tensor(rel_sample['labels'][:valid_pairs], device=device)

                        rel_correct += (preds == targets).sum().item()
                        rel_total += valid_pairs

        # Evaluation metrics
        ner_acc = ner_correct / ner_total if ner_total > 0 else 0
        rel_acc = rel_correct / rel_total if rel_total > 0 else 0
        
        print(f"\nEpoch {epoch+1} Results:")
        print(f"Loss: {epoch_loss/len(train_loader):.4f}")
        print(f"NER Accuracy: {ner_acc:.2%} ({ner_correct}/{ner_total})")
        print(f"Relation Accuracy: {rel_acc:.2%} ({rel_correct}/{rel_total})")

    save_dir = "saved_model"
    tokenizer.save_pretrained(save_dir)
    model.save_pretrained(save_dir)
    print(f"Model saved to {save_dir}")
    
    return model, tokenizer

def format_relation(arg1_text, arg2_text, rel_type, confidence):
    conf_str = f"{confidence:.2f}"
    return f"🔗 {colored(arg1_text, 'white', attrs=['bold'])} --{rel_type}({conf_str})--> {colored(arg2_text, 'white', attrs=['bold'])}"

def visualize_prediction_colored(prediction):
    text = prediction['text']
    entities = sorted(prediction['entities'], key=lambda e: e['start_char'])
    relations = prediction['relations']

    result_text = ""
    last_pos = 0

    for ent in entities:
        # Add raw text before this entity
        result_text += text[last_pos:ent['start_char']]

        # Color entity
        color = ENTITY_COLORS.get(ent['type'], 'white')
        entity_str = colored(ent['text'], color, attrs=["bold"])
        result_text += f"[{entity_str}]({ent['type']})"

        last_pos = ent['end_char']

    result_text += text[last_pos:]

    # Format relations
    rel_lines = []
    for rel in relations:
        rel_lines.append(format_relation(rel['arg1']['text'], rel['arg2']['text'], rel['type'], rel['confidence']))

    return "\n" + "\n".join(rel_lines) + "\n\n" + result_text

def predict(text, model, tokenizer, device="cuda", relation_threshold=None):
    # Tokenize input with offset mapping
    relation_threshold = {**RELATION_THRESHOLDS, **(relation_threshold or {})}
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
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0], skip_special_tokens=True)

    # Extract entities
    entities = []
    current_entity = None
    entity_id = 0
    
    for i, (token, pred) in enumerate(zip(tokens, ner_preds)):
        if token in ['[CLS]', '[SEP]', '[PAD]']:
            if current_entity:
                entities.append(current_entity)
                current_entity = None
            continue

        # Get the character offsets for this token
        token_start, token_end = offset_mapping[i]
        
        # Handle entity extraction
        if pred % 2 == 1:  # Beginning of entity (B- tag)
            if current_entity:
                entities.append(current_entity)
            
            entity_type = None
            if pred == 1: entity_type = "PERSON"
            elif pred == 3: entity_type = "PROFESSION"
            elif pred == 5: entity_type = "ORGANIZATION"
            elif pred == 7: entity_type = "FAMILY"
            elif pred == 9: entity_type = "LOCATION"
            
            if entity_type:
                current_entity = {
                    'id': f"T{entity_id}",
                    'type': entity_type,
                    'start': i,
                    'end': i,
                    'start_char': token_start,
                    'end_char': token_end,
                    'token_ids': [i]
                }
                entity_id += 1
                
        elif pred % 2 == 0 and pred != 0:  # Inside of entity (I- tag)
            if current_entity:
                # Check if this continues the current entity
                expected_type = None
                if pred == 2: expected_type = "PERSON"
                elif pred == 4: expected_type = "PROFESSION"
                elif pred == 6: expected_type = "ORGANIZATION"
                elif pred == 8: expected_type = "FAMILY"
                elif pred == 10: expected_type = "LOCATION"
                
                if expected_type and current_entity['type'] == expected_type:
                    current_entity['end'] = i
                    current_entity['end_char'] = token_end
                    current_entity['token_ids'].append(i)
                else:
                    # Type mismatch, save current and start new
                    entities.append(current_entity)
                    current_entity = None
        else:  # O (outside)
            if current_entity:
                entities.append(current_entity)
                current_entity = None
    
    # Add the last entity if exists
    if current_entity:
        entities.append(current_entity)

    # Now get the actual text for each entity
    for entity in entities:
        entity['text'] = text[entity['start_char']:entity['end_char']]

    if len(entities) < 2:  # Не может быть отношений, если меньше 2 сущностей
        return {
            'text': text,
            'entities': entities,
            'relations': []
        }

    # Extract relations
    relations = []
    entity_map = {e['id']: e for e in entities}

    if len(entities) >= 2:
        sequence_output = model.bert(input_ids, attention_mask).last_hidden_state

        # Create entity embeddings
        entity_embeddings = []
        for e in entities:
            # Get all token embeddings for this entity
            token_embeddings = sequence_output[0, e['token_ids']]
            # Average them
            entity_embed = token_embeddings.mean(dim=0)
            entity_embeddings.append(entity_embed)
        
        entity_embeddings = torch.stack(entity_embeddings).to(device)

        # Применяем GAT слои (как в forward)
        edge_pairs = [[i, j] for i in range(len(entities)) for j in range(len(entities)) if i != j]
        edge_index = torch.tensor(edge_pairs).t().to(device)

        x = model.gat1(entity_embeddings, edge_index)
        x = model.norm1(x)
        x = F.elu(x)
        x = model.gat2(x, edge_index)
        x = model.norm2(x)
        x = F.elu(x)

        # Предсказываем отношения с единым классификатором
        relations = []
        cls_token = sequence_output[:, 0, :].squeeze(0)

        for rel_type, rel_type_idx in RELATION_TYPES.items():
            valid_combinations = VALID_COMB.get(rel_type, [])

            for i, e1 in enumerate(entities):
                for j, e2 in enumerate(entities):
                    if i != j and (e1['type'], e2['type']) in valid_combinations:
                        # Для FOUNDED_BY меняем направление
                        if rel_type == 'FOUNDED_BY':
                            src, tgt = j, i
                        else:
                            src, tgt = i, j

                        rel_type_tensor = model.rel_type_emb(torch.tensor(rel_type_idx, device=device))
                        pair_vec = torch.cat([x[src], x[tgt], rel_type_tensor, cls_token])
                        logit = model.rel_classifier(pair_vec)
                        prob = torch.sigmoid(logit).item()

                        if prob > relation_threshold.get(rel_type, 0.5):
                            relations.append({
                                'type': rel_type,
                                'arg1_id': entities[src]['id'],
                                'arg2_id': entities[tgt]['id'],
                                'arg1': entities[src],
                                'arg2': entities[tgt],
                                'confidence': prob
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
        "Эмир Катара встретится с членами королевской семьи.Эмир Катара шейх Хамад бен Халиф Аль Тани встретится в понедельник с членами королевской семьи и высокопоставленными чиновниками страны на фоне слухов о том, что он намерен передать власть сыну — наследному принцу шейху Тамиму, передает агентство Рейтер со ссылкой на катарский телеканал 'Аль-Джазира'. 'Аль-Джазира', в свою очередь, ссылается на 'надежный источник в Катаре', но не приводит каких-либо деталей. Ранее в этом месяце в дипломатических кругах появились слухи, что эмир Катара, которому сейчас 61 год, рассматривает возможность передачи власти 33-летнему наследному принцу, отмечает агентство. При этом также предполагается, что в отставку подаст влиятельный премьер-министр и министр иностранных дел Катара шейх Хамад бен Джасем Аль Тани. По данным агентства, дипломаты западных и арабских стран оценивают такое решение как попытку осторожной передачи власти более молодому поколению правителей. Ранее новостной портал 'Элаф' отмечал, что перемены во властных структурах Катара могут произойти уже в конце июня. Согласно информации агентства Франс Пресс, Тамим бен Хамад Аль Тани родился в 1980 году и является вторым сыном эмира и его второй жены Мозы бинт Нассер. Наследный принц занимает офицерский пост в катарской армии, а также является главой Олимпийского комитета страны.",
        "Айрат Мурзагалиев, заместитель начальника управления президента РФ, встретился с главой администрации Уфы.",
        "Иван Петров работает программистом в компании Яндекс.",
        "Доктор Сидоров принял пациентку Ковалеву в городской больнице.",
        "Директор сводного экономического департамента Банка России Надежда Иванова назначена также на должность заместителя председателя ЦБ, сообщил в четверг регулятор.",
        "Дмитрий работает в организации 'ЭкоФарм'",
        "Компания 'Технологии будущего' является частью крупной корпорации, расположенной в Санкт-Петербурге",
        "Анна занимает должность главного врача в больнице 'Здоровье'."
    ]
    
    for text in test_texts:
        print("\n" + "="*80)
        print(f"Processing text: '{text}'")
        result = predict(text, model, tokenizer)
        print(visualize_prediction_colored(result))

        # print("\nEntities:")
        # for e in result['entities']:
        #     print(f"{e['type']}: {e['text']}")
        # print("\nRelations:")
        # for r in result['relations']:
        #     print(f"{r['type']}: {r['arg1']['text']} -> {r['arg2']['text']} (conf: {r['confidence']:.2f})")

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

