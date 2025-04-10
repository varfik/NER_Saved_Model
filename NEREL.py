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

RELATION_TYPES = {
    'WORKS_AS': 1,
    'MEMBER_OF': 2,
    'FOUNDED_BY': 3,
    'SPOUSE': 4,
    'PARENT_OF': 5,
    'SIBLING': 6

}

RELATION_TYPES_INV = {v: k for k, v in RELATION_TYPES.items()}

class NERRelationModel(nn.Module):
    def __init__(self, model_name="DeepPavlov/rubert-base-cased", num_ner_labels=9, num_rel_labels=7):
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
        
        # Graph attention network components (GAT)
        # Improved GAT architecture
        self.gat1 = GATConv(
            self.bert.config.hidden_size, 
            128, 
            heads=4, 
            dropout=0.3,
            concat=True
        )
        self.norm1 = nn.LayerNorm(128 * 4)
        self.gat2 = GATConv(
            128 * 4, 
            64, 
            heads=1, 
            dropout=0.3,
            concat=False
        )
        self.norm2 = nn.LayerNorm(64)
        # Concatenate heads from first layer
    
         # Relation classifiers with type-specific architectures
        self.rel_classifiers = nn.ModuleDict({
            'WORKS_AS': self._build_relation_classifier(input_dim=64*2, hidden_dim=256),
            'MEMBER_OF': self._build_relation_classifier(input_dim=64*2, hidden_dim=256),
            'FOUNDED_BY': self._build_relation_classifier(input_dim=64*2, hidden_dim=256),
            'SPOUSE': self._build_symmetric_classifier(input_dim=64*2, hidden_dim=256),
            'PARENT_OF': self._build_relation_classifier(input_dim=64*2, hidden_dim=256),
            'SIBLING': self._build_symmetric_classifier(input_dim=64*2, hidden_dim=256)
        })

        self._init_weights()

    def _build_relation_classifier(self, input_dim, hidden_dim):
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )

    def _build_symmetric_classifier(self, input_dim, hidden_dim):
        """For symmetric relations like SPOUSE and SIBLING"""
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
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
                    print(f"Пропуск примера {batch_idx}: нет пар отношений")
                    continue
                    
                # Get valid entities with type information
                valid_entities = [
                    e for e in sample['entities'] 
                    if isinstance(e, dict) and 'start' in e and 'end' in e and 'type' in e
                ]
                
                if len(valid_entities) < 2:
                    rint(f"Пропуск примера {batch_idx}: недостаточно сущностей ({len(valid_entities)})")
                    continue

                print(f"\nОбработка примера {batch_idx}:")
                print(f"Сущности: {[(e['type'], e['id']) for e in valid_entities]}")
                print(f"Пары отношений: {sample['pairs']}")
                print(f"Метки отношений: {sample['labels']}") 
                # Create entity embeddings
                entity_embeddings = []
                entity_types = []
                for e in valid_entities:
                    start = min(e['start'], sequence_output.size(1)-1)
                    end = min(e['end'], sequence_output.size(1)-1)
                    entity_embed = sequence_output[batch_idx, start:end+1].mean(dim=0)
                    entity_embeddings.append(entity_embed)
                    entity_types.append(e['type'])
                
                # Build complete graph
                edge_index = torch.tensor([
                    [i, j] for i in range(len(valid_entities)) 
                    for j in range(len(valid_entities)) if i != j
                ], dtype=torch.long).t().contiguous().to(device)
                
                x = torch.stack(entity_embeddings).to(device)
                
                # Apply improved GAT with normalization
                x = self.gat1(x, edge_index)
                x = self.norm1(x)
                x = F.elu(x)
                x = F.dropout(x, p=0.3, training=self.training)
                x = self.gat2(x, edge_index)
                x = self.norm2(x)
                x = F.elu(x)

                # Create entity index map
                entity_indices = {e['id']: i for i, e in enumerate(valid_entities)}
                
                # Process each relation type separately
                for rel_type in RELATION_TYPES:
                    rel_probs[rel_type] = []
                    rel_targets[rel_type] = []
                    pos_indices = set()
                    
                    # Collect positive examples for this relation type
                    for (e1_idx, e2_idx), label in zip(sample['pairs'], sample['labels']):
                        if label == RELATION_TYPES[rel_type]:
                            if e1_idx in entity_indices and e2_idx in entity_indices:
                                i = entity_indices[e1_idx]
                                j = entity_indices[e2_idx]
                                e1_type = valid_entities[i]['type']
                                e2_type = valid_entities[j]['type']

                                if self._is_valid_pair(e1_type, e2_type, rel_type):
                                    # For FOUNDED_BY, we need to reverse the direction
                                    if rel_type == 'FOUNDED_BY':
                                        i, j = j, i
                                    
                                    pair_features = torch.cat([x[i], x[j]])
                                    rel_probs[rel_type].append(self.rel_classifiers[rel_type](pair_features))
                                    rel_targets[rel_type].append(1.0)
                                    pos_indices.add((i, j))
                    print(f"Тип отношения {rel_type}: найдено {pos_count} положительных примеров")
                    
                    # Generate negative examples for this relation type
                    neg_pairs = self._generate_negative_examples(
                        entity_embeddings=x, 
                        entity_types=entity_types, 
                        rel_type=rel_type,
                        pos_indices=pos_indices,
                        ratio=1.0
                    )
                    
                    if neg_pairs:
                        neg_features, neg_targets = neg_pairs
                        rel_probs[rel_type].extend(neg_features)
                        rel_targets[rel_type].extend(neg_targets)

                        print(f"Добавлено {len(neg_targets)} отрицательных примеров для {rel_type}")
                    else:
                        print(f"Не удалось сгенерировать отрицательные примеры для {rel_type}")

                
                    # Calculate loss for this relation type
                    if rel_probs[rel_type]:
                        min_len = min(len(rel_probs[rel_type]), len(rel_targets[rel_type]))
                        probs_tensor = torch.cat(rel_probs[rel_type][:min_len]).view(-1)
                        targets_tensor = torch.tensor(rel_targets[rel_type][:min_len], dtype=torch.float, device=device)

                        # Adjust pos_weight based on class imbalance
                        pos_weight = torch.tensor([
                            max(1.0, len(targets_tensor) / (sum(targets_tensor) + 1e-6)) * 5.0  # Увеличенный коэффициент
                        ], device=device)
                        
                        rel_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(
                            probs_tensor, targets_tensor)
                        total_loss += rel_loss * 1.0  # Weight relation loss

                        # Отладочная информация
                        preds = (torch.sigmoid(probs_tensor) > 0.5).long()
                        correct = (preds == targets_tensor.long()).sum().item()
                        accuracy = correct / len(targets_tensor)
                        
                        print(f"Отношение {rel_type}: loss={rel_loss.item():.4f}, accuracy={accuracy:.2%}, "
                            f"pos/neg={sum(targets_tensor)}/{len(targets_tensor)-sum(targets_tensor)}")
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
        return (e1_type, e2_type) in relation_rules.get(rel_type, [])

    def _generate_negative_examples(self, entity_embeddings, entity_types, rel_type,  pos_indices=None, ratio=0.5):
        """Generate valid negative examples for specific relation type"""
        device = entity_embeddings.device
        neg_probs = []
        neg_targets = []
        
        if pos_indices is None:
            pos_indices = set()
        
        # Collect all possible valid pairs for this relation type
        possible_pairs = []
        for i, e1_type in enumerate(entity_types):
            for j, e2_type in enumerate(entity_types):
                if i == j:
                    continue
                if (i, j) in pos_indices or (j, i) in pos_indices:
                    continue
                if self._is_valid_pair(e1_type, e2_type, rel_type):
                    possible_pairs.append((i, j))

        print(f"Всего возможных валидный пар: {len(possible_pairs)}")
        
        # For symmetric relations, we need to consider both directions
        if rel_type in ['SPOUSE', 'SIBLING']:
            possible_pairs = list(set(
                (min(i, j), max(i, j)) 
                for i, j in possible_pairs
            ))
            print(f"Уникальные пары после обработки симметричных отношений: {len(possible_pairs)}")


         # 3. Стратегия выборки:
        # - 50% случайных пар
        # - 30% "сложных" пар (близкие по эмбеддингам)
        # - 20% пар с похожими типами сущностей
        
        num_random = max(1, int(len(possible_pairs) * ratio * 0.5))
        num_hard = max(1, int(len(possible_pairs) * ratio * 0.3))
        num_type = max(1, int(len(possible_pairs) * ratio * 0.2))

        # 3.1 Случайные отрицательные примеры
        if len(possible_pairs) > num_random:
            sampled_random = random.sample(possible_pairs, num_random)
        else:
            sampled_random = possible_pairs
        
        print(f"Выбрано случайных пар: {len(sampled_random)}")

        # 3.2 "Сложные" отрицательные примеры (близкие в пространстве эмбеддингов)
        if len(possible_pairs) > 1:
            # Вычисляем попарные расстояния
            with torch.no_grad():
                all_embs = entity_embeddings.cpu().numpy()
                distances = np.zeros((len(all_embs), len(all_embs)))
                
                for i in range(len(all_embs)):
                    for j in range(len(all_embs)):
                        if i != j:
                            distances[i,j] = np.linalg.norm(all_embs[i] - all_embs[j])
            
            # Выбираем пары с минимальными расстояниями
            flat_indices = np.argpartition(distances.ravel(), num_hard)[:num_hard]
            hard_pairs = [np.unravel_index(i, distances.shape) for i in flat_indices]
            hard_pairs = [(i,j) for i,j in hard_pairs if i != j and (i,j) in possible_pairs]
            
            if len(hard_pairs) > num_hard:
                hard_pairs = random.sample(hard_pairs, num_hard)
        else:
            hard_pairs = []
        
        print(f"Выбрано сложных пар: {len(hard_pairs)}")

        # 3.3 Пары с похожими типами сущностей
        type_pairs = []
        type_counts = defaultdict(int)
        
        for i, j in possible_pairs:
            key = (entity_types[i], entity_types[j])
            if type_counts[key] < num_type // len(ENTITY_TYPES)**2:
                type_pairs.append((i,j))
                type_counts[key] += 1
        
        if len(type_pairs) > num_type:
            type_pairs = random.sample(type_pairs, num_type)
        
        print(f"Выбрано пар по типам: {len(type_pairs)}")

        # Объединяем все стратегии
        all_neg_pairs = list(set(sampled_random + hard_pairs + type_pairs))
        print(f"Всего отрицательных пар после объединения: {len(all_neg_pairs)}")

        # 4. Создаем примеры
        for i, j in all_neg_pairs:
            # Для FOUNDED_BY нужно обратить направление
            if rel_type == 'FOUNDED_BY':
                i, j = j, i
            
            pair_features = torch.cat([entity_embeddings[i], entity_embeddings[j]])
            neg_probs.append(self.rel_classifiers[rel_type](pair_features))
            neg_targets.append(0.0)
            
            # Отладочная информация
            if random.random() < 0.1:  # Выводим 10% примеров для проверки
                print(f"Отрицательный пример: {entity_types[i]} -> {entity_types[j]}")

        if neg_probs:
            print(f"Сгенерировано отрицательных примеров для {rel_type}: {len(neg_probs)}")
            return torch.stack(neg_probs).view(-1, 1), torch.tensor(neg_targets, device=device)
        
        print(f"Не удалось сгенерировать отрицательные примеры для {rel_type}")
        return None

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
            bert = AutoModel.from_config(bert_config)
            
            # 3. Создаем экземпляр модели
            model = cls(
                model_name=config.get("model_name", "DeepPavlov/rubert-base-cased"),
                num_ner_labels=config.get("num_ner_labels", 9),
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
            model.bert.load_state_dict(state_dict, strict=False)  # Загружаем только совпадающие ключи
            
            model.eval()
            return model
            
        except Exception as e:
            raise RuntimeError(f"Error loading model from {model_dir}: {str(e)}")

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
                    
                    e1_type = entity_map[arg1]['type']
                    e2_type = entity_map[arg2]['type']
                
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
                # Set BIO labels
                if entity['type'] == 'PERSON':
                    ner_labels[start_token] = 1  # B-PER
                    ner_labels[start_token+1:end_token+1] = 2  # I-PER
                elif entity['type'] == 'PROFESSION':
                    ner_labels[start_token] = 3  # B-PROF
                    ner_labels[start_token+1:end_token+1] = 4  # I-PROF
                elif entity['type'] == 'ORGANIZATION':
                    ner_labels[start_token] = 5  # B-ORG
                    ner_labels[start_token+1:end_token+1] = 6  # I-ORG
                elif entity['type'] == 'FAMILY':
                    ner_labels[start_token] = 7  # B-FAM
                    ner_labels[start_token+1:end_token+1] = 8  # I-FAM

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
    {'params': model.bert.parameters(), 'lr': 3e-5},
    {'params': model.ner_classifier.parameters(), 'lr': 5e-5},
    {'params': model.crf.parameters(), 'lr': 5e-5},
    {'params': model.relation_head.parameters(), 'lr': 1e-4}
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
                        # Убедимся, что probs - это тензор
                        if isinstance(probs, list):
                            probs = torch.cat([p.view(-1) for p in probs if p is not None]) if len(probs) > 0 else torch.tensor([], device=device)
                        
                        if len(probs) > 0:
                            preds = (torch.sigmoid(probs) > 0.5).long()
                            
                            # Собираем метки для этого типа отношений
                            targets = []
                            for item in batch['rel_data']:
                                if 'labels' in item and len(item['labels']) > 0:
                                    # Фильтруем метки для текущего типа отношений
                                    rel_labels = [l for l, t in zip(item['labels'], item.get('rel_types', [])) if t == rel_type]
                                    targets.extend(rel_labels)
                            
                            if len(targets) > 0:
                                # Обрезаем до минимальной длины
                                min_len = min(len(preds), len(targets))
                                rel_correct += (preds[:min_len] == torch.tensor(targets[:min_len], device=device)).sum().item()
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

def predict(text, model, tokenizer, device="cuda", relation_threshold=0.3):
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
        
        # Build complete graph
        edge_index = torch.tensor([
            [i, j] for i in range(len(entities)) 
            for j in range(len(entities)) if i != j
        ], dtype=torch.long).t().contiguous().to(device)

        # Apply GAT
        x = model.gat1(entity_embeddings, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.3, training=False)
        x = model.gat2(x, edge_index)
        x = F.elu(x)
        
        # Predict all possible relations
        for rel_type in RELATION_TYPES:
            for i, e1 in enumerate(entities):
                for j, e2 in enumerate(entities):
                    if i != j and model._is_valid_pair(e1['type'], e2['type'], rel_type):
                        # For FOUNDED_BY we need to reverse the direction
                        if rel_type == 'FOUNDED_BY':
                            src, tgt = j, i
                        else:
                            src, tgt = i, j
                            
                        pair_features = torch.cat([x[src], x[tgt]])
                        logit = model.rel_classifiers[rel_type](pair_features)
                        prob = torch.sigmoid(logit).item()
                        
                        if prob > relation_threshold:
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

