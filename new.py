import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from transformers import AutoModel, AutoTokenizer, AutoConfig, BertConfig
from torchcrf import CRF
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import os
import json
from collections import defaultdict
from tqdm.auto import tqdm
from torch.optim import AdamW
import random
from typing import Dict, List, Tuple, Optional

# Конфигурация модели
class ModelConfig:
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

    RELATION_RULES = {
        'WORKS_AS': [('PERSON', 'PROFESSION')],
        'MEMBER_OF': [('PERSON', 'ORGANIZATION')],
        'FOUNDED_BY': [('ORGANIZATION', 'PERSON')],
        'SPOUSE': [('PERSON', 'PERSON')],
        'PARENT_OF': [('PERSON', 'PERSON'), ('PERSON', 'FAMILY')],
        'SIBLING': [('PERSON', 'PERSON')]
    }

    NER_LABEL_MAP = {
        'PERSON': (1, 2),       # B-PER, I-PER
        'PROFESSION': (3, 4),    # B-PROF, I-PROF
        'ORGANIZATION': (5, 6),  # B-ORG, I-ORG
        'FAMILY': (7, 8)        # B-FAM, I-FAM
    }

    def __init__(self):
        self.num_ner_labels = 9  # O + 8 entity labels
        self.num_rel_labels = len(self.RELATION_TYPES)

class NERRelationModel(nn.Module):
    def __init__(self, model_name="DeepPavlov/rubert-base-cased", config=None):
        super().__init__()
        self.config = config if config else ModelConfig()
        
        # BERT encoder
        self.bert = AutoModel.from_pretrained(model_name)
        
        # NER Head with CRF
        self.ner_classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.config.num_ner_labels)
        )
        self.crf = CRF(self.config.num_ner_labels, batch_first=True)
        
        # Graph attention network
        self.gat1 = GATConv(self.bert.config.hidden_size, 128, heads=4, dropout=0.3)
        self.gat2 = GATConv(128*4, 64, heads=1, dropout=0.3)
        
        # Улучшенный классификатор отношений
        self.rel_feature_extractor = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size * 3, 512),  # Используем в 3 раза больше признаков
            nn.LayerNorm(512),
            nn.LeakyReLU(),
            nn.Dropout(0.4)
        )

        # Relation classifiers
        self.rel_classifiers = nn.ModuleDict({
            rel_type: nn.Sequential(
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.LeakyReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 1)
            ) for rel_type in self.config.RELATION_TYPES
        })
        self._init_weights()

    def _init_weights(self):
        for module in [self.ner_classifier, *self.rel_classifiers.values()]:
            if isinstance(module, nn.Sequential):
                for submodule in module:
                    if isinstance(submodule, nn.Linear):
                        nn.init.xavier_uniform_(submodule.weight)
                        nn.init.constant_(submodule.bias, 0)

    def forward(self, input_ids, attention_mask, ner_labels=None, rel_data=None):
        outputs = self._forward_bert(input_ids, attention_mask)
        total_loss = 0
        results = {'ner_logits': outputs['ner_logits']}
        
        # NER loss
        if ner_labels is not None:
            mask = attention_mask.bool()
            ner_loss = -self.crf(outputs['ner_logits'], ner_labels, mask=mask, reduction='mean')
            total_loss += ner_loss

        # Relation extraction
        if rel_data:
            rel_results = self._process_relations(
                outputs['sequence_output'], 
                rel_data,
                attention_mask.device
            )
            results.update(rel_results)
            total_loss += rel_results.get('rel_loss', 0)

        results['loss'] = total_loss if total_loss != 0 else None
        return results

    def _forward_bert(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        return {
            'sequence_output': outputs.last_hidden_state,
            'ner_logits': self.ner_classifier(outputs.last_hidden_state)
        }

    def _process_relations(self, sequence_output, rel_data, device):
        rel_probs = {}
        rel_loss = 0
        
        for rel_type in self.config.RELATION_TYPES:
            batch_rel_probs = []
            batch_rel_targets = []
            
            for batch_idx, sample in enumerate(rel_data):
                if not sample.get('pairs', []):
                    continue
                
                # Process entities and build graph
                graph_data = self._build_entity_graph(
                    sequence_output[batch_idx],
                    sample['entities'],
                    rel_type,
                    device
                )
                
                if not graph_data:
                    continue
                
                x, entity_indices = graph_data
                
                # Process relation pairs
                pair_results = self._process_relation_pairs(
                    x, sample, entity_indices, rel_type, device
                )
                
                if pair_results:
                    batch_rel_probs.append(pair_results['probs'])
                    batch_rel_targets.append(pair_results['targets'])

            if batch_rel_probs:
                rel_probs[rel_type] = torch.cat(batch_rel_probs)
                rel_targets = torch.cat(batch_rel_targets)
                
                pos_weight = torch.tensor([2.0], device=device)
                rel_loss += nn.BCEWithLogitsLoss(pos_weight=pos_weight)(
                    rel_probs[rel_type], rel_targets)

        return {
            'rel_probs': rel_probs,
            'rel_loss': rel_loss if rel_probs else 0
        }

    def _build_entity_graph(self, sequence_output, entities, rel_type, device):
        # Filter valid entities
        valid_entities = [
            e for e in entities 
            if e['start'] <= e['end'] and 
            e['type'] in self.config.ENTITY_TYPES
        ]
        
        if len(valid_entities) < 2:
            return None

        # Create entity embeddings
        entity_embeddings = []
        for e in valid_entities:
            start = min(e['start'], sequence_output.size(0)-1)
            end = min(e['end'], sequence_output.size(0)-1)
            entity_embed = sequence_output[start:end+1].mean(dim=0)
            entity_embeddings.append(entity_embed)

        # Build graph edges
        edge_index = []
        for i, e1 in enumerate(valid_entities):
            for j, e2 in enumerate(valid_entities):
                if i != j and self._is_valid_pair(e1['type'], e2['type'], rel_type):
                    edge_index.append([i, j])
        
        if not edge_index:
            return None

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)
        x = torch.stack(entity_embeddings)
        
        # Apply GAT
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.gat2(x, edge_index)
        
        # Create entity index mapping
        entity_indices = {e['id']: i for i, e in enumerate(valid_entities)}
        
        return x, entity_indices

    def _process_relation_pairs(self, x, sample, entity_indices, rel_type, device):
        current_probs = []
        current_targets = []

        # Добавляем контекстные признаки
        context_features = x.mean(dim=0)  # Общие признаки всего предложения
        
        for (e1_id, e2_id), label in zip(sample['pairs'], sample['labels']):
            if e1_id in entity_indices and e2_id in entity_indices:
                e1_idx = entity_indices[e1_id]
                e2_idx = entity_indices[e2_id]
                
            # Улучшенные признаки пары
            e1_features = x[e1_idx]
            e2_features = x[e2_idx]
            
            # Расстояние между сущностями (важная особенность!)
            distance = torch.abs(torch.tensor(e1_idx - e2_idx, dtype=torch.float, device=device))
            distance_feature = torch.log(distance + 1).unsqueeze(0)
            
            # Комбинированные признаки
            pair_features = torch.cat([
                e1_features, 
                e2_features, 
                e1_features * e2_features,  # Взаимодействие признаков
                context_features,
                distance_feature
            ])
            
            # Пропускаем через улучшенный экстрактор признаков
            pair_features = self.rel_feature_extractor(pair_features)
            
            current_probs.append(self.rel_classifiers[rel_type](pair_features))
            current_targets.append(1.0)
        
        # Add negative examples
        neg_probs, neg_targets = self._generate_negative_examples(
            x, [e['type'] for e in sample['entities']], rel_type, device)
        
        if current_probs or neg_probs.numel() > 0:
            all_probs = torch.cat(current_probs + [neg_probs]) if current_probs else neg_probs
            all_targets = torch.tensor(current_targets + neg_targets.tolist(), device=device) if current_targets else neg_targets
            return {
                'probs': all_probs.view(-1),
                'targets': all_targets.float()
            }
        return None

    def _is_valid_pair(self, e1_type, e2_type, rel_type):
        return (e1_type, e2_type) in self.config.RELATION_RULES.get(rel_type, [])

    def _generate_negative_examples(self, entity_embeddings, entity_types, rel_type, device, ratio=0.5):
        # Collect all possible valid pairs for this relation type
        possible_pairs = [
            (i, j) for i, e1_type in enumerate(entity_types)
            for j, e2_type in enumerate(entity_types)
            if i != j and self._is_valid_pair(e1_type, e2_type, rel_type)
        ]
        
        if not possible_pairs:
            return torch.tensor([], device=device), torch.tensor([], device=device)
        
        # Sample negative examples
        num_neg = max(1, int(len(possible_pairs) * ratio))
        sampled_pairs = random.sample(possible_pairs, min(num_neg, len(possible_pairs)))

        neg_probs = []
        for i, j in sampled_pairs:
            e1_features = entity_embeddings[i]
            e2_features = entity_embeddings[j]
            distance = torch.abs(torch.tensor(i - j, dtype=torch.float, device=device))
            distance_feature = torch.log(distance + 1).unsqueeze(0)
            
            pair_features = torch.cat([
                e1_features, 
                e2_features, 
                e1_features * e2_features,
                entity_embeddings.mean(dim=0),
                distance_feature
            ])
            
            pair_features = self.rel_feature_extractor(pair_features)
            neg_probs.append(self.rel_classifiers[rel_type](pair_features))
        if neg_probs:
            # Объединяем все отрицательные примеры
            neg_probs_tensor = torch.cat([p.view(-1) for p in neg_probs])
            neg_labels_tensor = torch.zeros(len(neg_probs), dtype=torch.float, device=device)
            return neg_probs_tensor, neg_labels_tensor
        return torch.tensor([], device=device), torch.tensor([], device=device)

    def save_pretrained(self, save_dir, tokenizer=None):
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model weights
        torch.save(self.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
        
        # Save config
        config = {
            "model_type": "bert",
            "model_name": self.bert.name_or_path,
            "num_ner_labels": self.config.num_ner_labels,
            "bert_config": self.bert.config.to_dict()
        }
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
        
        if tokenizer is not None:
            tokenizer.save_pretrained(save_dir)
    
    @classmethod
    def from_pretrained(cls, model_dir, device="cuda"):
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Initialize with config
        model_config = ModelConfig()
        model = cls(
            model_name=config["model_name"],
            config=model_config
        ).to(device)
        
        # Load BERT
        bert_config = BertConfig.from_dict(config["bert_config"])
        model.bert = AutoModel.from_config(bert_config).to(device)
        
        # Load weights
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
        entities, relations = [], defaultdict(list)
        entity_map = {}
        
        with open(ann_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith('T'):
                    entity = self._parse_entity_line(line, text)
                    if entity:
                        entities.append(entity)
                        entity_map[entity['id']] = entity
                
                elif line.startswith('R'):
                    relation = self._parse_relation_line(line, entity_map)
                    if relation:
                        relations[relation['type']].append(relation)
        
        return entities, relations

    def _parse_entity_line(self, line, text):
        parts = line.split('\t')
        if len(parts) < 3:
            return None

        entity_id = parts[0]
        type_and_span = parts[1].split()
        entity_type = type_and_span[0]

        if entity_type not in ModelConfig.ENTITY_TYPES:
            return None

        start = int(type_and_span[1])
        end = int(type_and_span[-1])
        entity_text = parts[2]
        
        # Verify entity span matches text
        if text[start:end] != entity_text:
            found_pos = text.find(entity_text)
            if found_pos != -1:
                start = found_pos
                end = found_pos + len(entity_text)
        
        return {
            'id': entity_id,
            'type': entity_type,
            'start': start,
            'end': end,
            'text': entity_text
        }

    def _parse_relation_line(self, line, entity_map):
        parts = line.strip().split('\t')
        if len(parts) < 2:
            return None
        
        rel_type = parts[1].split()[0]
        arg1 = parts[1].split()[1].split(':')[1]
        arg2 = parts[1].split()[2].split(':')[1]
    
        if arg1 not in entity_map or arg2 not in entity_map:
            return None
        
        e1_type = entity_map[arg1]['type']
        e2_type = entity_map[arg2]['type']
    
        return {
            'type': rel_type,
            'arg1': arg1,
            'arg2': arg2,
            'e1_type': e1_type,
            'e2_type': e2_type
        }
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        encoding = self._tokenize_text(sample['text'])
        ner_labels = self._process_ner_labels(encoding, sample['entities'])
        rel_data = self._process_relations(sample, encoding)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'ner_labels': ner_labels,
            'rel_data': rel_data,
            'text': sample['text'],
            'offset_mapping': encoding['offset_mapping'].squeeze(0)
        }

    def _tokenize_text(self, text):
        return self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_offsets_mapping=True,
            return_tensors='pt'
        )

    def _process_ner_labels(self, encoding, entities):
        ner_labels = torch.zeros(self.max_length, dtype=torch.long)
        token_entities = []

        for entity in entities:
            start_token, end_token = self._find_entity_token_span(encoding, entity)
            
            if start_token is not None and end_token is not None:
                b_label, i_label = ModelConfig.NER_LABEL_MAP[entity['type']]
                ner_labels[start_token] = b_label
                ner_labels[start_token+1:end_token+1] = i_label

                token_entities.append({
                    'start': start_token,
                    'end': end_token,
                    'type': entity['type'],
                    'id': entity['id']
                })

        return ner_labels

    def _find_entity_token_span(self, encoding, entity):
        start_token = end_token = None
        for i, (start, end) in enumerate(encoding['offset_mapping'][0]):
            if start <= entity['start'] < end and start_token is None:
                start_token = i
            if start < entity['end'] <= end and end_token is None:
                end_token = i
            if start >= entity['end']:
                break
        return start_token, end_token

    def _process_relations(self, sample, encoding):
        rel_data = {
            'entities': [],
            'pairs': [],
            'labels': []
        }
        
        token_entity_id_to_idx = {}
        
        for entity in sample['entities']:
            start_token, end_token = self._find_entity_token_span(encoding, entity)
            if start_token is not None and end_token is not None:
                token_entity_id_to_idx[entity['id']] = len(rel_data['entities'])
                rel_data['entities'].append({
                    'start': start_token,
                    'end': end_token,
                    'type': entity['type'],
                    'id': entity['id']
                })
        
        for rel_type, rel_list in sample['relations'].items():  # Идём по типам отношений
            if rel_type not in ModelConfig.RELATION_TYPES:  # Пропускаем неизвестные типы
                continue
            for relation in rel_list:  # Идём по всем отношениям этого типа
                arg1_idx = token_entity_id_to_idx.get(relation['arg1'], -1)
                arg2_idx = token_entity_id_to_idx.get(relation['arg2'], -1)
                
                if arg1_idx != -1 and arg2_idx != -1:
                    rel_data['pairs'].append((arg1_idx, arg2_idx))
                    rel_data['labels'].append(1)
            
        return rel_data

def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    ner_labels = torch.stack([item['ner_labels'] for item in batch])
    offset_mapping = torch.stack([item['offset_mapping'] for item in batch])

    rel_data = []
    for item in batch:
        rel_entry = {
            'entities': item['rel_data']['entities'],
            'pairs': item['rel_data']['pairs'],
            'labels': torch.tensor(item['rel_data']['labels'], dtype=torch.long) 
                      if item['rel_data']['labels'] else torch.tensor([], dtype=torch.long)
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
    
    tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
    model = NERRelationModel().to(device)

    train_dataset = NERELDataset("NEREL/NEREL-v1.1/train", tokenizer)
    
    # Create weighted sampler
    sample_weights = [1.0 if len(sample['rel_data']['labels']) > 0 else 0.3 
                      for sample in train_dataset]
    sampler = WeightedRandomSampler(sample_weights, len(train_dataset), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=8, collate_fn=collate_fn, sampler=sampler)

    optimizer = AdamW([
        {'params': model.bert.parameters(), 'lr': 2e-5},
        {'params': model.ner_classifier.parameters(), 'lr': 1e-4},
        {'params': model.crf.parameters(), 'lr': 1e-4},
        {'params': model.gat1.parameters(), 'lr': 1e-3},
        {'params': model.gat2.parameters(), 'lr': 1e-3},
        {'params': model.rel_classifiers.parameters(), 'lr': 1e-3}
    ])
    
    for epoch in range(1):
        model.train()
        epoch_loss = 0
        ner_correct = ner_total = 0
        rel_correct = rel_total = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                ner_labels=batch['ner_labels'].to(device),
                rel_data=batch['rel_data']
            )
            
            if outputs['loss'] is not None:
                outputs['loss'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += outputs['loss'].item()
            
            # Calculate metrics
            metrics = calculate_metrics(outputs, batch, model, device)
            ner_correct += metrics['ner_correct']
            ner_total += metrics['ner_total']
            rel_correct += metrics['rel_correct']
            rel_total += metrics['rel_total']

        # Print epoch results
        print_epoch_results(epoch, epoch_loss, len(train_loader), 
                           ner_correct, ner_total, rel_correct, rel_total)

    save_dir = "saved_model"
    model.save_pretrained(save_dir, tokenizer=tokenizer)
    print(f"Model saved to {save_dir}")
    
    return model, tokenizer

def calculate_metrics(outputs, batch, model, device):
    metrics = {
        'ner_correct': 0,
        'ner_total': 0,
        'rel_correct': 0,
        'rel_total': 0
    }
    
    # NER metrics
    with torch.no_grad():
        mask = batch['attention_mask'].to(device).bool()
        ner_preds = model.crf.decode(outputs['ner_logits'], mask=mask)
        
        for i in range(len(ner_preds)):
            seq_len = mask[i].sum().item()
            pred = torch.tensor(ner_preds[i][:seq_len], device=device)
            true = batch['ner_labels'][i][:seq_len].to(device)
            
            metrics['ner_correct'] += (pred == true).sum().item()
            metrics['ner_total'] += seq_len
    
    # Relation metrics
    if 'rel_probs' in outputs and outputs['rel_probs']:
        for rel_type, probs in outputs['rel_probs'].items():
            preds = (torch.sigmoid(probs) > 0.5).long()
            
            # Собираем все метки для этого типа отношения
            rel_labels = []
            for item in batch['rel_data']:
                if 'pairs' in item and item['pairs']:
                    # Предполагаем, что все пары в batch для этого rel_type
                    rel_labels.extend(item['labels'])
            
            if len(rel_labels) > 0:
                true_labels = torch.tensor(rel_labels[:len(preds)], device=device)
                metrics['rel_correct'] += (preds == true_labels).sum().item()
                metrics['rel_total'] += len(true_labels)
    
    
    return metrics

def print_epoch_results(epoch, epoch_loss, num_batches, ner_correct, ner_total, rel_correct, rel_total):
    ner_acc = ner_correct / ner_total if ner_total > 0 else 0
    rel_acc = rel_correct / rel_total if rel_total > 0 else 0
    
    print(f"\nEpoch {epoch+1} Results:")
    print(f"Loss: {epoch_loss/num_batches:.4f}")
    print(f"NER Accuracy: {ner_acc:.2%} ({ner_correct}/{ner_total})")
    print(f"Relation Accuracy: {rel_acc:.2%} ({rel_correct}/{rel_total})")

def predict(text, model, tokenizer, device="cuda", relation_threshold=0.5):
    encoding = tokenizer(text, return_tensors="pt", return_offsets_mapping=True, 
                        max_length=512, truncation=True)
    encoding = {k: v.to(device) for k, v in encoding.items()}  # Переносим все на устройство
    with torch.no_grad():
        outputs = model(
            encoding['input_ids'].to(device),
            encoding['attention_mask'].to(device)
        )
    
    entities = extract_entities(outputs, encoding, tokenizer, text, model)
    relations = extract_relations(outputs, model, encoding, entities, text, device, relation_threshold)
    
    return {
        'text': text,
        'entities': entities,
        'relations': relations
    }

def extract_entities(outputs, encoding, tokenizer, text, model):
    mask = encoding['attention_mask'].bool().to(outputs['ner_logits'].device) 
    ner_preds = model.crf.decode(outputs['ner_logits'], mask=mask)[0]
    tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
    offset_mapping = encoding['offset_mapping'][0].cpu().numpy()
    
    entities = []
    current_entity = None
    
    for i, (token, pred) in enumerate(zip(tokens, ner_preds)):
        if token in ['[CLS]', '[SEP]', '[PAD]']:
            if current_entity:
                entities.append(current_entity)
                current_entity = None
            continue
        
        entity_type = None
        if pred == 1: entity_type = "PERSON"  # B-PER
        elif pred == 3: entity_type = "PROFESSION"  # B-PROF
        elif pred == 5: entity_type = "ORGANIZATION"  # B-ORG
        elif pred == 7: entity_type = "FAMILY"  # B-FAM
        
        if entity_type:
            if current_entity:
                entities.append(current_entity)
            current_entity = {
                'type': entity_type,
                'start': i,
                'end': i,
                'token_ids': [i],
                'text': token.replace('##', ''),
                'id': f"T{len(entities)+1}"
            }
        elif pred in [2, 4, 6, 8]:  # I- labels
            if current_entity and (
                (pred == 2 and current_entity['type'] == "PERSON") or
                (pred == 4 and current_entity['type'] == "PROFESSION") or
                (pred == 6 and current_entity['type'] == "ORGANIZATION") or
                (pred == 8 and current_entity['type'] == "FAMILY")
            ):
                current_entity['end'] = i
                current_entity['token_ids'].append(i)
        else:
            if current_entity:
                entities.append(current_entity)
                current_entity = None
    
    if current_entity:
        entities.append(current_entity)
    
    # Convert token positions to character positions
    for entity in entities:
        start_char = offset_mapping[entity['start']][0]
        end_char = offset_mapping[entity['end']][1]
        entity['text'] = text[start_char:end_char]
        entity['start_char'] = start_char
        entity['end_char'] = end_char
    
    return entities

def extract_relations(outputs, model, encoding, entities, text, device, threshold):
    if len(entities) < 2:
        return []
    
    sequence_output = model.bert(
        encoding['input_ids'],
        encoding['attention_mask']
    ).last_hidden_state
    
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
    
    # Predict relations
    relations = []
    for rel_type in ModelConfig.RELATION_TYPES:
        for i, e1 in enumerate(entities):
            for j, e2 in enumerate(entities):
                if i != j and model._is_valid_pair(e1['type'], e2['type'], rel_type):
                    pair_features = torch.cat([x[i], x[j]])
                    logit = model.rel_classifiers[rel_type](pair_features)
                    prob = torch.sigmoid(logit).item()
                    
                    if prob > threshold:
                        # For FOUNDED_BY we reverse the direction
                        if rel_type == 'FOUNDED_BY':
                            i, j = j, i
                        
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
    
    return sorted(unique_relations.values(), key=lambda x: x['confidence'], reverse=True)

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
            print(f"{e['type']}: {e['text']} ({e['start_char']}-{e['end_char']})")
        print("\nRelations:")
        for r in result['relations']:
            print(f"{r['type']}: {r['arg1']['text']} -> {r['arg2']['text']} (conf: {r['confidence']:.2f})")

    # Load saved model
    loaded_model = NERRelationModel.from_pretrained("saved_model")
    loaded_tokenizer = AutoTokenizer.from_pretrained("saved_model")
    
    # Test loaded model
    result = predict(
        "По улице шел красивый человек, его имя было Мефодий. И был он счастлив. "
        "Работал этот чувак в яндексе, разработчиком. Или директором. Он пока не определился!",
        loaded_model, loaded_tokenizer
    )
    print("\nEntities:")
    for e in result['entities']:
        print(f"{e['type']}: {e['text']} (позиция: {e['start_char']}-{e['end_char']})")
    print("\nОтношения:")
    for r in result['relations']:
        print(f"{r['type']}: {r['arg1']['text']} -> {r['arg2']['text']} (confidence: {r['confidence']:.2f})")