import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv  # Changed from GCNConv to GATConv
from transformers import AutoModel, AutoTokenizer
from torchcrf import CRF  # Added CRF layer
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

class NERRelationModel(nn.Module):
    def __init__(self, model_name="DeepPavlov/rubert-base-cased", num_ner_labels=6):
        super().__init__()
        self.num_ner_labels = num_ner_labels
        self.num_rel_labels = 1  # Только WORKS_AS (бинарная классификация)
        
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
        self.gat1 = GATConv(self.bert.config.hidden_size, 128, heads=4, dropout=0.3)
        self.gat2 = GATConv(128*4, 64, heads=1, dropout=0.3)
        # Concatenate heads from first layer
        
        # Relation classifier (только для WORKS_AS)
        self.rel_classifier = nn.Sequential(
            nn.Linear(64 * 2, 256),  # Конкатенированные эмбеддинги
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)       # Бинарный классификатор 
        )
        # Инициализация весов
        self._init_weights()
    
    def _init_weights(self):
        for module in [self.ner_classifier, self.rel_classifier]:
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
        ner_logits  = self.ner_classifier(sequence_output)
        total_loss = 0
        
        # NER loss
        if ner_labels is not None:
             # Mask for CRF (0 for padding, 1 for real tokens)
            mask = attention_mask.bool()
            ner_loss = -self.crf(ner_logits, ner_labels, mask=mask, reduction='mean')
            total_loss += ner_loss

        # Relation extraction
        rel_probs = None
        if rel_data:
            batch_rel_probs = []
            rel_targets = []
            
            for batch_idx, sample in enumerate(rel_data):
                if not sample.get('pairs', []):
                    continue
                    
                # Фильтрация сущностей
                valid_entities = [
                    e for e in sample['entities'] 
                    if e['start'] <= e['end'] and 
                    e['type'] in ['PERSON', 'PROFESSION']
                ]
                
                if len(valid_entities) < 2:
                    continue
                
                # Создаем эмбеддинги сущностей
                entity_embeddings = []
                for e in valid_entities:
                    start = min(e['start'], sequence_output.size(1)-1)
                    end = min(e['end'], sequence_output.size(1)-1)
                    entity_embed = sequence_output[batch_idx, start:end+1].mean(dim=0)
                    entity_embeddings.append(entity_embed)
                
                # Строим граф только для PERSON и PROFESSION
                person_indices = [i for i, e in enumerate(valid_entities) if e['type'] == 'PERSON']
                prof_indices = [i for i, e in enumerate(valid_entities) if e['type'] == 'PROFESSION']
                
                # Создаем ребра только от PERSON к PROFESSION
                edge_index = []
                for p_idx in person_indices:
                    for prof_idx in prof_indices:
                        edge_index.append([p_idx, prof_idx])
                
                if not edge_index:
                    continue
                
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(input_ids.device)
                x = torch.stack(entity_embeddings)
                
                # Применяем GAT
                x = self.gat1(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=0.3, training=self.training)
                x = self.gat2(x, edge_index)
                
                # Классифицируем отношения
                current_probs = []
                current_targets = []
                
                for (e1_idx, e2_idx), label in zip(sample['pairs'], sample['labels']):
                    # Учитываем только PERSON->PROFESSION и только WORKS_AS
                    if (e1_idx < len(valid_entities) and e2_idx < len(valid_entities)):
                        pair_features = torch.cat([x[e1_idx], x[e2_idx]])
                        current_probs.append(self.rel_classifier(pair_features))
                        current_targets.append(label)
                
                if current_probs:
                    batch_rel_probs.append(torch.cat(current_probs).view(-1))  # Ensure flattened
                    rel_targets.extend(current_targets)

            if batch_rel_probs and rel_targets:
                rel_probs = torch.cat(batch_rel_probs)
                rel_targets = torch.tensor(rel_targets, dtype=torch.float, device=device)

                # # Проверка размеров
                # if rel_probs.size(0) != rel_targets.size(0):
                #     min_len = min(rel_probs.size(0), rel_targets.size(0))
                #     rel_probs = rel_probs[:min_len]
                #     rel_targets = rel_targets[:min_len]
                
                # Add dynamic negative sampling
                if torch.all(rel_targets == 1):
                    neg_probs, neg_targets = self._generate_negative_examples(x, person_indices, prof_indices)
                    if len(neg_probs) > 0:
                        rel_probs = torch.cat([rel_probs, neg_probs])
                        rel_targets = torch.cat([rel_targets, neg_targets])
                        
                # Calculate relation loss with class weighting
                pos_weight = torch.tensor([max(1.0, len(rel_targets)/sum(rel_targets))], device=device)
                rel_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                rel_loss = rel_loss_fn(rel_probs, rel_targets)
                total_loss += rel_loss

                # print(f"Relation probs shape: {rel_probs.shape}, Targets shape: {rel_targets.shape}")
                # print(f"Sample predictions: {rel_probs[:5].tolist()}, Sample targets: {rel_targets[:5].tolist()}")
                # print(f"Relation targets: {rel_targets[:10]}")
                # print(f"Relation probs before sigmoid: {torch.logit(rel_probs[:10])}")

        return {
            'ner_logits': ner_logits,
            'rel_probs': rel_probs,
            'loss': total_loss if total_loss != 0 else None
        }

    def _generate_negative_examples(self, x, person_indices, prof_indices, ratio=0.5):
        neg_probs = []
        neg_targets = []
        
        num_neg = int(len(person_indices) * len(prof_indices) * ratio)
        
        for _ in range(num_neg):
            p_idx = random.choice(person_indices)
            prof_idx = random.choice(prof_indices)
            
            pair_features = torch.cat([x[p_idx], x[prof_idx]])
            neg_probs.append(self.rel_classifier(pair_features))
            neg_targets.append(0.0)
        
        if neg_probs:
            return torch.cat(neg_probs).view(-1), torch.tensor(neg_targets, device=x.device)
        return torch.tensor([], device=x.device), torch.tensor([], device=x.device)

    def save_pretrained(self, save_dir, tokenizer=None):
        """Сохраняет модель, конфигурацию и токенизатор в указанную директорию."""
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Сохраняем веса модели
        torch.save(self.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
        
        # 2. Сохраняем конфигурацию модели в формате Hugging Face
        config = {
            "model_type": "bert",  # Указываем тип модели для Hugging Face
            "architectures": ["NERRelationModel"],
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
        with open(os.path.join(model_dir, "config.json"), "r") as f:
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

# def visualize_relations(entities, relations, text):
#     G = nx.DiGraph()
    
#     # Добавляем узлы
#     for i, ent in enumerate(entities):
#         G.add_node(i, 
#                   label=f"{ent['type']}: {ent['text']}",
#                   color='skyblue' if ent['type'] == 'PERSON' else 'lightgreen')
    
#     # Добавляем ребра
#     for rel in relations:
#         if rel['type'] == 'WORKS_AS':
#             G.add_edge(rel['arg1_idx'], rel['arg2_idx'], 
#                       label='WORKS_AS', 
#                       weight=rel['confidence'])
    
#     # Визуализация
#     pos = nx.spring_layout(G)
#     plt.figure(figsize=(12, 8))
    
#     node_colors = [G.nodes[n]['color'] for n in G.nodes()]
#     edge_labels = nx.get_edge_attributes(G, 'label')
    
#     nx.draw(G, pos, 
#             with_labels=True, 
#             labels={n: G.nodes[n]['label'] for n in G.nodes()},
#             node_color=node_colors,
#             node_size=2000,
#             font_size=10,
#             arrows=True)
    
#     nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
#     plt.title(f"Relation Graph for: {text[:50]}...")
#     plt.show()

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
                if line.startswith('T'):
                    parts = line.strip().split('\t')
                    entity_id = parts[0]
                    type_and_span = parts[1].split()
                    entity_type = type_and_span[0]
                    
                    if entity_type in ['PERSON', 'PROFESSION']:
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
                    rel_type = parts[1].split()[0]
                    arg1 = parts[1].split()[1].split(':')[1]
                    arg2 = parts[1].split()[2].split(':')[1]
                    
                    if rel_type == 'WORKS_AS' and arg1 in entity_map and arg2 in entity_map:
                        # Проверяем типы сущностей для отношения WORKS_AS
                        if entity_map[arg1]['type'] == 'PERSON' and entity_map[arg2]['type'] == 'PROFESSION':
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
        
        # Initialize NER labels (0=O, 1=B-PER, 2=I-PER, 3=B-PROF, 4=I-PROF)
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
                else:
                    ner_labels[start_token] = 3  # B-PROF
                    ner_labels[start_token+1:end_token+1] = 4  # I-PROF
                
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
                
                # Убедимся, что PERSON идет первым в паре
                if e1_type == 'PERSON' and e2_type == 'PROFESSION':
                    rel_data['pairs'].append((arg1_token_idx, arg2_token_idx))
                    rel_data['labels'].append(1)
                elif e1_type == 'PROFESSION' and e2_type == 'PERSON':
                    rel_data['pairs'].append((arg2_token_idx, arg1_token_idx))
                    rel_data['labels'].append(1) 
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'ner_labels': ner_labels,
            'rel_data': rel_data,
            'text': text
        }

def collate_fn(batch):
    # Все элементы batch уже имеют одинаковую длину благодаря padding='max_length'
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    ner_labels = torch.stack([item['ner_labels'] for item in batch])
    
    # Собираем rel_data как список словарей
    rel_data = [item['rel_data'] for item in batch]
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'ner_labels': ner_labels,
        'rel_data': rel_data,
        'texts': [item['text'] for item in batch]  # Добавляем исходные тексты для отладки
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
        {'params': model.rel_classifier.parameters(), 'lr': 1e-3}
    ])
    
    # # Анализ данных перед обучением
    # print("\nData analysis:")
    # sample = train_dataset[0]
    # print("Sample keys:", sample.keys())
    # print("NER labels example:", sample['ner_labels'][:10])
    # print("Relation data example:", sample['rel_data'])

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
            if outputs['rel_probs'] is not None:
                rel_preds = (torch.sigmoid(outputs['rel_probs']) > 0.5).long()
                # Собираем все предсказания и метки для батча
                batch_rel_labels = []
                
                for item in batch['rel_data']:
                    if 'labels' in item and len(item['labels']) > 0:
                        batch_rel_labels.extend(item['labels'])
                
                if batch_rel_labels:
                    # Преобразуем в тензоры
                    rel_labels = torch.tensor(batch_rel_labels, device=device)
                    
                    # Обрезаем или дополняем предсказания до размера меток
                    min_len = min(len(rel_preds), len(rel_labels))
                    rel_correct += (rel_preds[:min_len] == rel_labels[:min_len]).sum().item()
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

def predict(text, model, tokenizer, device="cuda"):
    encoding = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
    tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
    encoding = {k: v.to(device) for k, v in encoding.items()}
    
    with torch.no_grad():
        outputs = model(encoding['input_ids'], encoding['attention_mask'])

    # Decode NER with CRF
    mask = encoding['attention_mask'].bool()
    ner_preds = model.crf.decode(outputs['ner_logits'], mask=mask)[0]

    # Extract entities
    entities = []
    current_entity = None
    
    for i, (token, pred) in enumerate(zip(tokens, ner_preds)):
        if token in ['[CLS]', '[SEP]', '[PAD]']:
            if current_entity:
                entities.append(current_entity)
                current_entity = None
            continue
            
        if pred == 1:  # B-PER
            if current_entity:
                entities.append(current_entity)
            current_entity = {
                'type': "PERSON",
                'start': i,
                'end': i,
                'token_ids': [i],
                'text': token.replace('##', '')
            }
        elif pred == 2:  # I-PER
            if current_entity and current_entity['type'] == "PERSON":
                current_entity['end'] = i
                current_entity['token_ids'].append(i)
        elif pred == 3:  # B-PROF
            if current_entity:
                entities.append(current_entity)
            current_entity = {
                'type': "PROFESSION",
                'start': i,
                'end': i,
                'token_ids': [i],
                'text': token.replace('##', '')
            }
        elif pred == 4:  # I-PROF
            if current_entity and current_entity['type'] == "PROFESSION":
                current_entity['end'] = i
                current_entity['token_ids'].append(i)
        else:  # O
            if current_entity:
                entities.append(current_entity)
                current_entity = None
    
    if current_entity:
        entities.append(current_entity)

    # Filter valid entities
    entities = [e for e in entities if e['type'] in ['PERSON', 'PROFESSION']]

    # Convert token positions to character positions
    offset_mapping = encoding['offset_mapping'][0].cpu().numpy()
    for entity in entities:
        start_char = offset_mapping[entity['start']][0]
        end_char = offset_mapping[entity['end']][1]
        entity['text'] = text[start_char:end_char]

    # Extract relations
    relations = []
    if len(entities) >= 2:
        sequence_output = model.bert(
            encoding['input_ids'], 
            encoding['attention_mask']
        ).last_hidden_state

        # Получаем индексы PERSON и PROFESSION
        person_ents = [i for i, e in enumerate(entities) if e['type'] == 'PERSON']
        prof_ents = [i for i, e in enumerate(entities) if e['type'] == 'PROFESSION']

        if person_ents and prof_ents:
            # Build graph
            edge_index = torch.tensor([
                [p_idx, prof_idx] 
                for p_idx in person_ents 
                for prof_idx in prof_ents
            ], dtype=torch.long).t().contiguous().to(device)
            
            # Create entity embeddings
            entity_embeddings = torch.stack([
                sequence_output[0, e['start']:e['end']+1].mean(dim=0) 
                for e in entities
            ])
            
            # Apply GAT
            x = model.gat1(entity_embeddings, edge_index)
            x = F.relu(x)
            x = model.gat2(x, edge_index)
            
            # Предсказываем отношения
            for p_idx in person_ents:
                for prof_idx in prof_ents:
                    pair_features = torch.cat([x[p_idx], x[prof_idx]])
                    prob = torch.sigmoid(model.rel_classifier(pair_features)).item()
                    
                    if prob > 0.5:
                        relations.append({
                            'type': "WORKS_AS",
                            'arg1_idx': p_idx,
                            'arg2_idx': prof_idx,
                            'arg1': entities[p_idx],
                            'arg2': entities[prof_idx],
                            'confidence': prob
                        })
    
    # Визуализация
    # visualize_relations(entities, relations, text)
    
    return {
        'text': text,
        'entities': entities,
        'relations': relations
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
    tokenizer = AutoTokenizer.from_pretrained("saved_model")
    
    # Использование модели
    result = predict("По улице шел красивый человек, его имя было Мефодий. И был он счастлив. Работал этот чувак в яндексе, разработчиком. Или директором. Он пока не определился!", loaded_model, tokenizer)


