import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from transformers import AutoModel, AutoConfig
from torchcrf import CRF
from typing import Dict, Optional, Tuple
from collections import defaultdict

from constants import (
    ENTITY_TYPES,
    RELATION_TYPES,
    DEFAULT_MODEL_NAME,
    GAT_CONFIG,
    TRAINING_CONFIG
)

# Комбинированная модель для совместного распознавания именованных сущностей (NER) и извлечения отношений между ними
class NERRelationModel(nn.Module):
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        num_ner_labels: int = len(ENTITY_TYPES) * 2 + 1,  # BIO-схема + O
        num_rel_labels: int = len(RELATION_TYPES)
    ):
        super().__init__()
        self.model_name = model_name
        self.num_ner_labels = num_ner_labels
        self.num_rel_labels = num_rel_labels
        
        # Инициализация компонентов
        self._init_bert()
        self._init_ner_head()
        self._init_gat()
        self._init_relation_classifiers()
        
        # Инициализация весов
        self._init_weights()

    # Инициализация BERT-энкодера
    def _init_bert(self) -> None:
        self.bert = AutoModel.from_pretrained(self.model_name)
        self.bert_hidden_size = self.bert.config.hidden_size

    # Инициализация NER-классификатора и CRF
    def _init_ner_head(self) -> None:
        self.ner_classifier = nn.Sequential(
            nn.Linear(self.bert_hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(TRAINING_CONFIG['dropout_rate']),
            nn.Linear(256, self.num_ner_labels)
        )
        self.crf = CRF(self.num_ner_labels, batch_first=True)

    # Инициализация Graph Attention Network
    def _init_gat(self) -> None:
        self.gat1 = GATConv(
            self.bert_hidden_size,
            GAT_CONFIG['hidden_size'],
            heads=GAT_CONFIG['heads'],
            dropout=TRAINING_CONFIG['dropout_rate'],
            concat=True
        )
        self.norm1 = nn.LayerNorm(GAT_CONFIG['hidden_size'] * GAT_CONFIG['heads'])
        self.gat2 = GATConv(
            GAT_CONFIG['hidden_size'] * GAT_CONFIG['heads'],
            GAT_CONFIG['output_size'],
            heads=1,
            dropout=TRAINING_CONFIG['dropout_rate'],
            concat=False
        )
        self.norm2 = nn.LayerNorm(GAT_CONFIG['output_size'])

    # Инициализация классификаторов отношений
    def _init_relation_classifiers(self) -> None:
        input_dim = GAT_CONFIG['output_size'] * 2
        hidden_dim = 256
        
        self.rel_classifiers = nn.ModuleDict({
            rel_type: self._build_classifier(rel_type, input_dim, hidden_dim)
            for rel_type in RELATION_TYPES
        })

    # Создает классификатор для конкретного типа отношения
    def _build_classifier(self, rel_type: str, input_dim: int, hidden_dim: int) -> nn.Module:
        if rel_type in {'SPOUSE', 'SIBLING', 'RELATIVE'}:
            return self._build_symmetric_classifier(input_dim, hidden_dim)
        return self._build_standard_classifier(input_dim, hidden_dim)

    # Стандартный классификатор отношений
    @staticmethod
    def _build_standard_classifier(input_dim: int, hidden_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(TRAINING_CONFIG['dropout_rate']),
            nn.Linear(hidden_dim, 1)
        )

    # Классификатор для симметричных отношений
    @staticmethod
    def _build_symmetric_classifier(input_dim: int, hidden_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(TRAINING_CONFIG['dropout_rate']),
            nn.Linear(hidden_dim, 1)
        )

    # Инициализация весов
    def _init_weights(self) -> None:
        for module in [self.ner_classifier, *self.rel_classifiers.values()]:
            if isinstance(module, nn.Sequential):
                for submodule in module:
                    if isinstance(submodule, nn.Linear):
                        nn.init.xavier_uniform_(submodule.weight)
                        nn.init.constant_(submodule.bias, 0)

    # Прямой проход модели    
    def forward(self, input_ids, attention_mask, ner_labels=None, rel_data=None) -> Dict:
        device = input_ids.device
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        rel_probs = defaultdict(list)
        total_loss = 0

        # NER предсказания с CRF
        ner_logits = self.ner_classifier(sequence_output)

        # NER потери
        if ner_labels is not None:
            mask = attention_mask.bool()
            ner_loss = -self.crf(ner_logits, ner_labels, mask=mask, reduction='mean')
            total_loss += ner_loss

        if rel_data and self.training:
            total_rel_loss = 0
            rel_correct = 0
            rel_total = 0

            # Обработка каждого образца
            for batch_idx, sample in enumerate(rel_data):
                if 'pairs' not in sample or len(sample['pairs']) == 0:
                    print(f"Пропуск примера {batch_idx}: нет пар отношений")
                    continue
                    
                # Корректные сущности с типом
                valid_entities = [e for e in sample['entities'] 
                    if isinstance(e, dict) and 'start' in e and 'end' in e and 'type' in e
                ]
                
                if len(valid_entities) < 2:
                    continue

                # print(f"\nОбработка примера {batch_idx}:")
                # print(f"Сущности: {[(e['type'], e['id']) for e in valid_entities]}")
                # print(f"Пары отношений: {sample['pairs']}")
                # print(f"Метки отношений: {sample['labels']}") 
                
                # Создание эмбеддингов для сущностей
                entity_embeddings = []
                entity_types = []
                for e in valid_entities:
                    start = min(e['start'], sequence_output.size(1)-1)
                    end = min(e['end'], sequence_output.size(1)-1)
                    entity_embed = sequence_output[batch_idx, start:end+1].mean(dim=0)
                    entity_embeddings.append(entity_embed)
                    entity_types.append(e['type'])
                
                # Построение полного графа
                edge_index = torch.tensor([
                    [i, j] for i in range(len(valid_entities)) 
                    for j in range(len(valid_entities)) if i != j
                ], dtype=torch.long).t().contiguous().to(device)
                
                x = torch.stack(entity_embeddings).to(device)
                
                # GAT с нормализацией
                x = self.gat1(x, edge_index)
                x = self.norm1(x)
                x = F.elu(x)
                x = F.dropout(x, p=0.3, training=self.training)
                x = self.gat2(x, edge_index)
                x = self.norm2(x)
                x = F.elu(x)

                entity_indices = {e['id']: i for i, e in enumerate(valid_entities)}
                
                # Обработка каждого типа отношения
                rel_probs = defaultdict(list)
                rel_targets = defaultdict(list)

                for rel_type in RELATION_TYPES:
                    pos_count = 0
                    
                    # Положительные примеры
                    for (e1_idx, e2_idx), label in zip(sample['pairs'], sample['labels']):
                        if label == RELATION_TYPES[rel_type]:
                            if e1_idx < len(valid_entities) and e2_idx < len(valid_entities):
                                i = e1_idx
                                j = e2_idx
                            
                                if rel_type == 'FOUNDED_BY':
                                    i, j = j, i
                                
                                pair_features = torch.cat([x[i], x[j]])
                                rel_probs[rel_type].append(self.rel_classifiers[rel_type](pair_features))
                                rel_targets[rel_type].append(1.0)
                                pos_count += 1
                    
                    print(f"Тип отношения {rel_type}: найдено {pos_count} положительных примеров")
                    
                    # Генерация отрицательных примеров для каждого типа отношений
                    neg_pairs = self._generate_negative_examples(
                        entity_embeddings=x, 
                        entity_types=entity_types, 
                        rel_type=rel_type,
                        pos_indices={(i,j) for (i,j), label in zip(sample['pairs'], sample['labels']) 
                                    if label == RELATION_TYPES[rel_type]},
                        ratio=0.5
                    )
                    
                    if neg_pairs:
                        neg_features, neg_targets = neg_pairs
                        rel_probs[rel_type].extend(neg_features)
                        rel_targets[rel_type].extend(neg_targets)
                        print(f"Добавлено {len(neg_targets)} отрицательных примеров для {rel_type}")

                    else:
                        print(f"Не удалось сгенерировать отрицательные примеры для {rel_type}")

                
                    # Потери для типа отношения
                    if rel_probs[rel_type]:
                        min_len = min(len(rel_probs[rel_type]), len(rel_targets[rel_type]))
                        probs_tensor = torch.cat(rel_probs[rel_type][:min_len]).view(-1)
                        targets_tensor = torch.tensor(rel_targets[rel_type][:min_len], dtype=torch.float, device=device)

                        # Установка pos_weight на основе дисбаланса классов
                        pos_weight = torch.tensor([
                            max(1.0, len(targets_tensor) / (sum(targets_tensor) + 1e-6)) * 5.0  # Увеличенный коэффициент
                        ], device=device)
                        
                        rel_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(
                            probs_tensor, targets_tensor)
                        total_loss += rel_loss * 1.0

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

    # Генерация отрицательных примеров для определенного типа отношений    
    def _generate_negative_examples(self, entity_embeddings, entity_types, rel_type,  pos_indices=None, ratio=0.5):
        device = entity_embeddings.device
        neg_probs = []
        neg_targets = []
        
        if pos_indices is None:
            pos_indices = set()
        
        if rel_type == 'WORKS_AS':
            valid_pairs = [(i,j) for i, e1 in enumerate(entity_types) 
                        for j, e2 in enumerate(entity_types)
                        if i != j and e1 == 'PERSON' and e2 == 'PROFESSION']
        elif rel_type == 'MEMBER_OF':
            valid_pairs = [(i,j) for i, e1 in enumerate(entity_types)
                        for j, e2 in enumerate(entity_types)
                        if i != j and e1 == 'PERSON' and e2 == 'ORGANIZATION']
        elif rel_type == 'FOUNDED_BY':
            valid_pairs = [(i,j) for i, e1 in enumerate(entity_types)
                        for j, e2 in enumerate(entity_types)
                        if i != j and e1 == 'ORGANIZATION' and e2 == 'PERSON']
        elif rel_type in ['SPOUSE', 'SIBLING']:
            valid_pairs = [(i,j) for i, e1 in enumerate(entity_types)
                        for j, e2 in enumerate(entity_types)
                        if i != j and e1 == 'PERSON' and e2 == 'PERSON']
        else:
            valid_pairs = [(i,j) for i in range(len(entity_types))
                        for j in range(len(entity_types)) if i != j]
        
        # Исключение положительных примеров
        valid_pairs = [p for p in valid_pairs if p not in pos_indices]
        
        # Для симметричных отношений только уникальные пары
        if rel_type in ['SPOUSE', 'SIBLING']:
            valid_pairs = list({(min(i,j), max(i,j)) for i,j in valid_pairs})
        
        # Выбор случайного подмножества (не более 5 отрицательных на 1 положительный)
        num_samples = min(len(valid_pairs), max(5 * len(pos_indices), 10))
        sampled_pairs = random.sample(valid_pairs, num_samples) if valid_pairs else []
        
        for i, j in sampled_pairs:
            if rel_type == 'FOUNDED_BY':
                i, j = j, i
            
            pair_features = torch.cat([entity_embeddings[i], entity_embeddings[j]])
            neg_probs.append(self.rel_classifiers[rel_type](pair_features))
            neg_targets.append(0.0)

        if neg_probs:
            return torch.stack(neg_probs).view(-1, 1), torch.tensor(neg_targets, device=device)
        return None