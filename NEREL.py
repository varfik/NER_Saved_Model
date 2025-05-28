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

import logging
import unicodedata
from sklearn.metrics.pairwise import cosine_similarity
from termcolor import colored

# Настройка логгера
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
#logger.setLevel(logging.DEBUG)

SYMMETRIC_RELATIONS = {'SIBLING', 'SPOUSE', 'RELATIVE'}

# Цвета для визуализации разных типов сущностей
ENTITY_COLORS = {
    'PERSON': 'cyan',
    'PROFESSION': 'green',
    'ORGANIZATION': 'yellow',
    'FAMILY': 'magenta',
    'LOCATION': 'blue',
}

# Маппинг типов сущностей в числовые индексы
ENTITY_TYPES = {
    'PERSON': 0,
    'PROFESSION': 1,
    'ORGANIZATION': 2,
    'FAMILY': 3,
    'LOCATION': 4
}

# Маппинг типов отношений в числовые индексы
RELATION_TYPES = {
    'WORKS_AS': 0,
    'MEMBER_OF': 1,
    'FOUNDED_BY': 2,
    'SPOUSE': 3,
    'PARENT_OF': 4,
    'SIBLING': 5,
    'PART_OF': 6,
    'WORKPLACE': 7,
    'RELATIVE': 8
}

RELATION_THRESHOLDS = {k: 0.5 for k in RELATION_TYPES}

VALID_COMB = {
    'WORKS_AS': [('PERSON', 'PROFESSION')],
    'MEMBER_OF': [('PERSON', 'ORGANIZATION'), ('PERSON', 'FAMILY'), ('PROFESSION', 'FAMILY')],
    'FOUNDED_BY': [('ORGANIZATION', 'PERSON'), ('LOCATION', 'PERSON'), ('ORGANIZATION', 'ORGANIZATION'), ('PROFESSION', 'ORGANIZATION')],
    'SPOUSE': [('PERSON', 'PERSON'), ('PROFESSION', 'PROFESSION'), ('PROFESSION', 'PERSON'), ('PERSON', 'PROFESSION')],
    'PARENT_OF': [('PERSON', 'PERSON'), ('PROFESSION', 'PERSON'), ('PERSON', 'PROFESSION')],
    'SIBLING': [('PERSON', 'PERSON'), ('PROFESSION', 'PERSON'), ('PERSON', 'PROFESSION')],
    'PART_OF': [('ORGANIZATION', 'ORGANIZATION'), ('LOCATION', 'LOCATION')],
    'WORKPLACE': [('PERSON', 'ORGANIZATION'), ('PERSON', 'LOCATION'),  ('PROFESSION', 'ORGANIZATION')],
    'RELATIVE': [('PERSON', 'PERSON'), ('PROFESSION', 'PERSON'), ('PERSON', 'PROFESSION')]
}


RELATION_TYPES_INV = {v: k for k, v in RELATION_TYPES.items()}

class NERRelationModel(nn.Module):
    def __init__(self, model_name="DeepPavlov/rubert-base-cased", num_ner_labels=len(ENTITY_TYPES)*2+1, num_rel_labels=len(RELATION_TYPES), relation_types=None, entity_types=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Initialize NER labels (0=O, 1=B-PER, 2=I-PER, ..., 9=B-LOC, 10=I-LOC)
        self.entity_type_to_idx = {etype: idx for idx, etype in enumerate(entity_types)}
        self.relation_types = relation_types
        self.entity_types = entity_types
        self.num_ner_labels = num_ner_labels
        self.num_rel_labels = num_rel_labels

        # BERT encoder
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size

        # NER Head with CRF
        self.ner_classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_ner_labels)
        )
        self.crf = CRF(num_ner_labels, batch_first=True)

        # Graph attention network components (GAT)
        # Improved GAT architecture
        self.gat1 = GATConv(
            hidden_size,
            128,
            heads=4,
            dropout=0.2,
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
        self.to(self.device)

    def _init_weights(self):
        for m in self.rel_classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _encode_entities(self, sequence_output, batch_idx, sample, device):
        """Encode entities from sample into embeddings.

            Args:
                sequence_output: Tensor of shape [batch_size, seq_len, hidden_size]
                batch_idx: Index of current sample in batch
                sample: Dictionary containing 'entities' list
                device: Target device for tensors

            Returns:
                tuple: (entities, id_map, embeddings) or (None, None, None) if invalid
            """
        if not isinstance(sample, dict) or 'entities' not in sample:
            logger.warning(f"Invalid sample format in batch {batch_idx}")
            return None, None, None

        if sequence_output is None or batch_idx >= sequence_output.shape[0]:
            logger.warning(f"Invalid batch index or sequence output in batch {batch_idx}")
            return None, None, None

        valid_entities = []
        for e in sample.get('entities', []):
            if not all(k in e for k in ['start', 'end', 'type', 'id']):
                logger.debug(f"Skipping invalid entity in batch {batch_idx}: missing required fields")
                continue

            if e['start'] >= e['end']:
                logger.debug(f"Skipping invalid entity span in batch {batch_idx}: start >= end")
                continue

            valid_entities.append(e)

        if len(valid_entities) < 2:
            if self.training:
                logger.debug(f"Skipping batch {batch_idx}: not enough entities ({len(valid_entities)})")
            return None, None, None

        entity_embeds, id_map = [], {}
        unknown_types = set()

        for i, e in enumerate(valid_entities):
            start = max(0, min(e['start'], sequence_output.shape[1] - 1))
            end = max(start, min(e['end'], sequence_output.shape[1] - 1))

            token_embeds = sequence_output[batch_idx, start:end + 1]
            if token_embeds.shape[0] == 0:
                logger.debug(f"Empty token embeddings for entity {e['id']} in batch {batch_idx}")
                continue

            # Взвешенное pooling
            attention_scores = token_embeds @ token_embeds.mean(dim=0)
            attention_weights = torch.softmax(attention_scores, dim=0)
            pooled = torch.sum(token_embeds * attention_weights.unsqueeze(-1), dim=0)

            # Обработка типа сущности
            type_idx = self.entity_type_to_idx.get(e['type'], -1)
            if type_idx == -1:
                if e['type'] not in unknown_types:
                    logger.warning(f"Unknown entity type '{e['type']}' in batch {batch_idx}")
                    unknown_types.add(e['type'])
                continue

            with torch.no_grad():
                type_emb = self.entity_type_emb(torch.tensor(type_idx, device=device))
            entity_embeds.append(pooled + type_emb)
            id_map[e['id']] = len(entity_embeds) - 1

        if len(entity_embeds) < 2:
            logger.debug(f"Not enough valid entity embeddings in batch {batch_idx}")
            return None, None, None

        try:
            x = torch.stack(entity_embeds)
        except RuntimeError as e:
            logger.error(f"Failed to stack entity embeddings in batch {batch_idx}: {str(e)}")
            return None, None, None

        return valid_entities, id_map, x

    def _compute_gat(self, x, device=None):
        """Apply Graph Attention Network layers.

            Args:
                x: Tensor of shape [num_nodes, hidden_size] with node features
                device: Target device (inferred from x if None)

            Returns:
                Updated node representations
            """
        device = device or x.device
        edge_index = self._build_knn_graph(x, k=5)

        if edge_index.size(1) == 0:
            logger.debug("[GAT] Empty graph - using fallback (no message passing)")
            return x  # fallback — без графа

        if not self.training:  # В режиме инференса сохраняем только уникальные ребра
            edge_index = torch.unique(edge_index, dim=1)
        else:
            # В тренировочном режиме добавляем обратные связи
            reversed_edges = edge_index[[1, 0], :]
            edge_index = torch.cat([edge_index, reversed_edges], dim=1)
            edge_index = torch.unique(edge_index, dim=1)  # Удаляем дубликаты

            # Добавляем self-loops если их еще нет
        num_nodes = x.size(0)
        self_loops = torch.arange(num_nodes, device=device).repeat(2, 1)
        edge_index = torch.cat([edge_index, self_loops], dim=1)
        edge_index = torch.unique(edge_index, dim=1)  # Удаляем дубликаты

        try:
            x = self.gat1(x, edge_index)
            x = self.norm1(x)
            x = F.elu(x)

            x = self.gat2(x, edge_index)
            x = self.norm2(x)
            x = F.elu(x)

            logger.debug(f"[GAT] Applied GAT with {edge_index.size(1)} edges "
                         f"for {num_nodes} nodes")
        except RuntimeError as e:
            logger.error(f"[GAT] Error in GAT layers: {str(e)}")
            return x  # Fallback в случае ошибки

        return x

    def _build_knn_graph(self, x, k=5):
        """Build KNN graph based on cosine similarity.

           Args:
               x: Tensor of shape [num_nodes, hidden_size] with node embeddings
               k: Number of nearest neighbors to connect

           Returns:
               edge_index: Tensor of shape [2, num_edges] with graph edges
           """
        if x.size(0) < 2:  # Нельзя построить граф для <2 узлов
            logger.debug("[KNN] Not enough nodes to build graph")
            return torch.empty((2, 0), dtype=torch.long, device=x.device)

        with torch.no_grad():
            # Используем torch.cosine_similarity вместо sklearn для работы на GPU
            x_norm = F.normalize(x, p=2, dim=1)
            sim_matrix = torch.mm(x_norm, x_norm.t())  # [num_nodes, num_nodes]

            # Получаем топ-k соседей для каждого узла (исключая себя)
            topk_values, topk_indices = torch.topk(sim_matrix, k=k + 1, dim=1)  # +1 чтобы учесть self-loop
            # Создаем ребра
            edge_list = []
            for i in range(x.size(0)):
                for j in topk_indices[i]:
                    if i != j and topk_values[i, j] > 0.1:  # Порог сходства
                        edge_list.append((i, j))

            if not edge_list:
                logger.debug("[KNN] No edges created - all similarities below threshold")
                return torch.empty((2, 0), dtype=torch.long, device=x.device)

            edge_index = torch.tensor(edge_list, dtype=torch.long, device=x.device).t()

            logger.debug(f"[KNN] Built graph with {edge_index.size(1)} edges "
                         f"for {x.size(0)} nodes (k={k})")
            return edge_index

    def forward(self, input_ids, attention_mask, ner_labels=None, rel_data=None):
        device = input_ids.device
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # (batch_size, seq_len, hidden)
        cls_token = sequence_output[:, 0, :]  # [batch, hidden]

        loss = 0
        ner_logits = self.ner_classifier(sequence_output)

        if ner_labels is not None:
            loss += self._compute_ner_loss(ner_logits, ner_labels, attention_mask)
            logger.debug(f"[Forward] NER loss: {loss.item():.4f}")

        rel_logits = defaultdict(list)
        rel_labels = defaultdict(list)
        rel_probs = defaultdict(list)

        if rel_data:
            logger.debug(f"[Forward] Обработка {len(rel_data)} примеров с отношениями")
            for batch_idx, sample in enumerate(rel_data):
                rel_info = self._process_relation_sample(batch_idx, sample, sequence_output, cls_token, device)
                if rel_info is None:
                    continue
                for rel_type_idx, scores_labels in rel_info.items():
                    scores, labels = zip(*scores_labels)
                    rel_logits[rel_type_idx].extend(scores)
                    rel_labels[rel_type_idx].extend(labels)

                    # Преобразуем логиты в вероятности с помощью сигмоиды
                    probs = [torch.sigmoid(score) for score in scores]
                    rel_probs[rel_type_idx].extend(zip(probs, labels))  # Сохраняем (вероятность, метку)

            if self.training:
                loss += self._compute_relation_loss(rel_logits, rel_labels, device)

        return {
            'ner_logits': ner_logits,
            'rel_logits': rel_logits,
            'rel_labels': rel_labels,
            'rel_probs': rel_probs,
            'loss': loss if loss != 0 else None
        }

    # Вычисляет лосс для задачи NER с помощью CRF слоя
    def _compute_ner_loss(self, ner_logits, ner_labels, attention_mask):
        mask = attention_mask.bool()
        return -self.crf(ner_logits, ner_labels, mask=mask, reduction='mean')

    # Обрабатывает одну запись в rel_data: извлекает представления сущностей,
    # формирует пары и вычисляет logits и метки
    def _process_relation_sample(self, batch_idx, sample, sequence_output, cls_token, device):
        if not sample or 'pairs' not in sample or 'labels' not in sample:
            logger.debug(f"Invalid sample format in batch {batch_idx}")
            return None
        entities, id_map, x = self._encode_entities(sequence_output, batch_idx, sample, device)
        if x is None or len(id_map) < 2:
            return None
        logger.debug(f"Sample entities: {entities}")
        logger.debug(f"ID map: {id_map}")
        logger.debug(f"Original pairs: {sample['pairs']}")
        logger.debug(f"Original labels: {sample['labels']}")
        try:
            x = self._compute_gat(x, device)
        except RuntimeError as e:
            logger.error(f"GAT failed in batch {batch_idx}: {str(e)}")
            return None

        cls_vec = cls_token[batch_idx]
        rel_info = defaultdict(list)

        pos_pairs = self._process_positive_relations(
            sample, id_map, x, cls_vec, device, batch_idx
        )

        if self.training:
            neg_pairs = self._generate_negative_relations(
                x, entities, pos_pairs, device, batch_idx
            )
            rel_info.update(neg_pairs)

        return rel_info

    def _process_positive_relations(self, sample, id_map, x, cls_vec, device, batch_idx):
        """Process positive relations with symmetric handling."""
        pos_pairs = defaultdict(list)
        entity_pairs = set()  # Track all unique entity pairs

        for (i1, i2), label in zip(sample['pairs'], sample['labels']):
            # Convert and validate entity IDs
            ent1_id, ent2_id = f"T{i1}", f"T{i2}"
            if ent1_id not in id_map or ent2_id not in id_map:
                continue

            idx1, idx2 = id_map[ent1_id], id_map[ent2_id]
            label_idx = self._validate_relation_label(label, batch_idx)
            if label_idx is None:
                continue

            # Handle symmetric relations
            if self.relation_types[label_idx] in SYMMETRIC_RELATIONS:
                idx1, idx2 = sorted([idx1, idx2])

            # Skip duplicates
            if (idx1, idx2) in entity_pairs:
                continue
            entity_pairs.add((idx1, idx2))

            # Create relation vector
            rel_vec = self._create_relation_vector(x, idx1, idx2, label_idx, cls_vec, device)
            pos_pairs[label_idx].append((rel_vec, 1.0))

        return pos_pairs

    def _generate_negative_relations(self, x, entities, pos_pairs, device, batch_idx):
        """Generate hard negative samples focusing on semantically similar pairs."""
        neg_pairs = defaultdict(list)
        entity_types = [e['type'] for e in entities]
        num_entities = len(entities)

        # Precompute all valid type combinations
        valid_masks = {}
        for rel_type_idx in range(len(self.relation_types)):
            rel_type = self.relation_types[rel_type_idx]
            mask = torch.zeros((num_entities, num_entities), dtype=torch.bool, device=device)

            for i, j in torch.nditer(torch.arange(num_entities), torch.arange(num_entities)):
                if i != j and (entity_types[i], entity_types[j]) in VALID_COMB.get(rel_type, []):
                    mask[i, j] = True
            valid_masks[rel_type_idx] = mask

        # Generate negatives for each relation type
        for rel_type_idx in range(len(self.relation_types)):
            pos_set = {(i, j) for i, j, _ in pos_pairs.get(rel_type_idx, [])}
            valid_mask = valid_masks[rel_type_idx]

            # Get candidate pairs
            candidates = (valid_mask & ~self._pairs_to_mask(pos_set, num_entities)).nonzero()

            if len(candidates) == 0:
                continue

            # Hard negative mining
            neg_samples = self._sample_hard_negatives(
                x, candidates, pos_pairs.get(rel_type_idx, []),
                n_samples=min(3 * len(pos_set), len(candidates))
            )

            # Create negative vectors
            for i, j in neg_samples:
                rel_vec = self._create_relation_vector(x, i, j, rel_type_idx, cls_vec, device)
                neg_pairs[rel_type_idx].append((rel_vec, 0.0))

        return neg_pairs

    def _pairs_to_mask(self, pairs, num_entities):
        """
        Convert list of entity pairs to boolean adjacency matrix.

        Args:
            pairs: Set/Tensor/List of tuples (i,j) representing relations
            num_entities: Total number of entities in the sample

        Returns:
            mask: Boolean tensor of shape [num_entities, num_entities]
        """
        mask = torch.zeros((num_entities, num_entities), dtype=torch.bool, device=self.device)
        if not pairs:
            return mask

        # Convert different input formats to tensor
        if isinstance(pairs, set):
            pairs = torch.tensor(list(pairs), device=self.device)
        elif isinstance(pairs, list):
            pairs = torch.tensor(pairs, device=self.device)

        # Fill the mask
        mask[pairs[:, 0], pairs[:, 1]] = True
        return mask

    # Создаёт вектор признаков для пары сущностей и пропускает через классификатор отношений
    def _create_relation_vector(self, x, idx1, idx2, rel_type_idx, cls_vec, device):
        """Optimized relation vector creation."""
        device = device or self.device
        with torch.no_grad():
            rel_type_tensor = self.rel_type_emb(
                torch.tensor(rel_type_idx, device=device)
            )
        return torch.cat([
            x[idx1],
            x[idx2],
            rel_type_tensor,
            cls_vec
        ])

    def _sample_hard_negatives(self, x, candidates, pos_pairs, n_samples):
        """Select hard negatives using similarity metrics."""
        if not pos_pairs or n_samples <= 0:
            return candidates[:n_samples]

        # Compute positive mean embedding
        pos_emb = torch.stack([
            torch.cat([x[i], x[j]]) for i, j, _ in pos_pairs
        ]).mean(0)

        # Compute candidate similarities
        cand_emb = torch.cat([x[candidates[:, 0]], x[candidates[:, 1]]], dim=1)
        sim = F.cosine_similarity(
            cand_emb,
            pos_emb.unsqueeze(0),
            dim=1
        )

        # Select most similar negatives
        _, indices = torch.topk(sim, k=min(n_samples, len(sim)))
        return candidates[indices]

    def _validate_relation_label(self, label, batch_idx):
        """Validate and normalize relation labels."""
        try:
            label_idx = int(label)
            if 0 <= label_idx < len(self.relation_types):
                return label_idx
        except (ValueError, TypeError):
            pass
        logger.warning(f"Invalid relation label {label} in batch {batch_idx}")
        return None

    # Вычисляет loss для задачи извлечения отношений.
    def _compute_relation_loss(self, rel_logits, rel_labels, device):
        loss = 0
        for rel_type_idx, scores in rel_logits.items():
            if not scores:
                continue
            logits = torch.stack(scores).view(-1)
            labels = torch.tensor(rel_labels[rel_type_idx], device=device, dtype=torch.float)
            weight = 3.0 if self.relation_types[rel_type_idx] in ['SPOUSE', 'SIBLING'] else 1.0
            pos_weight = torch.tensor([weight], device=device)
            loss += nn.BCEWithLogitsLoss(pos_weight=pos_weight)(logits, labels)
        return loss
    def _get_negatives(self, x, entities, rel_type_idx, rel_type, pos_indices_by_type, device):
        pos_set = set(pos_indices_by_type[rel_type_idx])
        if len(x) < 2:
            return []

        # Составим матрицу совместимых типов
        valid_combinations = VALID_COMB.get(self.relation_types[rel_type_idx], [])
        # valid_combinations = VALID_COMB.get(rel_type, [])
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
            logger.warning(f"[Negatives] Нет допустимых пар для отношения '{rel_type}'")
            return []

        # Построим отрицательные пары
        i_indices = valid_indices[:, 0]
        j_indices = valid_indices[:, 1]
        neg_pairs = torch.stack([i_indices, j_indices], dim=1)

        logger.debug(f"[Negatives] Найдено {len(neg_pairs)} допустимых отрицательных пар для типа '{rel_type}'")

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
        """Сохраняет модель и конфигурацию в указанную директорию"""
        os.makedirs(save_dir, exist_ok=True)

        # Сохраняем модель и конфигурацию
        torch.save({
            'state_dict': self.state_dict(),
            'config': {
                'num_ner_labels': self.num_ner_labels,
                'num_rel_labels': self.num_rel_labels,
                'entity_types': self.entity_types,
                'relation_types': self.relation_types,
                'bert_config': self.bert.config.to_dict()
            }
        }, os.path.join(save_dir, "model.bin"))

        # Сохраняем токенизатор если предоставлен
        if tokenizer:
            tokenizer.save_pretrained(save_dir)

    @classmethod
    def from_pretrained(cls, model_dir, device=None):
        """Загружает модель из указанной директории"""
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Загружаем сохраненные данные
        model_data = torch.load(os.path.join(model_dir, "model.bin"), map_location=device)
        config = model_data['config']

        # Инициализируем модель
        model = cls(
            model_name=model_dir,
            num_ner_labels=config['num_ner_labels'],
            num_rel_labels=config['num_rel_labels'],
            entity_types=config['entity_types'],
            relation_types=config['relation_types']
        )

        # Загружаем веса
        model.load_state_dict(model_data['state_dict'])
        model.to(device).eval()

        return model

class NERELDataset(Dataset):
    def __init__(self, data_dir, tokenizer, max_length=512):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self._load_samples()

        self.misaligned_entities = 0
        self.total_entities = 0
        self.skipped_relations_due_to_alignment = 0

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
                    if entity_type not in ENTITY_TYPES:
                        continue
                    try:
                        start = int(type_and_span[1])
                        end = int(type_and_span[-1])
                    except ValueError:
                        continue

                    entity_text = parts[2]
                    extracted_text = text[start:end]

                    norm_extracted = unicodedata.normalize("NFC", text[start:end].replace('\u00A0', ' '))
                    norm_expected = unicodedata.normalize("NFC", entity_text.replace('\u00A0', ' '))

                    if norm_extracted != norm_expected:
                        recovered = self._find_entity_span(norm_expected, text)
                        if recovered:
                            start, end = recovered
                        else:
                            logger.debug(f"Misalignment detected:\n"
                                         f"  entity_id: {entity_id}\n"
                                         f"  expected: '{entity_text}'\n"
                                         f"  found:    '{extracted_text}'\n"
                                         f"  context:  '{text[start - 20:end + 20].replace(chr(10), '⏎')}'")
                            logger.warning(f"Entity alignment failed: Entity: '{entity_text}' ({entity_type}), "
                                           f"Span: {start}-{end}, Text: '{text[start - 10:end + 10]}'")
                            self.misaligned_entities += 1
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

                    # Проверяем существование сущностей
                    if arg1 and arg2 and arg1 in entity_map and arg2 in entity_map:
                        logger.debug(f"Relation: {rel_type} between {arg1} and {arg2}")
                        relations.append({
                            'type': rel_type,
                            'arg1': arg1,
                            'arg2': arg2
                        })
        return entities, relations

    def __len__(self):
        return len(self.samples)

    def _find_entity_span(self, entity_text, full_text):
        for match in re.finditer(re.escape(entity_text), full_text):
            return match.start(), match.end()
        return None

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

        # Initialize NER labels (0=O, 1=B-PER, 2=I-PER, 3=B-PROF, 4=I-PROF, 5=B-ORGANIZATION, 6=I-ORGANIZATION, 7=B-FAMILY, 8=I-FAMILY)
        ner_labels = torch.zeros(self.max_length, dtype=torch.long)
        offset_mapping = encoding['offset_mapping'][0]
        token_entities = []

        # Align entities with tokenization
        for entity in sample['entities']:
            matched_tokens = []

            # Find token spans for entity
            start_token = end_token = None
            for i, (start, end) in enumerate(offset_mapping):
                if start == end:
                    continue  # спецтокены
                if start >= entity['start'] and end <= entity['end']:
                    matched_tokens.append(i)

            # if not matched_tokens:
            #     recovered = self._find_best_span(entity['text'], text, entity['start'])
            #     if recovered:
            #         entity['start'], entity['end'] = recovered
            #         for i, (start, end) in enumerate(offset_mapping):
            #             if start < entity['end'] and end > entity['start']:
            #                 matched_tokens.append(i)

            if not matched_tokens:
                logger.warning(f"Entity alignment failed: Entity: '{entity['text']}' ({entity['type']}), "
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

        # Prepare relation data
        rel_data = {
            'entities': token_entities,
            'pairs': [],
            'labels': []
        }

        token_entity_id_to_idx = {e['id']: i for i, e in enumerate(token_entities)}
        pos_pairs = set()

        for relation in sample['relations']:
            if relation['type'] not in RELATION_TYPES:
                continue
            idx1 = token_entity_id_to_idx.get(relation['arg1'], -1)
            idx2 = token_entity_id_to_idx.get(relation['arg2'], -1)
            if idx1 == -1 or idx2 == -1:
                self.skipped_relations_due_to_alignment += 1
                logger.warning(
                    f"Relation '{relation['type']}' skipped: unresolved entity id(s): {relation['arg1']}, {relation['arg2']}")
                continue
            if relation['type'] in SYMMETRIC_RELATIONS:
                idx1, idx2 = sorted([idx1, idx2])
            pair = (idx1, idx2)
            if pair not in pos_pairs:
                rel_data['pairs'].append(pair)
                rel_data['labels'].append(RELATION_TYPES[relation['type']])
                pos_pairs.add(pair)

        output = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'ner_labels': ner_labels,
            'rel_data': rel_data,
            'text': text,
            'offset_mapping': encoding['offset_mapping'].squeeze(0)
        }

        return output

def collate_fn(batch, device=None):
    # All elements already padded to max_length
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    ner_labels = torch.stack([item['ner_labels'] for item in batch])
    offset_mapping = torch.stack([item['offset_mapping'] for item in batch])

    device = device or input_ids.device

    rel_data = []
    # Собираем rel_data как список словарей
    for item in batch:
        if item['rel_data']['pairs']:
            pairs = torch.tensor(item['rel_data']['pairs'],
                                 dtype=torch.long).to(device)
        else:
            pairs = torch.zeros((0, 2), dtype=torch.long, device=device)

            # Обрабатываем метки
        if item['rel_data']['labels']:
            labels = torch.tensor(item['rel_data']['labels'],
                                  dtype=torch.long).to(device)
        else:
            labels = torch.zeros(0, dtype=torch.long, device=device)

        rel_entry = {
            'entities': item['rel_data']['entities'],
            'pairs': pairs,
            'labels': labels,
            'rel_types': [RELATION_TYPES_INV.get(l, 'UNK')
                          for l in item['rel_data']['labels']] if item['rel_data']['labels'] else []
        }
        rel_data.append(rel_entry)

    return {
        'input_ids': input_ids.to(device),
        'attention_mask': attention_mask.to(device),
        'ner_labels': ner_labels.to(device),
        'rel_data': rel_data,
        'texts': [item['text'] for item in batch],
        'offset_mapping': offset_mapping.to(device)
    }


def _update_metrics(self, metrics, outputs, batch):
    """Обновление метрик NER и отношений"""
    # Метрики NER
    preds = outputs.get("ner_preds", [])
    trues = batch["ner_labels"]
    mask = batch["attention_mask"].bool()

    for pred, true, m in zip(preds, trues, mask):
        valid = m.sum().item()
        metrics["ner"]["correct"] += (pred[:valid] == true[:valid]).sum().item()
        metrics["ner"]["total"] += valid

    # Метрики отношений
    if "rel_probs" in outputs:
        for rel_type, probs_labels in outputs["rel_probs"].items():
            if probs_labels:
                probs, labels = zip(*probs_labels)
                preds = [p > 0.5 for p in probs]
                metrics["rel"]["correct"] += sum(p == l for p, l in zip(preds, labels))
                metrics["rel"]["total"] += len(labels)

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = {
        "batch_size": 8,  # Увеличенный размер батча
        "num_epochs": 1,
        "lr_bert": 2e-5,
        "lr_ner": 3e-4,
        "lr_gat": 5e-4,
        "lr_rel": 5e-4,
        "grad_clip": 1.0,
        "warmup_steps": 500,
        "balance_alpha": 0.7  # Коэффициент баланса NER vs RE
    }

    tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
    model = NERRelationModel(
        model_name="DeepPavlov/rubert-base-cased",
        entity_types=list(ENTITY_TYPES.keys()),
        relation_types=list(RELATION_TYPES.keys())
    ).to(device)

    # Загрузка данных
    train_dataset = NERELDataset("NEREL/NEREL-v1.0/train", tokenizer)

    relation_counts = [len(sample['rel_data']['labels']) for sample in train_dataset]
    median_count = np.median([c for c in relation_counts if c > 0])
    sample_weights = [
        0.3 + 0.7 * (min(count, 2 * median_count) / (2 * median_count))
        for count in relation_counts
    ]
    sampler = WeightedRandomSampler(sample_weights, len(train_dataset), replacement=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        collate_fn=collate_fn,
        sampler=sampler,
        pin_memory=True
    )

    optimizer = AdamW([
        {"params": model.bert.parameters(), "lr": config["lr_bert"]},
        {"params": model.ner_classifier.parameters(), "lr": config["lr_ner"]},
        {"params": model.crf.parameters(), "lr": config["lr_ner"]},
        {"params": chain(model.gat1.parameters(), model.gat2.parameters()),
         "lr": config["lr_gat"]},
        {"params": model.rel_classifier.parameters(), "lr": config["lr_rel"]}
    ])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config["warmup_steps"],
        num_training_steps=len(train_loader) * config["num_epochs"]
    )

    # Цикл обучения
    for epoch in range(config["num_epochs"]):
        model.train()
        epoch_loss = 0
        metrics = {
            "ner": {"correct": 0, "total": 0},
            "rel": {"correct": 0, "total": 0}
        }
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        for batch in progress_bar:
            # Перенос данных с автоматической оптимизацие
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }

            # Forward pass с обработкой ошибок
            try:
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    ner_labels=batch["ner_labels"],
                    rel_data=batch["rel_data"]
                )
                if outputs["loss"] is None:
                    continue
            except RuntimeError as e:
                logger.error(f"Batch failed: {str(e)}")
                continue

            # Backward pass с градиентным клиппингом
            loss = outputs["loss"] * config["balance_alpha"] + \
                   outputs.get("rel_loss", 0) * (1 - config["balance_alpha"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config["grad_clip"]
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # Обновление метрик
            epoch_loss += loss.item()
            self._update_metrics(metrics, outputs, batch)

            # Обновление progress bar
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "ner_acc": f"{metrics['ner']['correct'] / metrics['ner']['total']:.2%}",
                "rel_acc": f"{metrics['rel']['correct'] / metrics['rel']['total']:.2%}"
            })

        # Логирование
        print(f"\nEpoch {epoch + 1} Results:")
        print(f"Train Loss: {epoch_loss / len(train_loader):.4f}")
        print(f"NER: Acc={metrics['ner']['correct'] / metrics['ner']['total']:.2%}")
        print(f"Relations: Acc={metrics['rel']['correct'] / metrics['rel']['total']:.2%}")

    # Финальное сохранение
    self._save_model(model, tokenizer, "final_model")
    print("Model saved!")
    return model, tokenizer

def format_relation(arg1_text, arg2_text, rel_type, confidence):
    conf_str = f"{confidence:.2f}"
    return f"~~~~~~~> {colored(arg1_text, 'black', attrs=['bold'])} --{rel_type}({conf_str})--> {colored(arg2_text, 'black', attrs=['bold'])}"

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
        color = ENTITY_COLORS.get(ent['type'], 'black')
        entity_str = colored(ent['text'], color, attrs=["bold"])
        result_text += f"[{entity_str}]({ent['type']})"

        last_pos = ent['end_char']

    result_text += text[last_pos:]

    # Format relations
    rel_lines = []
    for rel in relations:
        rel_lines.append(format_relation(rel['arg1']['text'], rel['arg2']['text'], rel['type'], rel['confidence']))

    return "\n" + "\n".join(rel_lines) + "\n\n" + result_text


class RelationPredictor:
    def __init__(self, model, tokenizer, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.tokenizer = tokenizer
        self.idx_to_type = {v: k for k, v in ENTITY_TYPES.items()}
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text, relation_threshold=None):
        """Main prediction method"""
        encoding = self.tokenizer(text, relation_threshold).to(self.device)
        relation_threshold = {**RELATION_THRESHOLDS, **(relation_threshold or {})}
        with torch.no_grad():
            outputs = self.model(encoding['input_ids'], encoding['attention_mask'])
            entities = self._extract_entities(text, encoding, outputs)

            if len(entities) < 2:
                return {'text': text, 'entities': entities, 'relations': []}

            relations = self._predict_relations(encoding, entities, relation_threshold)

        return {
            'text': text,
            'entities': entities,
            'relations': sorted(relations.values(), key=lambda x: -x['confidence'])
        }

    def _tokenize(self, text):
        return self.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            max_length=512,
            truncation=True
        ).to(self.device)

    def _extract_entities(self, text, encoding, outputs):
        """Extract and normalize entities from model output"""
        mask = encoding['attention_mask'].bool()
        preds = self.model.crf.decode(outputs['ner_logits'], mask=mask)[0]

        entities = []
        current = None

        for i, (token, pred) in enumerate(zip(
                self.tokenizer.convert_ids_to_tokens(encoding['input_ids'][0]),
                preds
        )):
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                if current: entities.append(current)
                current = None
                continue

            start, end = offset_mapping[i]
            is_begin = pred % 2 == 1
            is_inside = pred % 2 == 0 and pred != 0

            if is_begin:
                if current: entities.append(current)
                current = {
                    'id': f"T{len(entities)}",
                    'type': self.idx_to_type[(pred + 1) // 2],
                    'start': i, 'end': i,
                    'start_char': start, 'end_char': end,
                    'token_ids': [i]
                }
            elif is_inside and current:
                expected_type = self.idx_to_type[pred // 2]
                if current['type'] == expected_type:
                    current['end'] = i
                    current['end_char'] = end
                    current['token_ids'].append(i)
                else:
                    entities.append(current)
                    current = None
            elif current:
                entities.append(current)
                current = None

        if current: entities.append(current)

        # Add text spans
        for e in entities:
            e['text'] = text[e['start_char']:e['end_char']]

        return entities

    def _predict_relations(self, encoding, entities, thresholds):
        """Predict relations between extracted entities"""
        # Get embeddings
        seq_out = model.bert(encoding['input_ids'], encoding['attention_mask']).last_hidden_state
        entity_embs = torch.stack([
            seq_out[0, e['token_ids']].mean(0) for e in entities
        ])

        # Apply GAT
        edge_index = torch.tensor([
            [i, j] for i in range(len(entities))
            for j in range(len(entities)) if i != j
        ]).t().to(device)

        x = model.gat1(entity_embs, edge_index)
        x = model.norm1(F.elu(x))
        x = model.gat2(x, edge_index)
        x = model.norm2(F.elu(x))

        # Predict relations
        relations = {}
        cls_token = seq_out[:, 0, :].squeeze(0)

        for rel_type, rel_idx in RELATION_TYPES.items():
            for i, j in self._get_valid_pairs(entities, rel_type):
                src, tgt = (j, i) if rel_type == 'FOUNDED_BY' else (i, j)

                logit = model.rel_classifier(torch.cat([
                    x[src], x[tgt],
                    model.rel_type_emb(torch.tensor(rel_idx, device=device)),
                    cls_token
                ]))

                prob = torch.sigmoid(logit).item()
                if prob > thresholds.get(rel_type, 0.5):
                    key = (entities[src]['id'], entities[tgt]['id'], rel_type)
                    if key not in relations or prob > relations[key]['confidence']:
                        relations[key] = {
                            'type': rel_type,
                            'arg1': entities[src],
                            'arg2': entities[tgt],
                            'confidence': prob
                        }
        return relations

    def _get_valid_pairs(self, entities, rel_type):
        """Filter valid entity pairs for relation type"""
        valid = []
        for i, e1 in enumerate(entities):
            for j, e2 in enumerate(entities):
                if i != j and (e1['type'], e2['type']) in VALID_COMB.get(rel_type, []):
                    valid.append((i, j))
        return valid

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

    # Для загрузки модели
    loaded_model = NERRelationModel.from_pretrained("saved_model", device="cpu")
    loaded_tokenizer = AutoTokenizer.from_pretrained("saved_model")

    # Использование модели
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
        print("\n" + "=" * 80)
        print(f"Processing text: '{text}'")
        result = predict(text, loaded_model, loaded_tokenizer)
        print(visualize_prediction_colored(result))

    # / home / guest / NER_Saved_Model / NEREL.py: 315: UserWarning: To
    # copy
    # construct
    # from a tensor, it is recommended
    # to
    # use
    # sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather
    # than
    # torch.tensor(sourceTensor).
    # rel_type_tensor = self.rel_type_emb(torch.tensor(rel_type_idx, device=device))
