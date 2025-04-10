import os
import json
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from constants import ENTITY_TYPES, RELATION_TYPES, RELATION_TYPES_INV, MAX_SEQ_LENGTH

# Датасет для совместного извлечения именованных сущностей и отношений между ними
class NERELDataset(Dataset):
    
    def __init__(self, data_dir: str, tokenizer: AutoTokenizer, max_length: int = MAX_SEQ_LENGTH):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self._load_samples()

    def __len__(self) -> int:
        return len(self.samples)

    # Возвращает один элемент датасета в формате, готовом для модели.
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        encoding = self._tokenize_text(sample['text'])
        ner_labels = self._create_ner_labels(encoding, sample['entities'])
        rel_data = self._prepare_relation_data(sample, encoding)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'ner_labels': ner_labels,
            'rel_data': rel_data,
            'text': sample['text'],
            'offset_mapping': encoding['offset_mapping'].squeeze(0)
        }
    

    # Загружает и парсит все файлы из директории.
    def _load_samples(self) -> List[Dict]:
        samples = []
        for filename in os.listdir(self.data_dir):
            if not filename.endswith('.txt'):
                continue
                
            ann_path = os.path.join(self.data_dir, filename.replace('.txt', '.ann'))
            if not os.path.exists(ann_path):
                continue
                
            with open(os.path.join(self.data_dir, filename), 'r', encoding='utf-8') as f:
                text = f.read()
            
            entities, relations = self._parse_ann_file(ann_path, text)
            samples.append({
                'text': text,
                'entities': entities,
                'relations': relations
            })
        
        return samples

    # Парсит файл с аннотациями (.ann).
    def _parse_ann_file(self, ann_path: str, text: str) -> Tuple[List[Dict], List[Dict]]:
        entities, relations = [], []
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
                        relations.append(relation)
        
        return entities, relations

    # Парсит строку с сущностью
    def _parse_entity_line(self, line: str, text: str) -> Optional[Dict]:
        parts = line.split('\t')
        if len(parts) < 3:
            return None

        entity_id = parts[0]
        type_and_span = parts[1].split()
        entity_type = type_and_span[0]
        
        if entity_type not in ENTITY_TYPES:
            return None

        start, end = int(type_and_span[1]), int(type_and_span[-1])
        entity_text = parts[2]

        # Корректировка границ, если текст не совпадает
        if text[start:end] != entity_text:
            found_pos = text.find(entity_text)
            if found_pos != -1:
                start = found_pos
                end = found_pos + len(entity_text)
            else:
                return None

        return {
            'id': entity_id,
            'type': entity_type,
            'start': start,
            'end': end,
            'text': entity_text
        }

    # Парсит строку с отношением
    def _parse_relation_line(self, line: str, entity_map: Dict) -> Optional[Dict]:
        parts = line.split('\t')
        if len(parts) < 2:
            return None
            
        rel_info = parts[1].split()
        if len(rel_info) < 3:
            return None
            
        rel_type = rel_info[0]
        arg1 = rel_info[1].split(':')[1] if ':' in rel_info[1] else None
        arg2 = rel_info[2].split(':')[1] if ':' in rel_info[2] else None
        
        if not arg1 or not arg2 or rel_type not in RELATION_TYPES:
            return None
            
        if arg1 in entity_map and arg2 in entity_map:
            return {
                'type': rel_type,
                'arg1': arg1,
                'arg2': arg2
            }
        return None

    # Токенизация текста
    def _tokenize_text(self, text: str) -> Dict:
        return self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_offsets_mapping=True,
            return_tensors='pt'
        )

    # Создание меток для NER в BIO
    def _create_ner_labels(self, encoding: Dict, entities: List[Dict]) -> torch.Tensor:
        ner_labels = torch.zeros(self.max_length, dtype=torch.long)
        
        for entity in entities:
            start_token, end_token = self._find_entity_span(entity, encoding['offset_mapping'][0])
            
            if start_token is not None and end_token is not None:
                self._apply_bio_labels(ner_labels, start_token, end_token, entity['type'])
        
        return ner_labels

    # Границы сущности в токенах
    def _find_entity_span(self, entity: Dict, offset_mapping: List[Tuple[int, int]]) -> Tuple[Optional[int], Optional[int]]:
        start_token = end_token = None
        
        for i, (start, end) in enumerate(offset_mapping):
            if start <= entity['start'] < end and start_token is None:
                start_token = i
            if start < entity['end'] <= end and end_token is None:
                end_token = i
            if start >= entity['end']:
                break
                
        return start_token, end_token

    # Применение BIO-разметки
    def _apply_bio_labels(self, ner_labels: torch.Tensor, start: int, end: int, entity_type: str) -> None:
        label_map = {
            'PERSON': (1, 2),
            'PROFESSION': (3, 4),
            'ORGANIZATION': (5, 6),
            'FAMILY': (7, 8),
            'LOCATION': (9, 10)
        }
        
        if entity_type in label_map:
            b_label, i_label = label_map[entity_type]
            ner_labels[start] = b_label
            ner_labels[start+1:end+1] = i_label

    # Подготовка данных об отношениях между сущностями
    def _prepare_relation_data(self, sample: Dict, encoding: Dict) -> Dict:
        token_entities = self._align_entities_with_tokens(sample['entities'], encoding['offset_mapping'][0])
        token_entity_id_to_idx = {e['id']: i for i, e in enumerate(token_entities)}
        
        pairs, labels = [], []
        for relation in sample['relations']:
            arg1_idx = token_entity_id_to_idx.get(relation['arg1'], -1)
            arg2_idx = token_entity_id_to_idx.get(relation['arg2'], -1)
            
            if arg1_idx != -1 and arg2_idx != -1:
                pairs.append((arg1_idx, arg2_idx))
                labels.append(RELATION_TYPES[relation['type']])
        
        return {
            'entities': token_entities,
            'pairs': pairs,
            'labels': labels
        }

    # Сопоставление данных об отношениях между сущностями
    def _align_entities_with_tokens(self, entities: List[Dict], offset_mapping: List[Tuple[int, int]]) -> List[Dict]:
        token_entities = []
        
        for entity in entities:
            start_token, end_token = self._find_entity_span(entity, offset_mapping)
            
            if start_token is not None and end_token is not None:
                token_entities.append({
                    'start': start_token,
                    'end': end_token,
                    'type': entity['type'],
                    'id': entity['id']
                })
        
        return token_entities


def collate_fn(batch: List[Dict]) -> Dict:
    """Функция для объединения элементов батча.
    
    Args:
        batch: Список элементов датасета
        
    Returns:
        Словарь с объединенными тензорами:
        - input_ids: Токенизированный текст
        - attention_mask: Маска внимания
        - ner_labels: Метки для NER
        - rel_data: Данные для обучения отношений
        - texts: Исходные тексты
        - offset_mapping: Соответствие токенов и символов
    """
    # Тензоры уже паддингованы до max_length
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    ner_labels = torch.stack([item['ner_labels'] for item in batch])
    offset_mapping = torch.stack([item['offset_mapping'] for item in batch])

    # Подготовка данных об отношениях
    rel_data = []
    for item in batch:
        rel_entry = {
            'entities': item['rel_data']['entities'],
            'pairs': torch.tensor(item['rel_data']['pairs'], dtype=torch.long) 
                     if item['rel_data']['pairs'] else torch.zeros((0, 2), dtype=torch.long),
            'labels': torch.tensor(item['rel_data']['labels'], dtype=torch.long) 
                     if item['rel_data']['labels'] else torch.zeros(0, dtype=torch.long),
            'rel_types': [
                RELATION_TYPES_INV.get(l, 'UNK') 
                for l in item['rel_data'].get('labels', [])
            ]
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