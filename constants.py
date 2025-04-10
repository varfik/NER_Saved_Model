"""
Модуль для хранения констант, используемых в проекте.
"""

# Типы сущностей
ENTITY_TYPES = {
    'PERSON': 1,
    'PROFESSION': 2,
    'ORGANIZATION': 3,
    'FAMILY': 4,
    'LOCATION': 5
}

# Типы отношений
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

# Обратное отображение для отношений
RELATION_TYPES_INV = {v: k for k, v in RELATION_TYPES.items()}

# Пороги уверенности для разных типов отношений
RELATION_THRESHOLDS = {
    'WORKS_AS': 0.85,
    'MEMBER_OF': 0.8,
    'FOUNDED_BY': 0.8,
    'SPOUSE': 0.9,
    'PARENT_OF': 0.85,
    'SIBLING': 0.9,
    'PART_OF': 0.7,
    'WORKPLACE': 0.7,
    'RELATIVE': 0.75
}

# Параметры модели по умолчанию
DEFAULT_MODEL_NAME = "DeepPavlov/rubert-base-cased"
DEFAULT_BATCH_SIZE = 8
MAX_SEQ_LENGTH = 512

# BIO-метки для NER
BIO_LABELS = {
    'O': 0,
    'B-PER': 1,
    'I-PER': 2,
    'B-PROF': 3,
    'I-PROF': 4,
    'B-ORG': 5,
    'I-ORG': 6,
    'B-FAM': 7,
    'I-FAM': 8,
    'B-LOC': 9,
    'I-LOC': 10
}

# Параметры обучения
TRAINING_CONFIG = {
    'bert_lr': 3e-5,
    'classifier_lr': 5e-5,
    'gat_lr': 1e-3,
    'dropout_rate': 0.3,
    'epochs': 10
}

GAT_CONFIG = {
    'hidden_size': 128,
    'heads': 4,
    'output_size': 64
}