import os
import json
import torch
from transformers import AutoTokenizer, BertConfig
from typing import Dict, Optional

# Сохраняет модель, конфигурацию и токенизатор
def save_model(model, save_dir: str, tokenizer=None) -> None:
    os.makedirs(save_dir, exist_ok=True)
    
    # Сохранение весов модели
    model_path = os.path.join(save_dir, "pytorch_model.bin")
    torch.save(model.state_dict(), model_path)
    
    # Сохранение конфигурации модели
    config = {
        "model_type": "bert-ner-rel",
        "model_name": getattr(model.bert, "name_or_path", "custom"),
        "num_ner_labels": model.num_ner_labels,
        "num_rel_labels": len(RELATION_TYPES),
        "bert_config": model.bert.config.to_diff_dict(),
        "model_config": {
            "gat_hidden_size": GAT_CONFIG['hidden_size'],
            "gat_heads": GAT_CONFIG['heads']
        }
    }
    
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    # 3. Сохраняем токенизатор
    if tokenizer is not None:
        tokenizer.save_pretrained(save_dir)

# Загрузка модел
def load_model(model_dir: str, device="cuda"):
    try:
        device = torch.device(device)
        
        # Загрузка конфигурации
        config_path = os.path.join(model_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
            
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        # Инициализация BERT
        bert_config = BertConfig.from_dict(config["bert_config"])
        bert = AutoModel.from_pretrained(
            model_dir,
            config=bert_config,
            ignore_mismatched_sizes=True
        )
        
        # Создание экземпляра модели
        model = NERRelationModel(
            model_name=config.get("model_name", DEFAULT_MODEL_NAME),
            num_ner_labels=config.get("num_ner_labels", len(ENTITY_TYPES)*2+1),
            num_rel_labels=config.get("num_rel_labels", len(RELATION_TYPES))
        ).to(device)
        
        # Загрузка весов
        model_path = os.path.join(model_dir, "pytorch_model.bin")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found at {model_path}")
            
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        
        # Загрузка BERT
        model.bert = bert.to(device)
        
        model.eval()
        return model
        
    except Exception as e:
        raise RuntimeError(f"Error loading model from {model_dir}: {str(e)}")

# Создание оптимизатора для компонентов модели
def get_optimizer(model):
    return AdamW([
        {'params': model.bert.parameters(), 'lr': TRAINING_CONFIG['bert_lr']},
        {'params': model.ner_classifier.parameters(), 'lr': TRAINING_CONFIG['classifier_lr']},
        {'params': model.crf.parameters(), 'lr': TRAINING_CONFIG['classifier_lr']},
        {'params': model.gat1.parameters(), 'lr': TRAINING_CONFIG['gat_lr']},
        {'params': model.gat2.parameters(), 'lr': TRAINING_CONFIG['gat_lr']},
        {'params': model.rel_classifiers.parameters(), 'lr': TRAINING_CONFIG['gat_lr']}
    ])