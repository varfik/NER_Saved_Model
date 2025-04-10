import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from tqdm.auto import tqdm

from dataset import NERELDataset, collate_fn
from model import NERRelationModel
from utils import save_model, get_optimizer
from constants import DEFAULT_MODEL_NAME, DEFAULT_BATCH_SIZE, TRAINING_CONFIG

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Инициализация модели и токенизатора
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
    model = NERRelationModel().to(device)

    # Загрузка данных
    train_dataset = NERELDataset("NEREL/NEREL-v1.1/train", tokenizer)

    # Создание взвешенного семплера для балансировки отношений
    sample_weights = []
    for sample in train_dataset:
        has_relations = len(sample['rel_data']['labels']) > 0
        sample_weights.append(1.0 if has_relations else 0.3)
    
    sampler = WeightedRandomSampler(sample_weights, len(train_dataset), replacement=True)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=DEFAULT_BATCH_SIZE, 
        collate_fn=collate_fn, 
        sampler=sampler
    )

    optimizer = get_optimizer(model)
    
    # Цикл обучения
    best_ner_f1 = 0
    for epoch in range(TRAINING_CONFIG['epochs']):
        model.train()
        epoch_loss = 0
        ner_correct = ner_total = 0
        rel_correct = rel_total = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
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
            
            # Метрики для NER
            with torch.no_grad():
                mask = attention_mask.bool()
                ner_preds = model.crf.decode(outputs['ner_logits'], mask=mask)
                
                for i in range(len(ner_preds)):
                    seq_len = mask[i].sum().item()
                    pred = torch.tensor(ner_preds[i][:seq_len], device=device)
                    true = ner_labels[i][:seq_len]
                    
                    ner_correct += (pred == true).sum().item()
                    ner_total += seq_len
            
            # Метрики для отношений
            if outputs['rel_probs']:
                for rel_type, probs in outputs['rel_probs'].items():
                    if len(probs) > 0:
                        if isinstance(probs, list):
                            probs = torch.cat([p.view(-1) for p in probs if p is not None]) if len(probs) > 0 else torch.tensor([], device=device)
                        
                        if len(probs) > 0:
                            preds = (torch.sigmoid(probs) > 0.5).long()
                            
                            targets = []
                            for item in batch['rel_data']:
                                if 'labels' in item and len(item['labels']) > 0:
                                    rel_labels = [l for l, t in zip(item['labels'], item.get('rel_types', [])) if t == rel_type]
                                    targets.extend(rel_labels)
                            
                            if len(targets) > 0:
                                min_len = min(len(preds), len(targets))
                                rel_correct += (preds[:min_len] == torch.tensor(targets[:min_len], device=device)).sum().item()
                                rel_total += min_len

        # Вывод метрик
        ner_acc = ner_correct / ner_total if ner_total > 0 else 0
        rel_acc = rel_correct / rel_total if rel_total > 0 else 0
        
        print(f"\nEpoch {epoch+1} Results:")
        print(f"Loss: {epoch_loss/len(train_loader):.4f}")
        print(f"NER Accuracy: {ner_acc:.2%} ({ner_correct}/{ner_total})")
        print(f"Relation Accuracy: {rel_acc:.2%} ({rel_correct}/{rel_total})")

    # Сохранение модели
    save_dir = "saved_model"
    tokenizer.save_pretrained(save_dir)
    save_model(model, save_dir, tokenizer)
    print(f"Model saved to {save_dir}")
    
    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = train_model()