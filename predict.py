import torch
from typing import Dict, Optional
from collections import defaultdict

from model import NERRelationModel
from utils import load_model
from constants import RELATION_THRESHOLDS, ENTITY_TYPES, RELATION_TYPES_INV

# Предсказание сущностей и отношений в тексте
def predict(text: str, model, tokenizer, device="cuda", relation_thresholds: Optional[Dict]=None) -> Dict:
    relation_thresholds = {**RELATION_THRESHOLDS, **(relation_thresholds or {})}
    
    # Токенизация
    encoding = tokenizer(
        text, 
        return_tensors="pt", 
        return_offsets_mapping=True, 
        max_length=512,
        truncation=True
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    offset_mapping = encoding['offset_mapping'][0].cpu().numpy()
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)

    # Декодирование NER
    mask = attention_mask.bool()
    ner_preds = model.crf.decode(outputs['ner_logits'], mask=mask)[0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0], skip_special_tokens=True)

    # Извлечение сущностей
    entities = extract_entities(tokens, ner_preds, offset_mapping, text)
    
    if len(entities) < 2:
        return {
            'text': text,
            'entities': entities,
            'relations': []
        }

    # Извлечение отношений
    relations = extract_relations(
        model, 
        input_ids, 
        attention_mask, 
        entities, 
        device,
        relation_thresholds
    )
    
    # Удаление дубликатов и сортировка по уверенности
    unique_relations = remove_duplicate_relations(relations)
    
    return {
        'text': text,
        'entities': entities,
        'relations': unique_relations
    }

# Извлечение сущностей из предсказаний NER
def extract_entities(tokens, ner_preds, offset_mapping, text):
    entities = []
    current_entity = None
    entity_id = 0
    
    for i, (token, pred) in enumerate(zip(tokens, ner_preds)):
        if token in ['[CLS]', '[SEP]', '[PAD]']:
            if current_entity:
                entities.append(current_entity)
                current_entity = None
            continue

        token_start, token_end = offset_mapping[i]
        
        if pred % 2 == 1:  # Начало сущности (B-)
            if current_entity:
                entities.append(current_entity)
            
            entity_type = get_entity_type(pred)
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
                
        elif pred % 2 == 0 and pred != 0:  # Продолжение сущности (I-)
            if current_entity:
                expected_type = get_entity_type(pred - 1)
                if expected_type and current_entity['type'] == expected_type:
                    current_entity['end'] = i
                    current_entity['end_char'] = token_end
                    current_entity['token_ids'].append(i)
                else:
                    entities.append(current_entity)
                    current_entity = None
        else:  # Вне сущности (O)
            if current_entity:
                entities.append(current_entity)
                current_entity = None
    
    if current_entity:
        entities.append(current_entity)

    # Добавление текста сущностей
    for entity in entities:
        entity['text'] = text[entity['start_char']:entity['end_char']]

    return entities

# Возвращает тип сущности по предсказанию
def get_entity_type(pred):
    if pred == 1: return "PERSON"
    elif pred == 3: return "PROFESSION"
    elif pred == 5: return "ORGANIZATION"
    elif pred == 7: return "FAMILY"
    elif pred == 9: return "LOCATION"
    return None
# Извлекает отношения между сущностями
def extract_relations(model, input_ids, attention_mask, entities, device, thresholds):
    relations = []
    sequence_output = model.bert(input_ids, attention_mask).last_hidden_state

    # Эмбеддинги сущностей
    entity_embeddings = []
    for e in entities:
        token_embeddings = sequence_output[0, e['token_ids']]
        entity_embed = token_embeddings.mean(dim=0)
        entity_embeddings.append(entity_embed)
    
    entity_embeddings = torch.stack(entity_embeddings).to(device)
    
    # Построение графа
    edge_index = torch.tensor([
        [i, j] for i in range(len(entities)) 
        for j in range(len(entities)) if i != j
    ], dtype=torch.long).t().contiguous().to(device)

    # Применение GAT
    x = model.gat1(entity_embeddings, edge_index)
    x = F.elu(x)
    x = F.dropout(x, p=0.3, training=False)
    x = model.gat2(x, edge_index)
    x = F.elu(x)
    
    # Предсказание отношений
    for rel_type in RELATION_TYPES_INV.values():
        for i, e1 in enumerate(entities):
            for j, e2 in enumerate(entities):
                if i != j:
                    if rel_type == 'FOUNDED_BY':
                        src, tgt = j, i
                    else:
                        src, tgt = i, j
                        
                    pair_features = torch.cat([x[src], x[tgt]])
                    logit = model.rel_classifiers[rel_type](pair_features)
                    prob = torch.sigmoid(logit).item()
                    
                    if prob > thresholds.get(rel_type, 0.5):
                        relations.append({
                            'type': rel_type,
                            'arg1_id': entities[src]['id'],
                            'arg2_id': entities[tgt]['id'],
                            'arg1': entities[src],
                            'arg2': entities[tgt],
                            'confidence': prob
                        })
    return relations

def remove_duplicate_relations(relations):
    unique_relations = {}
    for rel in relations:
        key = (rel['arg1_id'], rel['arg2_id'], rel['type'])
        if key not in unique_relations or rel['confidence'] > unique_relations[key]['confidence']:
            unique_relations[key] = rel
    
    return sorted(unique_relations.values(), key=lambda x: x['confidence'], reverse=True)

if __name__ == "__main__":
    # Пример использования
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model("saved_model", device)
    tokenizer = AutoTokenizer.from_pretrained("saved_model")
    
    test_texts = [
        "Айрат Мурзагалиев, заместитель начальника управления президента РФ, встретился с главой администрации Уфы.",
        "Иван Петров работает программистом в компании Яндекс.",
        "Доктор Сидоров принял пациентку Ковалеву в городской больнице.",
        "Директор сводного экономического департамента Банка России Надежда Иванова назначена также на должность заместителя председателя ЦБ."
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