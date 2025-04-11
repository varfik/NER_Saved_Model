from another import NERRelationModel
from another import predict
from transformers import AutoTokenizer

if __name__ == "__main__":
    # Для загрузки модели
    loaded_model = NERRelationModel.from_pretrained("saved_model")
    loaded_tokenizer = AutoTokenizer.from_pretrained("saved_model")
    
    # Текст для анализа
    text = """"""    
    # Использование модели
    result = predict(text, loaded_model, loaded_tokenizer)
    
    # Вывод результатов
    print("\nEntities:")
    for entity in result['entities']:
        print(f"{entity['type']}: {entity['text']}")
    
    print("\nRelations:")
    for relation in result['relations']:
        print(f"{relation['type']}: {relation['arg1']['text']} -> {relation['arg2']['text']} (conf: {relation['confidence']:.2f})")