from NEREL import NERRelationModel
from NEREL import predict
from transformers import AutoTokenizer

if __name__ == "__main__":
    # Для загрузки модели
    loaded_model = NERRelationModel.from_pretrained("saved_model")
    loaded_tokenizer = AutoTokenizer.from_pretrained("saved_model")
    
    # Текст для анализа
    text = "Дмитрий работает в организации 'ЭкоФарм'. Елена является матерью Алексея. Компания 'Технологии будущего' является частью крупной корпорации, расположенной в Санкт-Петербурге. Сергей и Ольга - брат и сестра. Виктор является членом команды, расположенной в Москве. Михаил является супругом Ирины. Анна работает врачом в больнице 'Здоровье'."    
    # Использование модели
    result = predict(text, loaded_model, loaded_tokenizer)
    
    # Вывод результатов
    print("\nEntities:")
    for entity in result['entities']:
        print(f"{entity['type']}: {entity['text']}")
    
    print("\nRelations:")
    for relation in result['relations']:
        print(f"{relation['type']}: {relation['arg1']['text']} -> {relation['arg2']['text']} (conf: {relation['confidence']:.2f})")