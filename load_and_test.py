from NEREL import NERRelationModel
from NEREL import predict
from transformers import AutoTokenizer

if __name__ == "__main__":
    # Для загрузки модели
    loaded_model = NERRelationModel.from_pretrained("saved_model")
    loaded_tokenizer = AutoTokenizer.from_pretrained("saved_model")
    
    # Текст для анализа
    text = """30 сентября во время митинга-концерта «Выбор людей. Вместе навсегда» в поддержку принятия в состав России ЛНР, ДНР, Запорожской и Херсонской областей с речью выступил российский актер Иван Охлобыстин. Его речь наделала много шума, в основном из-за слова «гойда». Это междометие быстро стало ключевой темой для мемов в социальных сетях. А филологи и языковеды не устают объяснять, что же такое «гойда» и какое у него происхождение."""    
    # Использование модели
    result = predict(text, loaded_model, loaded_tokenizer)
    
    # Вывод результатов
    print("\nEntities:")
    for entity in result['entities']:
        print(f"{entity['type']}: {entity['text']}")
    
    print("\nRelations:")
    for relation in result['relations']:
        print(f"{relation['type']}: {relation['arg1']['text']} -> {relation['arg2']['text']} (conf: {relation['confidence']:.2f})")