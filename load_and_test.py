from NEREL import NERRelationModel
from NEREL import predict
from transformers import AutoTokenizer
if __name__ == "__main__":
    # Для загрузки модели
    loaded_model = NERRelationModel.from_pretrained("saved_model")
    loaded_tokenizer = AutoTokenizer.from_pretrained("saved_model")
    
    # Использование модели
    result = predict("По улице шел красивый человек, его имя было Мефодий. И был он счастлив. Работал этот чувак в яндексе, разработчиком. Или директором. Он пока не определился!", loaded_model, loaded_tokenizer)
    print("\nEntities:")
        for e in result['entities']:
            print(f"{e['type']}: {e['text']}")
        print("\nRelations:")
        for r in result['relations']:
            print(f"{r['type']}: {r['arg1']['text']} -> {r['arg2']['text']} (conf: {r['confidence']:.2f})")