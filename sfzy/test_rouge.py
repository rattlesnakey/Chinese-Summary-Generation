# import datasets
# rouge = datasets.load_metric('rouge')
from rouge import Rouge
# import jieba
rouge2 = Rouge()
# print(jieba.lcut("我很快乐"))
# predictions = [' '.join(jieba.lcut("我很快乐"))]
# references = [' '.join(jieba.lcut("你很快乐吗"))]

predictions = ' '.join(list("我很快乐"))
references = ' '.join(list("你很快乐吗，是的，我很快乐"))
print(predictions)
print(type(predictions))
print(references)
print(type(references))
# results2 = rouge.compute(predictions=predictions, references=references)
results = rouge2.get_scores([predictions, 'I love you'], [references, 'I love you too'], avg=True)
print(results)
results = {key: value['f'] * 100 for key, value in results.items()}
print(results)
# print(results2)