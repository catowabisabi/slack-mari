from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

# 加载预训练的RoBERTa模型和分词器
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base')

# 示例文本
text = "RoBERTa is a great NLP model!"

# 对文本进行编码
inputs = tokenizer(text, return_tensors="pt")

# 使用模型进行推理
outputs = model(**inputs)

# 获取预测结果
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)