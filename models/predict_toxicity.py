import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Загрузка модели и токенизатора rubert-tiny-toxicity
model_save_path = "/home/a/ds-phase-2/10-nlp/nlp-lstm-team/models"
tokenizer_toxicity = AutoTokenizer.from_pretrained(model_save_path)
model_toxicity = AutoModelForSequenceClassification.from_pretrained(model_save_path, use_safetensors=True)

# Устройство (GPU если доступно, иначе CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_toxicity.to(device)
model_toxicity.eval()

# Функция для предсказания токсичности
def predict_toxicity(phrase):
    inputs = tokenizer_toxicity(phrase, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model_toxicity(**inputs)
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    predicted_label = torch.argmax(probabilities, dim=-1).item()
    label_map = {0: "нетоксичный", 1: "токсичный"}
    prediction = label_map[predicted_label]
    return prediction, probabilities[0][predicted_label].item()