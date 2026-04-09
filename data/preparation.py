HateSpeech-Explainer/
├── data/               # (Veri setini buraya kaydetmek istersen)
├── notebooks/          # .ipynb (Jupyter) dosyaların için
│   └── 01_data_loading.ipynb 
├── src/                # .py (Python) scriptlerin için
│   └── data_prep.py   import requests
import json
from datasets import Dataset

# 1. Ham verileri GitHub'dan çekiyoruz
dataset_url = "https://raw.githubusercontent.com/punyajoy/HateXplain/master/Data/dataset.json"
divisions_url = "https://raw.githubusercontent.com/punyajoy/HateXplain/master/Data/post_id_divisions.json"

data_raw = requests.get(dataset_url).json()
divisions = requests.get(divisions_url).json()

# 2. Veriyi train/val/test splitlerine göre ayırma fonksiyonu
def create_split(split_name):
    split_data = []
    for post_id in divisions[split_name]:
        item = data_raw[post_id]
        split_data.append({
            "id": post_id,
            "post_tokens": item["post_tokens"],
            "text": " ".join(item["post_tokens"]), # DistilBERT için metin hali [cite: 14, 15]
            "annotators": item["annotators"],
            "rationales": item["rationales"] # XAI analizi için kritik [cite: 12, 13]
        })
    return Dataset.from_list(split_data)

# 3. Datasetlerin oluşturulması
train_ds = create_split('train')
val_ds = create_split('val')
test_ds = create_split('test')

print(f"Eğitim seti boyutu: {len(train_ds)}")
├── requirements.txt    # requests, datasets, json kütüphanelerini buraya yaz
└── README.md           # Projenin "vitrini"
