import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    answer_correctness,
    context_precision,
    context_recall
)
import json

# OpenAI API anahtarını tanımlayın
os.environ["OPENAI_API_KEY"] = ""

# Excel dosyasını oku
file_path = 'sorucevap.xlsx'
try:
    data = pd.read_excel(file_path)
except FileNotFoundError:
    print(f"{file_path} bulunamadı. Lütfen doğru dosya yolunu kontrol edin.")
    exit()

# Manuel olarak seçilen Ragas metrikleri
metrics = [
    faithfulness,
    answer_relevancy,
    answer_correctness,
    context_precision,
    context_recall
]

# Sonuçları yükle veya yeni bir liste oluştur
output_file = "ragas_results.json"
try:
    # Önceden kaydedilen sonuçları yükle
    with open(output_file, "r", encoding="utf-8") as f:
        results_list = json.load(f)
except FileNotFoundError:
    # Dosya yoksa yeni bir liste başlat
    results_list = []

# Her satır ve model için metrikleri işleme
for idx, row in data.iterrows():
    try:
        # Bağlamları al
        retrieved_contexts = [row[f"paragraf_{i}"] for i in range(1, 6)]

        # Her model için metrikleri hesapla
        for model in ["troubadour_cevap", "matrixportal_cevap", "erdiari_cevap", "cosmos_cevap", "mradermacher_cevap"]:
            single_data = {
                "user_input": row["soru"],
                "response": row[model],
                "retrieved_contexts": retrieved_contexts,
                "reference": row["gercek_cevap"],
                "model": model
            }

            # Hugging Face Dataset formatına dönüştürme
            dataset = Dataset.from_pandas(pd.DataFrame([single_data]))

            # Ragas ile değerlendirme
            results = evaluate(dataset, metrics)

            # Sonuçları kaydetme
            result_entry = {
                "soru": row["soru"],
                "model": model,
                "metrics": {metric.name: results[metric.name] for metric in metrics}
            }

            # Konsola yazdır
            print(f"İşlenen satır: {idx + 1}, Model: {model}")
            print(json.dumps(result_entry, ensure_ascii=False, indent=4))

            # Sonucu listeye ekle
            results_list.append(result_entry)

            # Her sonuçtan sonra JSON dosyasına yaz
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results_list, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Satır işlenirken hata oluştu: {e}")
        continue

print(f"Tüm sonuçlar {output_file} dosyasına başarıyla kaydedildi.")
