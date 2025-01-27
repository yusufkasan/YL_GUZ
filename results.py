import json
import pandas as pd
import matplotlib.pyplot as plt

# JSON dosyasını oku
input_file = "ragas_results.json"
try:
    with open(input_file, "r", encoding="utf-8") as f:
        results = json.load(f)
except FileNotFoundError:
    print(f"{input_file} bulunamadı. Lütfen dosyanın varlığını kontrol edin.")
    exit()

# JSON verilerini pandas DataFrame'e dönüştürme
data = []
for entry in results:
    row = {"soru": entry["soru"], "model": entry["model"]}
    # Metriklerin değerlerini liste içinden çıkar ve düzleştir
    for metric, value in entry["metrics"].items():
        if isinstance(value, list) and len(value) > 0:  # Eğer listeyse ve boş değilse
            row[metric] = value[0]  # Listenin ilk elemanını al
        else:
            row[metric] = None  # Boş bir listeyse, None ekle
    data.append(row)

df = pd.DataFrame(data)

# NaN değerleri temizleme
df = df.dropna()

# Sadece sayısal sütunları seç
numeric_df = df.select_dtypes(include=["number"])

# Modellerin metrik ortalamalarını hesaplama
average_metrics = numeric_df.groupby(df["model"]).mean()

# Her model için genel ortalama hesaplama
average_metrics["Genel Ortalama"] = average_metrics.mean(axis=1)

# Ortalamaları tablo olarak göster
print("\nModellerin Ortalama Metrik Sonuçları:")
print(average_metrics)

# Ortalamaları bir tabloya kaydetme
average_metrics.to_csv("average_metrics_with_general.csv", encoding="utf-8")

# Her metrik için 5 modelin başarısını ve genel ortalamayı sütun grafiği olarak görselleştirme
for metric in average_metrics.columns:
    plt.figure(figsize=(10, 6))
    if metric != "Genel Ortalama":  # Genel Ortalama için ayrı bir grafikte işlem yapılacak
        plt.bar(
            average_metrics.index,
            average_metrics[metric],
            color="orange",
            edgecolor="black",
            label=f"{metric}"
        )
        plt.title(f"Model Performansı - {metric}", fontsize=16, fontweight="bold")
        plt.xlabel("Model", fontsize=14, fontweight="bold")
        plt.ylabel("Ortalama Skor", fontsize=14, fontweight="bold")
        plt.ylim(0, 1)  # Skorlar genelde 0 ile 1 arasında olur
        plt.xticks(rotation=45, fontsize=12, fontweight="bold")
        plt.yticks(fontsize=12, fontweight="bold")
        plt.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.7)
        plt.legend(fontsize=12)

        # Grafiği düzenle ve kaydet
        plt.tight_layout()
        plt.savefig(f"{metric}_performance.png")  # Her metrik için grafiği kaydetme
        plt.show()

# Genel ortalama için ayrı bir sütun grafiği
plt.figure(figsize=(10, 6))
plt.bar(
    average_metrics.index,
    average_metrics["Genel Ortalama"],
    color="blue",
    edgecolor="black",
    label="Genel Ortalama"
)
plt.title("Model Performansı - Genel Ortalama", fontsize=16, fontweight="bold")
plt.xlabel("Model", fontsize=14, fontweight="bold")
plt.ylabel("Ortalama Skor", fontsize=14, fontweight="bold")
plt.ylim(0, 1)
plt.xticks(rotation=45, fontsize=12, fontweight="bold")
plt.yticks(fontsize=12, fontweight="bold")
plt.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.7)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("genel_ortalama_performance.png")
plt.show()
