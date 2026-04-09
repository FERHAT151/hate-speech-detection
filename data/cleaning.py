def check_true_disagreement(example):
    labels = [ann["label"] for ann in example["annotators"]]
    # Eğer benzersiz etiket sayısı 3 ise (herkes farklı bir şey demişse)
    if len(set(labels)) == 3:
        return True
    return False

disagreed_indices = [i for i, x in enumerate(train_ds) if check_true_disagreement(x)]

print(f"Gerçek Anlaşmazlık Sayısı: {len(disagreed_indices)}")
print(f"Toplam Veri İçindeki Oranı: %{len(disagreed_indices)/len(train_ds)*100:.2f}")

if len(disagreed_indices) > 0:
    print("\nÖrnek bir anlaşmazlık (Index: ", disagreed_indices[0], "):")
    print(train_ds[disagreed_indices[0]]["annotators"])
