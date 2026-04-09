# 1. Anlaşmazlık olanları temizle yani sadece çelişki olmayan veriler tutulur.
train_ds_clean = train_ds.filter(lambda x: not check_true_disagreement(x))

print(f"Temizlik sonrası train seti boyutu: {len(train_ds_clean)}")
# 2. Nefret söylemi olup gerekçesi (rationale) olmayanları temizle
def has_rationales(example):
    labels = [ann["label"] for ann in example["annotators"]]
    majority = max(set(labels), key=labels.count)
    if majority in ["hatespeech", "offensive"]:
        # Rationale listesi boşsa veya sadece 0'lardan oluşuyorsa False döner
        if not example["rationales"] or sum([sum(r) for r in example["rationales"]]) == 0:
            return False
    return True

train_ds_clean = train_ds_clean.filter(has_rationales)
print(f"Temizlik sonrası train seti boyutu: {len(train_ds_clean)}")
