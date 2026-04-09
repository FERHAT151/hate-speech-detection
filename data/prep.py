# 1. Anlaşmazlık olanları temizle yani sadece çelişki olmayan veriler tutulur.
train_ds_clean = train_ds.filter(lambda x: not check_true_disagreement(x))

print(f"Temizlik sonrası train seti boyutu: {len(train_ds_clean)}")
