# baseline/ — mevcut paketin OOF/test olasılıkları

`ensemble.py` bu klasördeki her alt dizini bir ensemble üyesi olarak görür ve
yeni modelini bunlarla birleştirip OOF'a göre karşılaştırır.

**Bu klasör bu depoda boş.** Mevcut 10 modelli paketin olasılık matrisleri
senin makinende; oraya kopyaladığında `ensemble.py` onları kendiliğinden bulur.

## Bir üyenin içermesi gerekenler

```
baseline/<uye_adi>/
  oof_prob.npy    (n_cache_rows, 5)  float, out-of-fold olasılıklar
  test_prob.npy   (n_test, 5)        float, fold ortalamalı test_public olasılıkları
  summary.json    (istege bagli)     tabloda gosterilecek meta bilgi
```

### Satır sırası

`oof_prob.npy` **cache'in tamamı kadar satıra** sahip olmalı ve satır sırası
`cache/index.csv` ile birebir aynı olmalı. Geliştirme kümesi dışındaki satırlar
(yani `test_public`) sıfır bırakılır — `ensemble.py` zaten `dev_idx` ile
indeksliyor.

`test_prob.npy` yalnızca `test_public` satırlarını, `index.csv`'deki sırayla
içerir.

`train.py` tam 5-fold koştuğunda `runs/<tag>/` altına tam olarak bu düzeni yazar.
Yani mevcut modellerin için de aynı formatı üretmen yeterli.

### Doğrulama

`ensemble.py` her üyeyi kabul etmeden önce şunları kontrol eder ve uymayanı
sessizce atmak yerine ekrana yazar:

- `oof_prob.npy` şekli `(n_cache_rows, 5)` mi
- `test_prob.npy` şekli `(n_test, 5)` mi
- geliştirme satırlarının en az %99'u dolu mu (yarım kalmış bir 5-fold buradan
  yakalanır)

### Uyarı

`baseline/` içindeki matrisler **aynı cache ve aynı ön işleme** ile üretilmiş
olmalı. `ecg_preprocess.py` değiştiyse eski matrisler başka bir ön işlemeye ait
demektir; `python prep.py` ile cache'i yeniden ürettikten sonra baseline
modellerinin OOF'unu da yeniden üretmen gerekir. Aksi halde ensemble ağırlıkları
birbiriyle uyumsuz iki dünyayı harmanlar ve OOF skoru yanıltıcı çıkar.
