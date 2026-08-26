# ecg_train — 12 derivasyonlu EKG sınıflandırma boru hattı

5 sınıf: **Normal(0) AFIB(1) AFL(2) LBBB(3) RBBB(4)**.
Sadece CPU. Teslim ONNX; kullanıcı makinesinde PyTorch gerekmez.

> **Bu depodaki durum:** kod tam ve çalışır durumda, ama **eğitilmiş ağırlık
> yok**. SYZ veri kümesi (5000 kayıt) bu ortamda bulunmadığı için hiçbir model
> gerçek EKG üzerinde eğitilmedi. Ayrıntı ve gerekçe için `SONUC.md`.
> Kendi makinende koşacağın plan: `PLAN_8SAAT.md`.

## Dosyalar

| dosya | işi |
|---|---|
| `ecg_preprocess.py` | ön işlemenin **tek** kaynağı: filtre, R tepe, 37 özellik, normalizasyon |
| `wfdb_lite.py` | `.hea`/`.dat`/`.mat` okuyucu, saf numpy |
| `prep.py` | ham → 150 Hz `X.npy` + `F.npy` (37 özellik) + `index.csv` |
| `prep_fs.py` | başka hızda X üretir: `python prep_fs.py 250` |
| `model.py` | `r18` `r34` `r18k11` `wide` `w64` `w80` `inception` `hybrid` + özellik dalı |
| `train.py` | 5-fold CV, mixup, EMA, cosine, fold-atlamalı devam, TTA |
| `ensemble.py` | ağırlık araması + stacking + uzman entegrasyonu, **sadece OOF ile** |
| `export.py` | ONNX + int8 + manifest + kendi kendini doğrular |
| `predict.py` (paket içinde) | onnxruntime çıkarım, PyTorch'suz |
| `bench.py` | donanım ölçümü, veri gerekmez |
| `afib_afl_diag.py` | FAZ 2.5 — AFIB/AFL etiket teşhisi |
| `baseline/` | mevcut modelin OOF/test olasılıkları (bkz. `baseline/README.md`) |
| `tools/test_preprocess.py` | ön işleme ve WFDB okuyucu testleri |
| `tools/make_synth.py` | sentetik veri üreteci — **sadece boru hattı doğrulaması için** |

## Kurulum

```bash
pip install numpy scipy pandas scikit-learn
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install onnx onnxruntime
```

Paketi çalıştırmak için sadece `numpy` ve `onnxruntime` yeter.

## Sıra

```bash
# Windows
set ECG_ROOT=D:\TUNA_ISPIR\Documents\Claude\Projects\SYZ
set ECG_WORK=%ECG_ROOT%\ecg_train
cd %ECG_WORK%

# Linux/macOS
export ECG_ROOT=/yol/SYZ
cd "$ECG_ROOT/ecg_train"
```

```bash
python tools/test_preprocess.py          # önce bunun geçtiğini gör
python prep.py                           # hatali=0 ve 3500/750/750 bekle
python bench.py --json bench.json        # bütçeyi ölç
python train.py --preset w64 --tag cap_b64 --only_fold 0 --epochs 40 --patience 99
python train.py --preset w64 --tag main_v2 --epochs 40 --patience 99
python ensemble.py
python export.py
cd package && python predict.py --batch %ECG_ROOT%\test_public.csv --root %ECG_ROOT%
```

## Değişmez kurallar

1. **Sadece CPU.**
2. **Teslim ONNX**, kullanıcıda PyTorch yok.
3. **`test_public` üzerinde hiçbir seçim yapılmaz.** Mimari, hiperparametre,
   eşik, ensemble ağırlığı — hepsi OOF ile seçilir. `test_public` yalnızca
   rapor için, tek sefer okunur. Bu kural koda gömülü: `ensemble.choose_rule`
   test dizilerini hiç görmez.
4. **`test_public` kayıtları eğitime asla girmez** — ne fold'a, ne kalibrasyona,
   ne `--exclude` kararına.
5. **Ön işleme tek kaynaktan:** `ecg_preprocess.py`. Eğitim ve çıkarım aynı
   fonksiyonu çağırır.
6. **`ecg_preprocess.py` değişirse `python prep.py` yeniden koşulur.** Yoksa
   eğitim eski veriyle çalışır ve fark etmezsin.
7. **Her deney `DENEY_KAYDI.md`'ye yazılır:** komut, süre, OOF, test, karar.

## Bilinen bir çelişki

GOREV.md'nin 3. kuralı `test_public` ile seçim yapmayı yasaklıyor, ama FAZ 2'nin
kapısı "fold 0 **test** skoru > 0.855" diyor. İkisi aynı anda tutulamaz.

Bu depodaki kod ikisini de raporlar, kararı **OOF/val skoruyla** vermeni önerir
ve `train.py` çıktısında bunu açıkça yazar. Gerekçe: fold 0 test skoruna bakıp
genişlik seçmek, 750 kayıtlık test kümesine 0.015'lik fold gürültüsüyle uyum
sağlamak demektir; TEKNOFEST jürisine savunulabilir olan OOF'tur.

## Ön işleme sözleşmesi

```
signal    (batch, 12, 1500)  float32   ecg_preprocess.preprocess_signal
features  (batch, 37)        float32   ecg_preprocess.extract_features
```

Derivasyon sırası `I II III aVR aVL aVF V1 V2 V3 V4 V5 V6`. Başlıkta tanınmayan
derivasyon adı varsa dosya sırası kullanılır.

37 özellik **her zaman 500 Hz ham sinyalden** hesaplanır, ağın gördüğü hızdan
bağımsız olarak. Bu sayede `prep_fs.py 250` ile yapılan çözünürlük deneyi tek
değişkenli kalır: `F.npy` aynen kullanılır, sadece `X.npy` değişir.
