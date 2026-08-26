# DENEY KAYDI

Her satır: komut, süre, sonuç, karar. İstisnasız.

> **Bu dosyadaki tüm deneyler SENTETİK veriyle koşuldu.** Gerçek SYZ veri
> kümesi bu ortamda yoktu (gerekçe: `SONUC.md`). Buradaki skorlar boru hattının
> **çalıştığını** kanıtlar, model kalitesi hakkında **hiçbir şey söylemez**.
> Gerçek veriyle koştuğun deneyleri bu dosyanın altındaki boş tabloya ekle.

Ortam: 4 çekirdek Intel Xeon @2.80 GHz, 15 GB RAM, AVX512F var, **AVX512-BF16
ve AMX yok**, GPU yok. torch 2.13.0 (CPU), onnxruntime 1.29.0, Python 3.11.

---

## A — Ön işleme ve okuyucu doğrulaması

| # | komut | süre | sonuç | karar |
|---|---|---|---|---|
| A1 | `python tools/test_preprocess.py` | 40 s | 27/27 PASS | geç |

Ne doğrulandı:

- Elle yazılan Butterworth biquad kaskadları `scipy.signal.butter` ile
  **1e-12 içinde** aynı (alçak geçiren 25/40/60 Hz, yüksek geçiren 0.5/1 Hz).
  Yani `ecg_preprocess.py` scipy'siz ama scipy kadar doğru.
- scipy hızlandırıcısı ile saf numpy referansı **bit düzeyinde aynı**
  (max fark 0.00e+00). Paket scipy'siz çalışır, eğitim scipy ile hızlı çalışır,
  ikisi aynı sayıyı üretir.
- 500 → 150 Hz FFT yeniden örnekleme: 3 Hz sinüsü 3.19e-14 hatayla korur.
- 80 Hz ton 40 Hz alçak geçirenden %2.2 ile geçiyor, 5 Hz ton %100 geçiyor.
- WFDB format 16, format 212, `.mat` (sıkıştırılmış ve sıkıştırılmamış)
  gidiş-dönüş: max hata 1e-7.
- Karışık sırada yazılmış derivasyon başlıkları kanonik sıraya çekiliyor.
- Bozuk girdi (hep sıfır, sabit, tek diken, 1e-9 genlik) NaN üretmiyor,
  istisna atmıyor.
- 72 bpm sentetik ritimde R tepe sayısı tam doğru, konum hatası 0 örnek.

## B — Sentetik veri kümesi

| # | komut | süre | sonuç | karar |
|---|---|---|---|---|
| B1 | `python tools/make_synth.py --out SYNTH --per-class 200 --difficulty easy` | 1.5 dk | 1000 kayıt | — |
| B2 | `python prep.py --root SYNTH --out cache_synth --workers 4` | 5.3 s | **hatali=0**, 700/150/150 | geç |
| B3 | 37 özellik + lojistik regresyon | 2 s | test macro-F1 **1.0000** | **çok kolay** |
| B4 | `--difficulty hard` ile yeniden üret | 1.5 dk | 1000 kayıt | — |
| B5 | B2 tekrar | 5.3 s | hatali=0 | geç |
| B6 | B3 tekrar | 2 s | test macro-F1 **0.9867** | hâlâ kolay, kabul |

**B3/B6 kararı:** sentetik veri gerçek EKG'den çok daha ayrılabilir. `hard`
modunda AFIB/AFL örtüşmesi (değişken bloklu flutter, kaba AFIB, derivasyon
düşmesi, artefakt) eklendi ama ikili doğruluk yine 0.97'de kaldı — gerçek
veride bu sayı 0.76. Daha fazla yapay zorlaştırma **tiyatro** olurdu: sahte
veriyi zorlaştırmak model hakkında bilgi üretmez. Sentetik küme boru hattı
testi olarak bırakıldı, zorluk göstergesi olarak **kullanılmadı**.

**B3 yan bulgusu (işe yarar):** AFIB/AFL ayrımında en ayırt edici 8 özellik
`rr_pnn50`, `rr_pnn20`, `rr_irregular_frac`, `flutter_concentration`,
`rr_sd2`, `rr_min`, `flutter_autocorr`, `rr_std` çıktı — yani RR düzensizliği
+ flutter bandı dar-bantlılığı. Bu, özelliklerin fizyolojik olarak doğru şeye
baktığını gösteriyor; özellik tasarımı için olumlu sinyal.

## C — Eğitim

| # | komut | süre | OOF/val | test | karar |
|---|---|---|---|---|---|
| C1 | `train.py --preset r18 --tag smoke --only_fold 0 --epochs 2` | 0.2 dk | **0.0667** | 0.0667 | **HATA** |
| C2 | C1 tekrar (EMA düzeltmesi sonrası, 6 epoch) | 0.7 dk | 1.0000 | 1.0000 | geç |
| C3 | C2 aynı komut tekrar | 3 s | fold atlandı | — | devam çalışıyor |
| C4 | `train.py --preset r18 --tag synth_r18 --epochs 8` (5-fold) | 4.4 dk | **0.9977** | 0.9867 | geç |
| C5 | `train.py --preset inception --tag synth_inc --epochs 8 --lr 0.002` | 4.8 dk | **0.9977** | 0.9933 | geç |

**C1 — bulunan gerçek hata.** val macro-F1 = 0.0667 tek sınıf tahmini demek.
Kayıp düşüyordu (1.349 → 0.852), yani model öğreniyordu ama **EMA ağırlıkları
hâlâ rastgele başlangıçtı**: decay 0.999 ile 54 adım sonra
`0.999^54 = 0.947`, yani ortalama %95 başlangıç ağırlığı. Düzeltme: standart
EMA ısınması, `d = min(decay, (1+n)/(10+n))`. C2'de 0.0667 → 1.0000.

Bu hata 40 epoch'luk gerçek koşuda kendini göstermezdi (4280 adım sonra
`0.999^4280 ≈ 0.014`), ama **kısa taramalarda sessizce yanlış sonuç**
verirdi — FAZ 2'nin tek-fold kapasite taraması tam olarak böyle bir koşu.

**C4/C5 notu:** iki modelin OOF'u aynı (0.9977) ve **tahminleri %100 örtüşüyor**
(`ensemble.py` çeşitlilik ölçümü: 1.0000). Sentetik problem çok kolay olduğu
için her mimari aynı çözüme yakınsıyor. Gerçek veride FAZ 5'in aradığı sinyal
bu ölçümün 0.85'in **altına** düşmesi.

## D — Ensemble

| # | komut | süre | OOF | test | karar |
|---|---|---|---|---|---|
| D1 | `python ensemble.py --cache cache_synth` | — | — | — | **HATA** |
| D2 | D1 tekrar (sklearn düzeltmesi sonrası) | 25 s | **0.9977** | 0.9933 | geç |

**D1 hatası:** `LogisticRegression(multi_class=...)` scikit-learn 1.9'da
kaldırılmış. Argüman silindi (multinomial zaten varsayılan).

**D2 sonucu:** kural karşılaştırması `flat 0.9977 | weighted 0.9977 |
stacked 0.9965` → **flat seçildi.** Seçim mantığı doğru çalıştı: `weighted`
`flat`'i 0.002'den az geçtiği için basit olan tercih edildi, `stacked` zaten
daha kötüydü. Ensemble tek üyeye göre **+0.0000** kazandırdı — beklenen, çünkü
iki üye aynı tahminleri veriyor.

**D3 — sonradan bulunan hata:** `ensemble_oof_prob.npy` sadece geliştirme
satırlarını (850) yazıyordu, oysa `baseline/README.md`'de tanımlı üye
sözleşmesi tam cache uzunluğu (1000) istiyor. `afib_afl_diag.py` bu yüzden
reddetti. Kaynakta düzeltildi: dosya artık tam uzunlukta, geliştirme dışı
satırlar sıfır.

## E — ONNX dışa aktarım

| # | komut | sonuç | karar |
|---|---|---|---|
| E1 | `python export.py --cache cache_synth` | onnxscript eksik | bağımlılık kuruldu |
| E2 | E1 tekrar | ONNX=PyTorch (1.8e-07), ama int8 **başarısız** | **iki hata** |
| E3 | E2 (TorchScript dışa aktarıcıya geçiş) | int8 çalışıyor, ama reddedildi | tolerans hatalı |
| E4 | E3 (int8 kararı skora bağlandı) | **int8 kabul, 3.7x küçük** | geç |

**E2 — hata 1: harici ağırlık dosyaları.** Yeni dynamo dışa aktarıcısı
ağırlıkları `*.onnx.data` yan dosyalarına yazdı. Grafik dosyaları 0.2 MB
görünüyordu, gerçek paket 66 MB'tı. Manifest boyutları ve **sha256'ları
sadece grafiği kapsıyordu**; `models/*.onnx` kopyalayan biri **bozuk paket**
alırdı. Düzeltme: `dynamo=False` (TorchScript dışa aktarıcı) + yan dosya
oluşursa dışa aktarımı durduran açık kontrol.

**E2 — hata 2: int8 çalışmıyordu.** dynamo grafiğinde
`quantize_dynamic` şekil çıkarımı hatası veriyordu
(`Inferred shape ... (37) vs (128)`). Aynı düzeltme bunu da çözdü: TorchScript
grafiği sorunsuz kuantalanıyor.

**E3 — hata 3: yanlış kabul ölçütü.** int8'i grafik başına olasılık farkıyla
(2e-3) yargılıyordum; int8 sapmaları 5.6e-3 – 1.8e-2 çıktı ve hepsi
reddedildi. Ama **kuantalama olasılıkları tasarımı gereği ~%1 oynatır**;
GOREV'in ölçütü zaten olasılık değil **skor** farkı (0.005). Karar mekanizması
düzeltildi: fp32 ve int8 paketlerinin **ensemble macro-F1'i** karşılaştırılıyor,
fark eşiğin altındaysa int8 teslim ediliyor.

**E4 sonucu:** float32 ensemble 0.9933, int8 ensemble 0.9933, fark **0.0000**
→ int8 kabul. Paket 66.9 MB → **18.1 MB**.

## F — Paket doğrulaması (PyTorch'suz)

| # | komut | sonuç | karar |
|---|---|---|---|
| F1 | paket `/tmp/deliver`'a kopyalandı, `PYTHONPATH` ile `import torch` **engellendi** | — | — |
| F2 | `predict.py --batch test_public.csv --root SYNTH` | macro-F1 **0.9933** | geç |
| F3 | manifest karşılaştırması | fark **0.0000** | geç |
| F4 | `predict.py <tek kayit>.hea` | AFL %73.9 | geç |
| F5 | yan dosya (`*.data`) sayımı | **0** | geç |

`torch/__init__.py` içine `raise ImportError` koyan sahte bir paket
`PYTHONPATH`'in başına konuldu, yani `import torch` **kesin olarak**
başarısız. Paket bu koşulda tam çalıştı → teslim koşulu sağlandı.

Hız: 150 kayıt, 10 ONNX grafiği → ön işleme 11.4 s, çıkarım 29.4 s,
**271 ms/kayıt**. Gerçek 750 kayıtlık test için ~3.4 dk.

## G — FAZ 2.5 teşhis aracı

| # | komut | sonuç | karar |
|---|---|---|---|
| G1 | `afib_afl_diag.py --oof ensemble_oof_prob.npy --cache cache_synth` | ikili doğruluk 1.0000, **0 şüpheli** | araç çalışıyor |

Sentetik veride etiketler tanım gereği hatasız, dolayısıyla şüpheli kayıt
çıkmaması **doğru davranış**. Tavan taraması çalıştı ve GOREV'in iddiasını
sayısallaştırdı (aşağıda).

## H — Donanım ölçümü

| # | komut | sonuç |
|---|---|---|
| H1 | `python bench.py --dev 4250 --epochs 40 --json bench.json` | aşağıdaki tablo |

Bu makinede (4 çekirdek, AVX512F, **BF16 yok**), 4250 geliştirme kaydı,
batch 32, 150 Hz giriş (1500 örnek), fold başına 107 adım/epoch:

| preset | param | s/adım | s/epoch | 5-fold × 40 epoch |
|---|---|---|---|---|
| r18 | 2.32 M | 0.210 | 22.5 | **1.25 saat** |
| r18k11 | 3.55 M | 0.237 | 25.4 | 1.41 saat |
| inception | 1.00 M | 0.261 | 27.9 | 1.55 saat |
| hybrid | 1.25 M | 0.318 | 34.0 | 1.89 saat |
| wide (b48) | 5.14 M | 0.386 | 41.3 | 2.29 saat |
| r34 | 4.33 M | 0.400 | 42.8 | 2.38 saat |
| **w64** | 9.07 M | 0.585 | 62.6 | **3.48 saat** |
| **w80** | 14.11 M | 0.871 | 93.2 | **5.18 saat** |

GOREV'deki referans (bulut, 2 çekirdek, AMX + AVX512-BF16) r18 için 14 s/epoch
diyordu; bu makine 22.5 s/epoch, yani **1.6x yavaş**. GOREV'in "2-3 kat yavaş
olacak" tahmini doğru mertebede.

**bf16 kararı:** bu CPU'da AVX512-BF16 ve AMX **yok**. `bench.py` bunu
saptayıp uyarıyor, `train.py` de bf16'yı kendiliğinden kapatıp
`bf16 kapali: CPU'da AVX512-BF16/AMX yok` yazıyor. `--no_bf16` ile koş.

---

## Bulunan ve düzeltilen hatalar — özet

| # | hata | nasıl yakalandı | etkisi düzeltilmeseydi |
|---|---|---|---|
| 1 | EMA ısınması yok | 2 epoch'luk duman testi | tek-fold taramaları sessizce çöp üretirdi |
| 2 | ONNX ağırlıkları yan dosyada | dosya boyutu tutarsızlığı | `models/*.onnx` kopyalayan bozuk paket alırdı |
| 3 | int8 dynamo grafiğinde çalışmıyor | export çıktısı | teslim int8 olamazdı (GOREV şartı) |
| 4 | int8 kabul ölçütü olasılık farkıydı | int8 hep reddediliyordu | paket 3.7x gereksiz büyük kalırdı |
| 5 | `ensemble_oof_prob.npy` yanlış uzunlukta | teşhis aracı reddetti | FAZ 2.5 hiç koşamazdı |
| 6 | `LogisticRegression(multi_class=)` kaldırılmış | ensemble çöktü | stacking hiç koşamazdı |

---

## Gerçek veriyle koşulacak deneyler — DOLDUR

Aşağıdaki tabloyu kendi makinende doldur. Sütunlar GOREV'in istediği gibi:
değişen tek şey, fold 0 OOF, 5-fold OOF, 5-fold test, karar.

| # | komut | değişen tek şey | süre | fold0 OOF | 5f OOF | 5f test | karar |
|---|---|---|---|---|---|---|---|
| R0 | `python prep.py` | — | | hatali=? | | | |
| R1 | `python train.py --preset wide --tag base_b48 --only_fold 0 --epochs 40 --patience 99` | referans nokta | | | | | |
| R2 | `python train.py --preset w64 --tag cap_b64 --only_fold 0 --epochs 40 --patience 99` | base 48→64 | | | | | |
| R3 | `python train.py --preset w64 --tag cap_b64_d3 --only_fold 0 --epochs 40 --patience 99 --dropout 0.3` | dropout 0.2→0.3 | | | | | |
| R4 | `python train.py --preset w80 --tag cap_b80 --only_fold 0 --epochs 40 --patience 99 --dropout 0.3` | base 64→80 | | | | | |
| R5 | `python prep_fs.py 250` + `--cache cache_250` | 150→250 Hz | | | | | |
| R6 | `python train.py --preset <kazanan> --tag main_v2 --epochs 40 --patience 99` | tam 5-fold | | | | | |
| R7 | `python ensemble.py` | baseline ile birleştir | | | | | |
| R8 | `python afib_afl_diag.py --oof ensemble_oof_prob.npy` | teşhis | | | | | |
| R9 | `python train.py --preset inception --tag div_inc --only_fold 0 --epochs 30 --lr 0.002` | mimari | | | | | |
| R10 | `python train.py --preset hybrid --tag div_hyb --only_fold 0 --epochs 30 --lr 0.001` | mimari | | | | | |
| R11 | `python export.py` | paketle | | | | | |

**Kural:** her satırda **tek bir şey** değişsin. Fold'lar arası oynaklık
±0.015 olduğu için, karar vermek için **≥0.01 fark** iste; altındakine gürültü
de ve geç.
