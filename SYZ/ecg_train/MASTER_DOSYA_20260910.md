# TEKNOFEST 2026 ECG — Ana Belge

**Takım:** tkt-26 · Özel İzmir Bahçeşehir 50. Yıl Fen ve Teknoloji Lisesi
**Kategori:** Sağlıkta Yapay Zekâ, **Lise Seviyesi**
**Güncelleme:** 10 Eylül 2026 · **Freeze:** 15 Eylül · **Yarışma:** 16–18 Eylül, Dicle Üniversitesi

> **v2 — 10 Eylül akşamı düzeltildi.** İlk sürümde üç hatam vardı ve bir
> hipotezim deneyle çürütüldü; hepsi aşağıda açıkça işaretli.
>
> **Kaynak hiyerarşisi:** sayısal bir çelişkide sıra şudur —
> (1) kodun kendisi, (2) `runs/*/summary.json`, (3) `DENEY_KAYDI.md`,
> (4) bu belge. Ezberden yazılmış hiçbir sayı sunuma girmemeli.

---

# BÖLÜM 0 — v1'DEKİ HATALARIM

Bu belgenin ilk sürümünde şunlar yanlıştı. Kayda geçiyorlar çünkü sunumun
omurgası "her iddiayı ölçtük" — kendi iddialarım dahil.

## 0.1 Veri bölünmesini yanlış yazdım

| v1'de yazdığım | gerçek |
|---|---|
| 5000 üzerinde 5-fold | **3500 train / 750 val / 750 test_public** |
| fold-0 val = 1000 kayıt | **dev = 4250, 5-fold → fold val = 850** |

**Sonucu:** fold-0 gürültü tabanı hesabımı n=1000 ile yaptım. n=850 ile
SE ≈ 0.0117 (0.0108 değil), yani gözlenen −0.021 ≈ **1.8 SE** (1.98 değil).
Sonuç değişmiyor (sınırda) ama girdi yanlıştı.

## 0.2 "Sınıf öncülü kayması ana neden" hipotezim ÇÜRÜTÜLDÜ

Dış veri fold-0 düşüşünün birinci nedeni olarak sınıf öncülü kaymasını
göstermiştim. **Test edildi ve yanlış çıktı:**

```
class-balanced sampler ile epoch exposure = [800, 800, 800, 800, 800]
val(tta) = 0.8164   vs   baseline 0.8335   →   -0.0171
```

Öncül düzeltildi, düşüş **devam etti**. Yani sorun sınıf dengesizliği değil;
domain/etiket uyumsuzluğu daha güçlü aday.

Aynı şekilde `ep30 → ep11`'i bu hipotezin "parmak izi" diye sunmuştum — o
yorum da dayanaksız kaldı.

## 0.3 Önerdiğim üç deneyin ikisi zaten yapılmıştı

| önerim | durum |
|---|---|
| E0 sınıf bazlı F1 kırılımı | zaten yapılmış (AFL300 tablosu mevcut) |
| E2 öncül korunarak karıştırma | zaten yapılmış, hipotezimi çürüttü |
| "+300/+300 yapmayın, ayrıştırmaz" | yapılmış: −0.0075, bilgi verdi |
| "pretrain→finetune'a girmeyin" | yapılmış: −0.0173, net cevap alındı |

Son ikisinde fazla temkinliydim. Deneyler ucuzdu ve **kesin negatif cevap**
verdiler — bu, belirsizlikte kalmaktan iyi.

## 0.4 Ön işleme sabitlerinde çelişki — ÇÖZÜLMELİ

`TEKNOFEST_ECG_TUM_BILINENLER` belgesi ile depodaki kod uyuşmuyor:

| | o belge | depodaki `ecg_preprocess.py` |
|---|---|---|
| low-pass | 47 Hz | **40 Hz** (`LP_CUTOFF = 40.0`) |
| notch | yalnız 50 Hz | **50 ve 60 Hz** (`NOTCH_FREQS = (50.0, 60.0)`) |
| normalizasyon | derivasyon bazında z-score | **medyan-merkezli, 1.4826×MAD, global** |

O belgedeki dil ("yaklaşık ayar", "z-score benzeri") ezberden yazıldığını
düşündürüyor. **Sunumdan önce koddan doğrulayın:**

```bat
python -c "import ecg_preprocess as e; print(e.preprocess_config())"
```

Jüri kodu tekrar koşturma yetkisine sahip (kılavuz md. 2.6). Slaytta 47 Hz
yazıp kodda 40 Hz çıkması gereksiz bir güven kaybı olur.

---

# BÖLÜM 1 — PROBLEM VE SONUÇ

## 1.1 Görev

12 derivasyonlu EKG kaydını 5 sınıfa ayırmak:

```
NORMAL · AFIB (atriyal fibrilasyon) · AFL (atriyal flutter)
LBBB (sol dal bloğu) · RBBB (sağ dal bloğu)
```

- Metrik: **macro-F1** (eşitlikte PR-AUC)
- Kayıtlar: 500 Hz, 10 s, WFDB (`.hea` + `.mat`/`.dat`)
- **CPU-only**, PyTorch'suz teslim, internet yasak
- Geliştirme verisi: 5.000 kayıt (05.05.2026'da paylaşıldı) + 750 `test_public`

## 1.2 Nihai sonuç

| ölçüm | değer |
|---|---|
| **test_public macro-F1** | **0.841204** |
| test_public accuracy | 0.845333 |
| Dev OOF (5-fold) | 0.843755 |
| Kayıt | 750/750, **0 hata** |
| Ensemble | **20 ONNX** (101.5 MB) |
| Inference | **6.21 dk / 750 kayıt** (MP11) |

Sınıf bazlı F1:

| sınıf | F1 |
|---|---|
| NORMAL | 0.941558 |
| AFIB | 0.746177 |
| **AFL** | **0.666667** ← darboğaz |
| LBBB | 0.960526 |
| RBBB | 0.891089 |

**Güven aralığı:** SE = 0.0129, %95 CI = [0.8140, 0.8644] (20.000 bootstrap).

### Aile bazlı sonuçlar

| aile | OOF | test | ağırlık | fold |
|---|---|---|---|---|
| `cv10` | 0.840129 | 0.841608 | **0.671958** | 10 |
| `main_v2` | 0.836945 | 0.840913 | 0.090122 | 5 |
| `seed99` | 0.828813 | — | 0.237921 | 5 |
| `full_v1` | — (OOF yok) | 0.846927 | — | — |

`seed99` tek başına en zayıf ama **ensemble çeşitliliğine katkısı** için tutuldu.
`full_v1` OOF'suz olduğu için **model seçim kanıtı değildir**, yalnız rapor.

### Veri bölünmesi

```
5000 kayıt = 3500 train + 750 validation + 750 test_public
development = train + validation = 4250
5-fold → her fold ≈ 3400 train / 850 validation
test_public: sınıf başına 150, toplam 750
```

### ONNX doğrulaması

```
ONNX     macro-F1 = 0.841204
PyTorch  macro-F1 = 0.839928
fark              = +0.001276      (kabul eşiği < 0.005)  → PASS
```

PyTorch'suz `.venv_no_torch` ortamında: 750/750, 0 hata, manifest farkı 0.000000.

---

# BÖLÜM 2 — MİMARİ

## 2.1 Boru hattı

```
WFDB okuma (wfdb_lite.py, saf numpy)
   ↓
Ön işleme (ecg_preprocess.py — TEK KAYNAK)
   0.5 Hz yüksek geçiren → 50/60 Hz çentik → 40 Hz alçak geçiren
   → 150 Hz'e FFT yeniden örnekleme → global normalizasyon (±8σ kırpma)
   ↓
İki kol:
   (a) sinyal  (12, 1500)
   (b) 37 özellik  (RR, atriyal bant, QRS morfolojisi)
   ↓
ResNet1D + Squeeze-Excitation  ⊕  37-özellik MLP
   ↓
20 model ensemble (ağırlıklı) → 5 olasılık
```

### Ön işleme sabitleri (`ecg_preprocess.py`)

```python
NATIVE_FS = 500.0        TARGET_FS = 150.0       TARGET_SECONDS = 10.0
HP_CUTOFF = 0.5  (order 2)                       LP_CUTOFF = 40.0 (order 4)
NOTCH_FREQS = (50.0, 60.0)  Q = 30.0
NORM_MODE = "global"     CLIP_SIGMA = 8.0        TARGET_LEN = 1500
```

**Kritik detay — filtre arka ucu:** `_sosfilt` scipy varsa onu kullanır, yoksa
saf-Python örnek-örnek döngüsüne (`_sosfilt_reference`) düşer. Ölçüldü:

| arka uç | `preprocess_record` |
|---|---|
| scipy | **16.8 ms** |
| saf numpy | **978.5 ms** |
| | **58×** |

**Çıktı bit-birebir aynı** (20 kayıtta fark `0.000e+00`). scipy saf hızlandırıcıdır.
`ECG_FORCE_NUMPY_FILTER=1` ile referans yola sabitlenir.

## 2.2 Ensemble bileşimi

| aile | fold | not |
|---|---|---|
| `cv10` | 10 | |
| `main_v2` | 5 | ana koşu |
| `seed99` | 5 | farklı seed |
| **toplam** | **20** | |

OOF-only budama: 5 aile 0.842924 → 3 aile/20 model **0.843755** (+0.000831).
Budama `test_public`'e bakılmadan yapıldı.

## 2.3 Dışa aktarım

- ONNX **opset 17**, TorchScript exporter (`dynamo=False`)
- **int8 dinamik kuantizasyon**
- `export.py` PyTorch↔ONNX eşleşmesini kendisi doğrular
  (`--tol-prob 2e-3`, `--tol-prob-int8 0.10`)
- Modeller dinamik batch destekliyor: `['batch', 12, 1500]`, `['batch', 37]`

---

# BÖLÜM 3 — DARBOĞAZ: AFIB/AFL

## 3.1 Dört bağımsız yöntem, aynı duvar

| yöntem | AFIB/AFL çift doğruluğu |
|---|---|
| Kaggle 8.8 M parametreli GPU modeli | 0.701 |
| Bizim CNN ensemble | 0.760 |
| 37 özellik + GBM | 0.725 |
| **Yalnız o çifte eğitilmiş uzman model** | 0.744 |

**8.8 milyon parametreli model, küçüğünden daha kötü.** Farklı kapasite, mimari
ve özellik kümesi — hepsi ~0.70–0.76. Bu, veri kıtlığının değil **sinyal/etiket
sınırının** imzasıdır.

## 3.2 Hataların anatomisi

- Çift doğruluğu: **0.7376** (dev) / **0.751** (test_public)
- AFL recall: **0.573**
- **Çift hatalarının %93.4'ü düşük güvenli** — model kendinden emin biçimde
  yanılmıyor, gerçekten kararsız

## 3.3 Neden zor — fizik

AFL'nin ayırt edici işareti atriyal testere dişi dalgası, **2.5–12 Hz** bandında.
Ama 75 bpm'de QRS harmonikleri 5.0 ve 6.25 Hz'e düşüyor — **tam o bandın içine**.
QRST iptali denendi (aşağıda), gerçek veride kazanç +0.0006.

---

# BÖLÜM 4 — ELENEN FİKİRLER (22 adet)

Her biri ölçülerek elendi. Sunumun en güçlü kısmı budur.

| # | fikir | nasıl elendi |
|---|---|---|
| 1 | Kapasite artırma (64/80 kanal) | OOF iyileşmedi |
| 2 | 250 Hz örnekleme | iyileşme yok |
| 3 | Etiket temizleme | iyileşme yok |
| 4 | Uzman (2 sınıflı) model | 0.744, ensemble'dan kötü |
| 5 | Eşik kalibrasyonu | iyileşme yok |
| 6 | Ölçek TTA | iyileşme yok |
| 7 | Daha çok fold | iyileşme yok |
| 8 | **Balanced decoding** (transport dual) | OOF 0.843755 → 0.834084, **−0.0097** |
| 9 | **mixup çift koruması** | McNemar **p = 0.7552**, gürültü |
| 10 | **QRST artık özellikleri** | sentetikte umutlu, gerçek veride **+0.0006** |
| 11 | **Inception1D** | aynı tahmin oranı 0.9235, çeşitlilik kapısını geçemedi |
| 12 | **CNN+Transformer hibrit** | aynı tahmin oranı 0.9259, aynı |
| 13 | **Batch inference** | 4.7× hızlı ama olasılıklar **0.006** saptı |
| 14 | **2×MP5 hibrit paralellik** | %1 kazanç, 2 kat operasyon riski |
| 15 | Dış veri +900 AFIB +900 AFL | fold-0 val **−0.0207** |
| 16 | Dış veri +300 AFIB +300 AFL | **−0.0075** |
| 17 | Dış veri sınıf-dengeli 300+300 | **−0.0171** ← öncül hipotezini çürüttü |
| 18 | Dış veri yalnız AFL +300 | −0.0019, bootstrap P(Δ>0) = **0.40** |
| 19 | Dış veri pretrain → fine-tune | **−0.0173**, AFL 0.6216 → 0.5784 |
| 20 | Sıfırdan AFIB/AFL uzman | çift F1 0.7070, birleşik **−0.0103** |
| 21 | Warm-start uzman | inner 0.8492 → **outer 0.6931**, birleşik −0.0071 |
| 22 | **Waveform-SVM gate** | fold 0/1 **+0.008/+0.011**, fold 2/3/4 −0.004/−0.006/**−0.024** |

## 4.0 Türetilmiş özellikler ve basit sınıflandırıcılar (ayrıca elendi)

| yöntem | çift F1 | karar |
|---|---|---|
| Logistic (37 özellik) | 0.7026 | yetersiz |
| RBF-SVM (37 özellik) | 0.7123 | yetersiz |
| RBF-SVM (atriyal 18) | 0.6699 | yetersiz |
| RBF-SVM (ritim 15) | 0.6963 | yetersiz |
| Türetilmiş 17 özellik | 0.5952 | reddet |
| 37 + türetilmiş 17 | 0.6867 (inner 0.7706) | reddet |
| **Wave-only (~90 ölçüm)** | **0.7262** | küçük sinyal |
| **37 + Wave** | **0.7498** (+0.0375) | umut verdi → 5-fold'da elendi |

## 4.1 Metodolojik notlar

**Balanced decoding neden reddedildi:** OOF −0.0097 ve AFIB/AFL çifti
0.6988 → 0.6888. İki yönde de kötüleşti.

**mixup çift koruması:** hipotez "mixup AFIB+AFL karışımları üretip sınırı
bulanıklaştırıyor" idi. `hb_nopair` ve `hb_nomix` **ikisi de 0.8000** çıktı,
McNemar p = 0.7552. Hipotez yanlıştı, yama gönderilmedi.

**QRST artığı:** sentetik Bayes-tavanı kıyas kümesinde umut vericiydi. Gerçek
veride +0.0006. Bir ara hata da yakalandı: `test_resid.py` ilk 200 kaydı sırayla
alıyordu, `index.csv` sınıfa göre gruplu olduğu için hepsi AFIB çıkıyor ve tüm
AUC'ler tam 0.500 oluyordu. Sınıf başına örnekleme ile düzeltildi.

**Çeşitlilik kapısı:** "farklı mimari → farklı hatalar → ensemble kazancı"
varsayımı ölçüldü. Anlaşma oranı yalnızca bir vekil; asıl ölçüt doğrudan harman
kazancı. Inception ve Hibrit ikisi de ~0.92 anlaşmayla kapıyı geçemedi.

---

# BÖLÜM 5 — HIZLANDIRMA (14.8×)

## 5.1 Başlangıç ve bitiş

| aşama | 750 kayıt | ms/kayıt |
|---|---|---|
| İlk seri koşu | 5518.1 sn (92 dk) | 7357 |
| **MP11 (final)** | **372.6 sn (6.21 dk)** | **497** |

**Çıktı bit-birebir aynı:** `class_mismatch = 0`, `max_abs_diff = 0.0`.

## 5.2 Darboğaz profillemesi

```
WFDB okuma      = 0.001 sn
E.prepare       = 0.012 sn   (scipy kurulu; scipy YOKKEN ~1.0 sn)
özellik ölçekleme = 0.000 sn
20 ONNX çıkarım = 6.535 sn   ← sürenin neredeyse tamamı
```

## 5.3 Çürütülen hipotez

"Oturumlar her kayıtta yeniden yükleniyor" — **yanlış.**

```
sessions() 1. çağrı = 3.707 sn
sessions() 2. çağrı = 0.000 sn
```

`_SESSIONS` önbelleği zaten çalışıyordu. `--cache-sessions` bu pakette kazanç
getirmiyor; yalnızca önbelleklemeyen bir pakete karşı sigorta olarak duruyor.

## 5.4 Gerçek sebep: ORT thread aşırı aboneliği

20-model ensemble, tek kayıt:

| intra-op thread | süre |
|---|---|
| 1 | 1.708 sn |
| **2** | **1.286 sn** ← en iyi |
| 4 | 2.119 sn |
| 8 | 5.448 sn |
| 16 | 11.686 sn |

Varsayılan `intra_op_num_threads = 0` (tüm çekirdekler) küçük 1B modellerde
**zarar veriyor** — iş parçacıklarını senkronize etmek hesaptan pahalıya geliyor.

> **Tuzak:** ORT'nin kendi thread havuzu var; `OMP_NUM_THREADS`'i **dinlemez**.
> Doğrulandı: onnxruntime 1.29'da `OMP_NUM_THREADS=2` sonrası
> `intra_op_num_threads` hâlâ 0.

## 5.5 Hız merdiveni

| yol | 750 kayıt | ms/kayıt | süreç |
|---|---|---|---|
| 2×MP5 | 368.7 sn | 491.6 | 2 (reddedildi) |
| **MP11** | **372.6 sn** | **497** | **1 ← ANA YOL** |
| MP10 | 374.1 sn | 499 | 1 |
| MP8 | 398.7 sn | 532 | 1 |
| 2×MP4 | 432.6 sn | 576.8 | 2 |
| threads=2 | 1252.8 sn | 1670 | 1 (güvenli yedek) |
| ilk seri | 5518.1 sn | 7357 | eski |

**2×MP5 neden reddedildi:** MP11'den 3.9 sn (%1) hızlı, ama iki süreç, 40 oturum,
iki JSON, birleştirme adımı. %1 için operasyon riskini iki katına çıkarmak.

## 5.6 Batch inference neden reddedildi

| batch | ms/kayıt | class mismatch | max olasılık farkı |
|---|---|---|---|
| 1 | 6537.9 | 0 | 1.34e-6 |
| 5 | 2131.1 | 0 | 0.00147 |
| 10 | 1428.5 | 0 | 0.00676 |
| 20 | 1109.4 | 0 | 0.00620 |

Sınıflar aynı kaldı ama olasılıklar doğal gürültünün (1.34e-6) **binlerce katı**
saptı. Kılavuz md. 4 eşitliği PR-AUC ile bozuyor → olasılıklar önemli. Reddedildi.

## 5.7 Model-içi paralellik — tasarım kararı

`--model-parallel N` 20 ONNX'i `ThreadPoolExecutor` ile paralel koşturur.

**Kritik:** ensemble toplamı `predict.py` **içinde** ve o dosya değişmeyecek.
Toplamı dışarıda yeniden yazmak, görülmeyen bir mantığı (ağırlıklar, ölçekleme,
normalizasyon) taklit etmek ve doğrulanmış 0.841204'ten sessizce sapmak olurdu.

Bunun yerine **oturumlar sarmalandı:**

```
_SessionProxy.run()  →  _ParallelRunner
    ilk çağrı : 20 modelin HEPSİNİ havuza gönderir,
                sonuçları İNDEKSE göre saklar
    sonraki 19: önbellekten döner
    girdi değişince (blake2b anahtarı) yeni tur
```

`predict.py` yine kendi döngüsünü **manifest sırasında** koşar ve toplamı kendi
yapar → float toplama sırası birebir korunur → **çıktı bit-birebir aynı.**

---

# BÖLÜM 6 — TESLİM MEKANİĞİ

## 6.1 Kılavuzdan çıkarılan zorunlu kurallar (md. 3 / 3.1)

- UTF-8, `.json`, önerilen ad `TEAM_<TAKIM_ID>_FINAL.json`
- Üst alanlar: `team_name`, `team_id`, `application_id`, `competition_level`
- Lise için `competition_level` tam olarak **`"LISE"`**
- Sınıf adları **BÜYÜK HARF**: NORMAL, AFIB, AFL, LBBB, RBBB
- Her test örneği `predictions` içinde **tam olarak bir kez**
- Hiçbir id eksik/fazla/değiştirilmiş olmayacak
- `probabilities` zorunlu, tüm sınıflar için değer
- Olasılıklar 0–1, toplamı **1 ±0.001**, ondalık ayırıcı **nokta**
- NaN, Infinity, boş olasılık, yorum satırı, tekrar eden id **kabul edilmez**

## 6.2 KYS bilgileri

```
team_name      : tkt-26
team_id        : 807466
application_id : 4997133
```

`team_id` düz sayı (`TEAM_807466` değil) — kılavuz "KYS'deki bilgilerle aynı"
diyor ve eşleştirme `team_id` üzerinden. Dosya adı otomatik `TEAM_807466_FINAL.json`.

## 6.3 `make_submission.py` — SHA-256 `7fbdec41…`

İki predict API'sini de destekler (`hasattr` ile otomatik seçim):

- **`Bundle`** (klasik) — `Bundle()` + `load_one()` + `predict_proba()`
- **`predict_record`** (final paket) — `MANIFEST`, `CLASSES`, `sessions()`,
  `predict_arrays()`, `predict_record(path) -> (idx, prob)`

### Bayraklar

| bayrak | ne yapar |
|---|---|
| `--threads N` | 20 oturumu `intra_op=N, inter_op=1` ile önceden kurar |
| `--model-parallel N` | kayıt içi 20 model paralelliği (ThreadPoolExecutor) |
| `--workers N` | kayıtlar arası süreç paralelliği (spawn) |
| `--cache-sessions` | `sessions()` memoize (bu pakette gereksiz) |
| `--retag <json>` | var olan JSON'un yalnız kimlik alanlarını değiştirir |
| `--validate <json>` | var olan dosyayı denetler |
| `--max-fail N` | eşik aşılırsa **JSON hiç yazılmaz** |
| `--ids <dosya>` | id listesi ve sırası |

`--model-parallel` ile `--workers` **birlikte reddedilir** (çekirdekleri N×M aşırı
abone eder).

### Korumalar

- Yol biçimi (`.hea` mi, uzantısız mı) **ilk kayıtta keşfedilir**, varsayılmaz
- İlk kayıt işlenemezse hemen durur, kalan 749'u denemez
- Her `prob`: 5 eleman, finite, negatif yok, toplam > 0, 1'e normalize
- Düşen kayda eşit olasılık — **id asla eksik kalmaz**
- MANIFEST'te model listesi tanınmazsa **sert durur**
- Yazdıktan sonra **diskten geri okuyup tekrar doğrular**
- Yuvarlama kayması en yüksek sınıfa yedirilir
- >%90 tek sınıf → güçlü uyarı
- `_SESSIONS`/model-parallel yapısı uymazsa **geri alınır**, koşu devam eder

### Sessiz çöp teslim koruması

Bir ara sürümde, özellik uzunluğu uyuşmazlığıyla **50 kaydın hepsi düşmüş**,
hepsine eşit olasılık yazılmış, hepsi NORMAL tahmin edilmiş ve betik
`all checks passed` demişti. Üç koruma eklendi: (a) Bundle iç alanlarına
bağımlılık yok, (b) ilk kayıt probu gerçek şekilleri öğrenir, (c) `--max-fail`
reddi + tek sınıf uyarısı. Düzeltmeden sonra bozuk boru hattı **hiç dosya yazmıyor**.

## 6.4 Yarışma günü komutu

```bat
python make_submission.py --root <TEST_KLASORU> ^
    --team-name "tkt-26" --team-id 807466 --application-id 4997133 ^
    --threads 1 --model-parallel 11
```

`--ids` **kullanılmayacak** — gerçek test verisinde `--root` taranacak.

Doğrulama:

```bat
python make_submission.py --validate TEAM_807466_FINAL.json --root <TEST_KLASORU>
```

Son satır tam olarak `all checks passed` olmalı.

## 6.5 Yapılan teslim (8 Eylül 21:00)

Organizasyona örnek veri seti üzerinden üretilen JSON gönderildi:

```
TEAM_807466_FINAL.json
750 kayıt · 106.140 bayt
SHA-256 800d002688c70a033b41e3ce7712008b5c700b42aefa24ae1688e241e01e0899
macro-F1 0.841204 (release ile birebir)
sert 0/1 olasılık: 0/750 · ortalama en yüksek 0.8227
toplam sapması 2.22e-16
```

---

# BÖLÜM 7 — YARIŞMA KURALLARI

## 7.1 Takvim

| tarih | ne |
|---|---|
| **16 Eylül 09:30** | alanda hazır bulunmak **ZORUNLU** |
| 16 Eylül sabah | Lise seviyesi model testi |
| 17 Eylül | değerlendirme, **her seviyede ilk 10** duyurulur |
| 18 Eylül öğleden sonra | **yalnız ilk 10** sunum (10 dk + 3 dk S-C) |

Yer: **Dicle Üniversitesi Konferans Salonu**, Diyarbakır.

## 7.2 Yarışma günü akışı (kılavuz md. 2)

1. Teknik bilgilendirme + hesap açma
2. **Küçük örnek veri** → JSON format testi (puana etkisi yok, kaçırmayın)
3. **Şifreli USB** ile gerçek test verisi; şifre herkese aynı anda; **süre şifreyle başlar**
4. Üret → yükle → süre dolunca sistem kapanır
5. **Birden fazla yükleme serbest**, son geçerli dosya resmî teslim
6. Ayrıca USB ile model paketi teslimi

**Kılavuzun söylemediği üç şey:** süre ne kadar, kaç kayıt, hangi format.
Sadece *"yeterli süre verilir"* yazıyor.

**Madde 7:** *"Takımın kendi bilgisayarı, yazılım ortamı, model dosyaları veya
bağımlılıklarından kaynaklanan sorunlar **süreyi durdurmaz**."*

## 7.3 Puanlama

| bileşen | ağırlık |
|---|---|
| Final model performansı | **%90** |
| Final sunumu | %10 |

**Madde 5:** *"Her seviyede model testi sonucunda **ilk 10'a giren** takımlar sunum
yapmaya hak kazanır."* → Sunum, model performansının **arkasında kilitli**.

Eşitlik bozucu: **PR-AUC** → olasılıkları ezmeyin.

## 7.4 Dış veri

Şartname **§3.1.1 dış veri kullanımına açıkça izin veriyor.**
**§7.2:** final, **görmediğiniz yeni bir veri setinde** dış doğrulama.

## 7.5 Taahhütname

- **md. 2:** "Veri Seti" türetilmiş dosyaları da kapsar
- **md. 5:** GitHub, Drive, Kaggle, HuggingFace, üçüncü taraf YZ servislerine
  yükleme **yasak**

→ Depoda 42 takipli dosyanın tamamı `.py` ve `.md`. Hiçbir kayıt/cache/ağırlık
izlenmiyor. `.gitignore`'a ham uzantılar ve split csv'leri eklendi.

→ **`test_public_ids.txt` USB paketinden çıkarılacak** (türetilmiş dosya).

---

# BÖLÜM 8 — DIŞ VERİ DENEYİ — **KAPANDI**

## 8.1 İndirilen ve taranan

```
external_data altında  77.333 .hea
kaynaklar: Ningbo · Chapman-Shaoxing · PTB-XL · Georgia
```

## 8.2 Gerçek dry-run sonucu (AFIB+AFL)

```
competition readable  = 5000/5000
external .hea         = 77333

aday                     = 10.615
hedef tanı yok           = 63.573
istenmeyen sınıf         =  1.591
ad çakışması             =    950   ← taramanın çalıştığının kanıtı
birden fazla hedef tanı  =    604   ← --single-label eledi

şekil/imza aynı          =    141   ← SIZINTI, eğitimden atıldı
korelasyon >= 0.995      =      0
temiz kabul              = 10.474
```

> **`stem-only overlaps = 950`** en önemli sağlık kontrolüydü. Sıfır çıksaydı
> tarama bozuk demekti — temiz demek değil.

**Normal için SNOMED eşlemesi türetilemedi**, bu yüzden dış veri deneyleri
güvenli biçimde yalnız **AFIB ve AFL** ile yürütüldü.

## 8.3 Beş deney, tek tablo (kontrollü Fold-0, `wide` baseline)

| deney | val(tta) | Δ |
|---|---|---|
| **Competition-only baseline** | **0.8335** | — |
| AFL-only +300 | 0.8316 | −0.0019 |
| AFIB +300, AFL +300 | 0.8260 | −0.0075 |
| 300+300 **sınıf-dengeli** | 0.8164 | −0.0171 |
| pretrain → fine-tune | 0.8162 | −0.0173 |
| AFIB +900, AFL +900 | 0.8128 | −0.0207 |

**Hepsi negatif. Monoton: ne kadar çok dış veri, o kadar kötü.**

AFL-only +300 için eşleştirilmiş bootstrap:

```
gözlenen Δ = -0.00188
%95 aralık = [-0.019361, +0.015187]
P(Δ > 0)   = 0.4004
```

## 8.4 Neden — ölçülen cevap

**Sınıf dengesizliği değil.** Dengeli sampler ile epoch exposure
`[800,800,800,800,800]` yapıldı, düşüş devam etti (−0.0171).

**Classifier head'de de değil.** Pretrain→fine-tune'da 5 sınıflı head
tamamen sıfırdan kuruldu, backbone aktarıldı (166/168 tensör, %99.97).
Yine düştü (−0.0173) ve **AFL 0.6216 → 0.5784** ile en çok zarar gören sınıf oldu.

**Domain/etiket uyumsuzluğu.** Yarışma baseline modeli 600 dış AFIB/AFL kaydına
koşuldu:

| kaynak | AFIB doğru | AFL doğru |
|---|---|---|
| Chapman-Shaoxing | 79/100 | 77/96 |
| PTB-XL | 70/100 | 9/12 |
| Ningbo | — (n=0) | 73/96 |
| **Georgia** | **52/100** ← en kötü | 75/96 |
| **toplam** | **201/300 (%67)** | **234/300 (%78)** |

Georgia AFIB'in %22'si AFIB→AFL yönünde kaydı. Dış AFIB'in bazı kaynaklarda
yarışma AFIB tanımından belirgin farklı olma ihtimali var.

> Bu "veri seti hatalı" kanıtı **değildir** — popülasyon, cihaz ve tanı
> konvansiyonu farkları da mümkün.

## 8.5 Karar

> **Dış veri hattı final sisteme alınmadı.**

Bilimsel olarak güçlü bir sonuç: 77.333 kayıt tarandı, sızıntı temizlendi,
beş farklı strateji denendi, **hiçbiri in-domain validation'da güvenilir kazanç
üretmedi** — ve `test_public` artışlarına bakılarak model seçilmedi.

---

# BÖLÜM 8B — AFIB/AFL UZMAN ARAŞTIRMASI — **KAPANDI**

## 8B.1 Darboğazın büyüklüğü

Fold-0 karışıklık matrisi:

```
        Norm AFIB  AFL LBBB RBBB
Norm  [ 157,   2,   2,   0,   9]
AFIB  [   1, 141,  25,   1,   2]
AFL   [   6,  60,  92,   4,   8]
LBBB  [   3,   0,   4, 161,   2]
RBBB  [   3,   2,   3,   1, 161]
```

```
toplam hata          = 138
AFIB↔AFL çapraz hata =  85
pay                  = 61.6%
```

**Tüm hataların %61.6'sı yalnız bu iki sınıfın karışması.**

### Oracle tavan analizi

Yalnız AFIB↔AFL çapraz hataları kusursuz düzeltilse:

```
baseline = 0.8335  →  oracle = 0.9377   (headroom +0.1043)
```

Gerçek performans değil, **teorik tavan** — ama darboğazın büyüklüğünü gösterir.

## 8B.2 Gate doğru, uzman yetersiz

Ana modelin top-2 sınıfı tam `{AFIB, AFL}` olan kayıtlar:

```
GATE_N              = 271
gerçek AFIB/AFL     = 266
gate saflığı        = 0.9815
çapraz hata kapsamı = 0.9529
non-pair kirlenme   = 5
```

**Gate mükemmele yakın.** Sorun gate değil, uzmanın kendisi:

| uzman | çift F1 | birleşik | Δ |
|---|---|---|---|
| Sıfırdan derin | 0.7070 | 0.8232 | −0.0103 |
| Warm-start | 0.6931 | 0.8264 | −0.0071 |

## 8B.3 En öğretici bulgu: inner → outer çöküşü

```
Warm-start uzman:     inner 0.8492  →  outer 0.6931
Türetilmiş özellikler: inner 0.7706  →  outer 0.6867
```

Küçük inner split'lere aşırı uydurma riski, sayılarla belgelendi.

## 8B.4 Waveform-SVM gate — 5-fold dersi

Ham dalga formundan ~90 yeni ölçüm (harmonik yapı, zamansal frekans kararlılığı,
otokorelasyon periyodikliği, derivasyonlar arası frekans uyumu, atriyal/ventriküler
iletim oranı). Çift F1: **0.7123 → 0.7498 (+0.0375)** — ilk gerçek umut verici sinyal.

Gate ile birleştirildiğinde:

| fold | baseline | wave gate | Δ |
|---|---|---|---|
| 0 | 0.833483 | 0.841856 | **+0.008374** |
| 1 | 0.843981 | 0.855034 | **+0.011053** |
| 2 | 0.821613 | 0.817266 | −0.004346 |
| 3 | 0.829070 | 0.822755 | −0.006315 |
| 4 | 0.856021 | 0.831589 | **−0.024431** |
| **ort.** | **0.836834** | **0.833700** | **−0.003134** |

**İlk iki fold'da +0.008 ve +0.011 görüp durmak cazipti. Durmadık.** Kalan üç
fold'da hiçbir şey değiştirmeden (aynı özellikler, gate, seed, C-grid) koşuldu
ve genellenmedi.

> **Bu projenin en iyi metodolojik dersi:** tek veya iki iyi fold, final
> entegrasyon için yeterli değildir.

---

# BÖLÜM 9 — VERİ KÖKENİ

Kendi 5.000 kaydınızın kaynakları (header adlarından tespit):
**Chapman-Shaoxing · Ningbo · PTB-XL · MIMIC-IV-ECG**

Çözülebilen bölümde:
- **AFIB → Chapman ağırlıklı**
- **AFL → Ningbo ağırlıklı**

Bu bir risk: model sınıfı ayırt etmek yerine kısmen **kaynağı** tanıyor olabilir.
16 Eylül'de yeni veri setinde o koltuk değneği yok.

AFIB/AFL kaynak örtüşmesi %54.1 — yani confound hipotezim büyük ölçüde yanlıştı,
ama tamamen değil. Kaynak-only kural 0.7295, model 0.7376 — **çok yakın.**

> ⚠️ **`tools/source_breakdown.py` hâlâ koşulmadı.** 5 saniye. Çıkardığı
> "dengeli kaynaklarda beklenen doğruluk" sayısı §7.2 dış doğrulamasında ne
> bekleneceğini söyler ve sunumun en güçlü dürüstlük cümlesidir.

---

# BÖLÜM 10 — DOSYA ENVANTERİ

## Çekirdek

| dosya | ne |
|---|---|
| `wfdb_lite.py` | saf-numpy WFDB okuyucu |
| `ecg_preprocess.py` | **ön işlemenin tek kaynağı** — DEĞİŞTİRME |
| `model.py` | mimariler, `PRESETS` = r18/r34/r18k11/wide/w64/w80 |
| `model_diverse.py` | Inception1D, Hibrit (elendi) |
| `train.py` | 5-fold CV, `EXTRA_SPLIT="extra"` |
| `ensemble.py` | ağırlık optimizasyonu |
| `export.py` | ONNX + int8 + PyTorch eşleşme doğrulaması |
| `prep.py` / `prep_fs.py` | cache kurucular |

## Teslim

| dosya | SHA-256 (ilk 16) |
|---|---|
| `package_src/make_submission.py` | `7fbdec417398a527…` |
| `prepare_competition_package.py` | `8e354e15d074295f…` |

## Araçlar

| dosya | ne |
|---|---|
| `tools/add_external.py` | dış veri + 3 kademeli sızıntı taraması (`d87da160…`) |
| `tools/test_add_external.py` | 23 test, 5000 ölçeğinde |
| `tools/compare_runs.py` | eşleştirilmiş McNemar |
| `tools/source_breakdown.py` | kaynak bağımlılığı ⚠️ **koşulmadı** |
| `tools/hiz_teshis.py` | hız darboğazı teşhisi |
| `tools/data_provenance.py` | kaynak tespiti |
| `tools/make_synth_hard.py` | Bayes-tavanı kıyas kümesi |
| `tools/test_preprocess.py` | ön işleme doğrulaması |
| `tools/sph_probe.py` | SPH üst verisi ⚠️ koşulmadı |

## Belgeler

| dosya | durum |
|---|---|
| `MASTER_DOSYA_20260910.md` | **bu belge** |
| `DENEY_KAYDI.md` | asıl deney kaydı |
| `PLAN_DIS_VERI_20260909.md` | dış veri planı |
| `SUNUM_BRIEF.md` | sunum içeriği |
| `GOREV.md` | özgün görev ve değişmezler |
| `PLAN_FINAL.md`, `PLAN_5GUN.md`, `GUNCEL_PLAN_FINAL_20260908.md` | ⚠️ geçersiz |

---

# BÖLÜM 11 — DEĞİŞMEZLER

## Asla ihlal edilmeyecek

1. **`test_public` üzerinde HİÇBİR seçim yapılmaz.** Mimari, hiperparametre,
   ensemble üyeleri, dış veri kararı — hepsi OOF'a bakar.
2. **`ecg_preprocess.py` değiştirilmez.** Ön işlemenin tek kaynağı.
3. **`predict.py`, `manifest.json`, 20 ONNX değiştirilmez.** PyTorch'suz ortamda
   750/750 doğrulanmış final inference kodu.
4. **`release/verified_20model_final_2026-09-03/` asla üzerine yazılmaz.**
5. **Her deney `DENEY_KAYDI.md`'ye yazılır** — negatif sonuçlar dahil.
6. **Yarışma verisi USB paketine kopyalanmaz** (kılavuz md. 6).
7. **Türetilmiş dosyalar üçüncü taraflara gitmez** (taahhütname md. 2/5).
8. Push yalnız `claude/gorev-md-implementation-b2aj7y` dalına.

## Kabul kapıları

| kapı | eşik |
|---|---|
| Hızlandırma | JSON referansla **bit-birebir** aynı, yoksa reddet |
| Dış veri fold-0 | ≤ 0 → DUR |
| Dış veri tam CV | McNemar **p < 0.05**, yoksa gürültü |
| Yeni ensemble | `export.py` PyTorch↔ONNX doğrulaması geçmeli |
| **Sert abort** | **12 Eylül akşamı** |

---

# BÖLÜM 12 — KALAN TAKVİM — **MODEL ARAŞTIRMASI KAPANDI**

## 12.1 Karar (10 Eylül)

> Dış veri ve AFIB/AFL post-processing deneylerinin **hiçbiri** güvenilir
> validation kazancı üretmedi. Model araştırması durdurulmuştur.
> Doğrulanmış 20-model ONNX release korunur; enerji sunum ve teslim
> güvenliğine kaydırılır.

Planladığım "12 Eylül sert abort" tarihine gerek kalmadı — abort koşulu
**iki gün erken ve çok daha fazla kanıtla** karşılandı: 5 dış veri deneyi +
6 uzman/gate deneyi, hepsi negatif.

## 12.2 Kalan takvim

| tarih | iş |
|---|---|
| **10–13 Eyl** | **Sunum.** Yeni model deneyi yok. |
| 13–14 Eyl | Offline prova ×2 (Wi-Fi kapalı, temiz klasör, ikinci bilgisayar) |
| 14 Eyl | Sunum provası ×3, süre tutarak; soru-cevap kartları |
| **15 Eyl** | Freeze. USB ×2 + bulut yedeği. **Kod değişmez.** |
| **16 Eyl** | 09:30 alanda · örnek veri format testi · şifre → koştur → **hemen yükle** |

## 12.3 Yapılacaklar

**Teslim güvenliği (öncelik):**
- [x] `make_submission.py` gerçek veriyle test edildi
- [x] KYS kimlikleri alındı (807466 / 4997133)
- [x] 8 Eylül örnek JSON gönderildi ve doğrulandı
- [ ] Final release SHA-256 kontrolü
- [ ] Offline prova #1 (Wi-Fi kapalı, temiz klasör)
- [ ] Offline prova #2 (mümkünse ikinci bilgisayar)
- [ ] `make_submission.py --validate` hattı tekrar prova
- [ ] USB ×2 + bulut yedeği (yarışma verisi hariç)
- [ ] `test_public_ids.txt` USB'den çıkarıldı

**Sunum:**
- [ ] **Ön işleme sabitlerini KODDAN doğrula** (Bölüm 0.4)
- [ ] Organizasyonun şablonu
- [ ] Her sayının doğru etiketi: OOF / test_public / accuracy / fold-mean
- [ ] 10 dakikaya sığdır, 3 kez süre tutarak prova
- [ ] 3 dakikalık soru-cevap kartları

**YAPILMAYACAK:**
- [ ] ~~Yeni model deneyi~~
- [ ] ~~Son gün eğitim başlatmak~~

---

# BÖLÜM 13 — SUNUM İÇİN

## 13.1 Sayıların DOĞRU adı — bu kritik

Bu üç sayı **farklıdır**, karıştırılmamalı:

```
OOF Macro-F1          = 0.843755    ← model SEÇİMİ bununla yapıldı
test_public Macro-F1  = 0.841204    ← bağımsız doğrulama
test_public Accuracy  = 0.845333    ← farklı metrik
```

Ayrıca **fold ortalaması ≠ OOF**:

```
kontrollü wide baseline fold val(tta):
  0.833483 · 0.843981 · 0.821613 · 0.829070 · 0.856021
  aritmetik ortalama = 0.836834      ← BU BİRLEŞİK OOF DEĞİL
```

Yanlış: *"Doğruluğumuz %84.38"* (eğer OOF macro-F1 kastediliyorsa).
Doğru: *"OOF Macro-F1 %84.38, bağımsız public test Macro-F1 %84.12."*

## 13.2 Anlatılacak hikâye

Çoğu takımda olmayan şey: **elenmiş fikirlerin kaydı — 22 adet.**

> "Her fikri ölçtük ve ölçüme göre eledik. Kendi hipotezlerimiz dahil."

- Bayes tavanı hesaplanabilen sentetik kıyas kümesi
- Eşleştirilmiş **McNemar** testi
- Üç kademeli **sızıntı taraması** (77.333 kayıt tarandı, 141 sızıntı yakalandı)
- **Fresh fold doğrulaması** — iki iyi fold'a kanmadık
- Hız optimizasyonunda **bit-birebir** aynı çıktı kanıtı

## 13.3 En güçlü üç anlatı

**(1) "Daha fazla veri her zaman daha iyi değildir."**
77.333 dış kayıt tarandı, sızıntı temizlendi, beş strateji denendi:

| deney | Δ |
|---|---|
| AFL-only +300 | −0.0019 |
| +300/+300 | −0.0075 |
| sınıf-dengeli | −0.0171 |
| pretrain→FT | −0.0173 |
| +900/+900 | −0.0207 |

Monoton negatif. Sebep ölçüldü: **domain/etiket uyumsuzluğu** (Georgia AFIB
yalnız 52/100 doğru), sınıf dengesizliği değil (dengeli sampler da düzeltmedi).

**(2) "İki iyi fold yeterli değildir."**
Waveform-SVM gate fold 0'da +0.0084, fold 1'de +0.0111 verdi. Durup entegre
etmek çok cazipti. Kalan üç fold'da hiçbir şey değiştirmeden koştuk:
−0.004, −0.006, **−0.024**. Ortalama −0.0031. **Finale almadık.**

**(3) "Hızlandırma doğruluğu bozmamalı."**
92 dk → 6.21 dk (**14.8×**), `class_mismatch = 0`, `max_abs_diff = 0.0`.
Batch inference 4.7× daha hızlıydı ama olasılıkları 0.006 saptırdığı için
reddettik — kılavuz md. 4 eşitliği PR-AUC ile bozuyor.

## 13.4 Jüri soruları — hazır cevaplar

**"En zor sınıf?"**
→ AFIB/AFL, özellikle AFL. Fold-0'da **tüm hataların %61.6'sı** bu iki sınıfın
çapraz karışması (138 hatanın 85'i). Bu çapraz hatalar kusursuz düzeltilse
macro-F1 0.8335 → 0.9377 olurdu.

**"Neden AFIB/AFL'de düşük?"**
→ Dört bağımsız yöntem aynı duvara çarptı (0.701 / 0.760 / 0.725 / 0.744) ve
8.8 M parametreli model küçüğünden kötü. AFL'nin atriyal bandı (2.5–12 Hz)
75 bpm'de QRS harmonikleriyle çakışıyor.

**"Uzman model denediniz mi?"**
→ Evet, altı yolla. Gate'imiz **%98.15 saf** ve çapraz hataların **%95.29'unu**
yakalıyor — sorun gate değil. Sıfırdan uzman 0.7070, warm-start 0.6931; ikisi
de birleşikte kaybettirdi.

**"Neden dış veri yok?"**
→ 77.333 header tarandı, 141 sızıntı yakalandı, beş strateji denendi. Hiçbiri
in-domain validation'da kazanç vermedi. Kullanmadık.

**"Büyük model denemediniz mi?"**
→ base48 0.8335, base64 0.8316, base64+dropout 0.8240, base80+dropout 0.8291.
Kapasite artışı güvenilir kazanç vermedi.

**"250 Hz daha iyi olmaz mıydı?"**
→ Test edildi: 0.8355 vs 0.8335, **+0.002** — önceden belirlediğimiz 0.01
anlamlılık eşiğinin altında. 150 Hz daha verimli, korundu.

**"Test setine bakıp seçim yaptınız mı?"**
→ Hayır. Birkaç deneyde test yükselirken validation düştü (ör. dış veri
+900/+900: test +0.009, val −0.021). **O modelleri seçmedik.**

**"Yarışma bilgisayarında PyTorch gerekir mi?"**
→ Hayır. `.venv_no_torch` ortamında 750/750, 0 hata, manifest farkı 0.000000.

## 13.5 Dürüst sınırlamalar (sunumda söylenebilir)

1. AFIB/AFL hâlâ en zor ayrım; özellikle AFL
2. Ensemble ağırlıkları aynı 4250 OOF üzerinde optimize edildi — meta-seviye
   aşırı uydurma ihtimali sıfır değil
3. `cv10` artışı küçük (+0.0032), "sıçrama" gibi sunulmamalı
4. Public test 750 kayıt → küçük farkların belirsizliği büyük (SE 0.0129)
5. Final yarışma dağılımı public test'ten farklı olabilir (şartname §7.2)

---

# GÜVENLİ GERİ DÖNÜŞ

Herhangi bir adım başarısız olursa:

```
competition_package/  +  MP11
  → 6.21 dk / 750 kayıt
  → macro-F1 0.841204
  → bit-birebir doğrulanmış
  → 750/750, 0 hata
```

Bu paket yarışmaya götürülebilir durumda ve **hiçbir deney onu değiştirmiyor.**

Temel geri dönüş: `release/verified_20model_final_2026-09-03/package` (92 dk).
