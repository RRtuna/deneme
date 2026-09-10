# TEKNOFEST 2026 ECG — Ana Belge

**Takım:** tkt-26 · Özel İzmir Bahçeşehir 50. Yıl Fen ve Teknoloji Lisesi
**Kategori:** Sağlıkta Yapay Zekâ, **Lise Seviyesi**
**Güncelleme:** 10 Eylül 2026 · **Freeze:** 15 Eylül · **Yarışma:** 16–18 Eylül, Dicle Üniversitesi

> Bu belge, projenin tamamının tek kaynaklı özetidir. Sıfır bağlamla okunabilir:
> ne yaptık, hangi sayıyı nereden aldık, neyi neden eledik, yarışma günü ne olacak.
> Bir çelişki görürsen `DENEY_KAYDI.md` ve git geçmişi asıl kayıttır.

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

**Güven aralığı:** SE = 0.0129, %95 CI = [0.8140, 0.8644].

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

# BÖLÜM 4 — ELENEN FİKİRLER (14 adet)

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

# BÖLÜM 8 — DIŞ VERİ DENEYİ (devam ediyor)

## 8.1 İndirilen kaynaklar

| kaynak | inen `.hea` | toplam |
|---|---|---|
| Ningbo | 34.905 | 34.905 (%100) |
| Chapman-Shaoxing | 5.015 | 10.247 |
| PTB-XL | 5.059 | 21.837 |
| Georgia | 5.024 | 10.344 |

## 8.2 `add_external.py` — SHA-256 `d87da160…`

### Üç kademeli sızıntı taraması

1. **Kayıt adı** (uzantı/harf duyarsız)
2. **Şekil imzası** — her derivasyon z-skorlanır → kazanç/ofset duyarsız
3. **Korelasyon ≥ 0.995** — II derivasyonu 128 noktaya indirgenir → yeniden
   örneklenmiş/kırpılmış kopyaları yakalar

### Neden kritik

`test_public` kayıtları açık veri setlerinin **içinde**. Sızarsa test üzerinde
eğitmiş olursunuz ve **OOF'ta göremezsiniz** — skor yükselir, gerçek başarı düşer.

**Gerçek örnek:** `record_id = NORM_000777` ama dosya adı `JS36591`, ve aynı
`JS36591` `external_data\ningbo\g26` altında da var. Stem karşılaştırması
`record_id` üzerinden yapılsaydı **bu sızıntı kaçırılırdı.** Artık çözümlenmiş
gerçek `.hea` yolundan alınıyor.

### Şema uyumluluğu

Gerçek cache: `record_id, relative_path, header_path, signal_path, label,
class_id, sampling_rate_hz, lead_count, duration_sec, file_format, split`

Yol çözümü: olduğu gibi → proje kökü → veri kökü → `data/` öneki atılıp veri kökü.

### Sert kapı

Tüm yarışma kayıtları okunmadan cache **yazılmaz**. Okunamayan her kayıt
taramanın kör noktasıdır; o durumda `leak=0` bir kanıt değil, yanıltmadır.

### Diğer korumalar

- `--single-label` **varsayılan açık** — çoklu SNOMED'li kayıt atlanır
- `--balance-sources` — kota sınıf içinde kaynaklar arasında dağıtılır
  (Ningbo 34.905 vs yarışma 5.000 = 7:1; dengelenmezse model sınıfı değil
  **kaynağı** öğrenir)
- Etiket haritası tahmin edilmez, **kendi verinizden türetilir**
- Eklenenler `split="extra"` → her fold'un **eğitimine**, hiçbir fold'un
  **doğrulamasına** girmez → OOF yalnız yarışma verisinde ölçülmeye devam eder

Test: `tools/test_add_external.py` — 350 kayıtta 23/23, **5000 kayıtta 23/23**.

## 8.3 Fold-0 sonucu (9 Eylül)

```
Baseline:            best EMA val 0.8370 @ ep30 · TTA val 0.8335 · test_public 0.8281
+1800 (900 AFIB
       + 900 AFL):   best EMA val 0.8158 @ ep11 · TTA val 0.8128 · test_public 0.8371
```

### Gürültü tabanı (simüle edildi)

| ölçüm | 1 SE | gözlenen | kaç SE |
|---|---|---|---|
| fold-0 val (1000 kayıt) | 0.011 | −0.021 | 1.98 |
| test_public (750 kayıt) | 0.013 | +0.009 | **0.72** |

**Tek fold, tek seed. Test farkı açıkça gürültü.**

### Nedenler, önem sırasıyla

**(a) Sınıf öncülü kayması — en büyük ve mekanik.** `train.py`'de sınıf ağırlığı
da dengeli örnekleyici de **yok**:

```
AFIB    800 → 1700   %20 → %29.3   (×1.47)
AFL     800 → 1700   %20 → %29.3   (×1.47)
diğer 3  800 → 800   %20 → %13.8   (×0.69)
```

Dengeli val kümesinde bu, diğer üç sınıfın recall'unu düşürür → macro-F1 düşer.
**Gözlenen yön tam olarak beklenen yön.**

**(b) `ep30 → ep11` — (a)'nın parmak izi.** `train.py:528`: *"patience 99 lets
cosine fully anneal, which beat early stopping"*. Baseline 30/40'ta (%75 anneal),
external 11/40'ta (%27.5) zirve yaptı ve sonra 29 epoch **kötüleşti**. Bu,
"eğitim dağılımına yakınsadıkça dengeli val'den uzaklaşma" imzasıdır.

**(c) Zamanlama uyumsuzluğu** — +%45 veri = epoch başına +%45 adım; cosine 40
epoch'a göre ayarlı. Karşılaştırma elmayla elma değil.

**(d) Domain kayması** — gerçek ama bu desende değil.

**(e) Etiket konvansiyonu** — gerçek risk ama (a)–(c)'den ayrıştırılamıyor.

### Karar: deney kurgusu karışık

Aynı anda üç şey değişti — boyut (+%45), öncül (uniform → çarpık), alan.
Atıf yapılamaz. **Direct mixing'in kendisi değil, dengelenmemiş direct mixing
hatalı.**

## 8.4 Sıradaki üç deney

**E0 — sınıf bazlı F1 kırılımı · maliyet SIFIR**

| gözlenen | anlamı |
|---|---|
| AFL/AFIB ↑, diğer üç ↓ | öncül kayması doğrulandı |
| AFL ↓ | etiket konvansiyonu sorunu |
| hepsi hafif ↓ | domain kayması |

**E1 — baseline fold-0, ikinci seed · 1 fold**
Fold-0 val'in seed'ler arası oynamasını bilmeden hiçbir karşılaştırma
yorumlanamaz.

**E2 — öncül korunarak karıştırma · 1 cache + 1 fold**
Beş sınıfa da eşit ekle (`--per-class N --balance-sources`, N = en kıt sınıf).
val toparlanıyorsa → öncül kaymasıydı. Hâlâ düşükse → **DUR**.

## 8.5 Yapılmayacaklar

1. **`test_public`'teki +0.009'a bakarak seçim yapmak** — GOREV.md'nin birinci
   kuralını çiğner ve 0.72 SE'lik gürültüye 20 checkpoint yatırmaktır
2. Tam 5-fold × 3 aile başlatmak (karışık öncül üzerine 20 checkpoint)
3. Pretrain → fine-tune (her modeli iki kez = 40 koşu, 5 güne sığmaz)
4. `+300 AFIB +300 AFL` (aynı karışık deneyin küçüğü, hiçbir şeyi ayrıştırmaz)

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

# BÖLÜM 12 — KALAN TAKVİM

| tarih | iş |
|---|---|
| **10 Eyl** | E0 (bedava) → E1 + E2 (gece) |
| **11 Eyl** | E1/E2 sonucu; geçerse tam CV başlat |
| **12 Eyl akşam** | **SERT ABORT** — McNemar-anlamlı sonuç yoksa MP11 teslim |
| **13 Eyl** | (geçtiyse) ensemble → export → MP11 prova → karşılaştırma |
| **14 Eyl** | sunum, offline prova ×2 |
| **15 Eyl** | freeze, USB ×2, bulut yedeği. **Kod değişmez.** |
| **16 Eyl** | 09:30 alanda; örnek veri format testi; şifre → koştur → **hemen yükle** |

## Yapılacaklar

**Teslim:**
- [x] `make_submission.py` gerçek veriyle test edildi
- [x] KYS kimlikleri alındı
- [x] 8 Eylül örnek JSON gönderildi
- [ ] Offline prova #1 (Wi-Fi kapalı, temiz klasör)
- [ ] Offline prova #2 (mümkünse ikinci bilgisayar)
- [ ] USB ×2 + bulut yedeği
- [ ] `test_public_ids.txt` USB'den çıkarıldı

**Model:**
- [ ] `source_breakdown.py` (5 sn) ⚠️
- [ ] E0 sınıf bazlı F1 kırılımı
- [ ] E1 ikinci seed
- [ ] E2 öncül korunarak
- [ ] (geçerse) tam CV + McNemar

**Sunum:**
- [ ] Organizasyonun şablonu
- [ ] 10 dakikaya sığdır
- [ ] 3 kez süre tutarak prova
- [ ] Soru-cevap hazırlığı

---

# BÖLÜM 13 — SUNUM İÇİN

## Anlatılacak hikâye

Çoğu takımda olmayan bir şey var: **elenmiş fikirlerin kaydı.**

> "Her fikri ölçtük ve ölçüme göre eledik."

- Bayes tavanı hesaplanabilen bir sentetik kıyas kümesi kurduk
- Eşleştirilmiş **McNemar** testi kullandık
- Üç kademeli **sızıntı taraması** yazdık
- Fikirleri **kapılardan** geçirdik, sezgiyle değil

## Muhtemel sorular

**"Neden AFIB/AFL'de düşük?"**
→ Dört bağımsız yöntem aynı duvara çarptı (0.701/0.760/0.725/0.744). 8.8 M
parametreli model küçüğünden kötü. Hataların %93.4'ü kararsız bölgede. Bu veri
kıtlığı değil, sinyal/etiket sınırı. AFL'nin atriyal bandı (2.5–12 Hz) 75 bpm'de
QRS harmonikleriyle çakışıyor.

**"Modeli neden büyütmediniz?"**
→ Büyüttük. 8.8 M parametreli GPU modeli 0.701 verdi, bizim küçük ensemble 0.760.

**"Genelleme?"**
→ `source_breakdown.py` sayısı + dış veri deneyi. Kaynak-only kural 0.7295,
model 0.7376 — farkın küçüklüğünü biliyoruz ve söylüyoruz.

**"Hızlandırma doğruluğu bozmadı mı?"**
→ 92 dk → 6.21 dk, **14.8×**. `class_mismatch = 0`, `max_abs_diff = 0.0`.
Batch inference 4.7× daha hızlıydı ama olasılıkları 0.006 saptırdığı için
reddettik — PR-AUC eşitlik bozucu.

## Dürüstlük cümlesi

> "Geliştirmede 0.8412 aldık. Kaynak dengesi düzeltildiğinde beklentimiz şu —
> ve bunu ölçtük."

Jüri önünde bunu söyleyebilen takım az olacak.

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
