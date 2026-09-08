# TEKNOFEST ECG — Güncel Final Planı

**Güncelleme tarihi:** 8 Eylül 2026
**Ana hedef:** 16 Eylül model testine hatasız, offline çalışan ve doğrulanmış bir paketle girmek; model iyileştirmelerini yalnızca güvenli kapılardan geçirmek.

> Bu belge geçerli plandır. `PLAN_FINAL.md` ve `PLAN_5GUN.md` geçersizdir.

---

## 0. BUGÜN 21:00 — organizasyona örnek JSON teslimi

> Organizasyon duyurusu: *"…daha önce paylaşılan örnek veri seti üzerinden
> ürettiğiniz örnek JSON dosyasını özelden gönderiniz."*
> **Son saat: bugün (8 Eylül) 21:00.**

Bu, planın geri kalanının önüne geçer. Şartnamede ve Final Kılavuzu'nda yok —
ayrı bir duyuru. Kılavuz madde 3'ün son maddesi gereği **final puanına etkisi
yoktur**, ama format uyarılarını gerçek testten önce almanın tek yolu budur.
Kaçırılırsa 16 Eylül sabahı formatı ilk kez orada sınamış oluruz.

### Bugünün akışı

| # | İş | Süre | Bloke eden |
|---|---|---|---|
| 0.1 | **Prova koşusu** (`TEAM_TEST`) — hattı kanıtla | ~12 dk | yok, **hemen başlat** |
| 0.2 | KYS'den `team_id` + `application_id` | ~5 dk | t3kys.com erişimi |
| 0.3 | **Teslim koşusu** (gerçek kimliklerle) | ~12 dk | 0.2 |
| 0.4 | `--validate` | 10 sn | 0.3 |
| 0.5 | Özelden gönder | — | 0.4 |

**0.1 ve 0.2 paralel yürür.** Prova koşusunu KYS'yi beklemeden başlat: hatta bir
sorun varsa (yol biçimi, paket kurulumu, süre) bunu saatler önce öğren, 20:00'de
değil. KYS geldiğinde ikinci koşu yalnızca kimlik alanlarını değiştirir.

### Neden iki koşu

`TEAM_TEST` ile üretilmiş dosya organizasyona **gönderilemez**. Kılavuz madde 3:

> "team_name ve kimlik bilgileri **KYS'deki bilgilerle aynı olmalıdır**."

JSON'u elle düzenlemek de yapılmaz — düzenlenen dosya doğrulama hattından
geçmemiş olur. 12 dakikayı tekrar koşmak güvenli olanıdır.

### Gönderirken

**Yalnız organizasyonun belirttiği özel kanaldan gönder.** Taahhütname md. 2 ve
5, yarışma verisinden türetilen dosyaların üçüncü taraflara aktarılmasını
yasaklıyor; bu JSON türetilmiş bir dosyadır. Veri sahibine (organizasyon)
göndermek sorun değil, ama **açık gruba/kanala düşmemeli.**

### Format geri bildirimi gelirse

Kılavuz madde 2.2: *"Gerekli format uyarıları gerçek test başlamadan önce
yapılır."* Bir uyarı gelirse **P0'dır** — dış veri dahil her şeyin önüne geçer.

---

## 1. Değişmez kurallar

- Doğrulanmış final paket korunacak: `release/verified_20model_final_2026-09-03/package`
- 20 ONNX model, PyTorch gerektirmiyor.
- Doğrulanmış test_public macro-F1: **0.841204**
- Accuracy: **0.845333**
- 750/750 kayıt, hata: **0**
- Bu release klasörüne elle müdahale edilmeyecek.
- Yarışma için ayrı `competition_package` kullanılacak.
- Model geliştirme kararları OOF ile verilecek; `test_public` model seçmek için kullanılmayacak.
- Yarışma günü internet yokmuş gibi hazırlanılacak.
- 15 Eylül sonrası kod değiştirilmemesi hedefleniyor.
- Yarışma verisi ve ondan türetilen dosyalar hiçbir üçüncü tarafa (GitHub,
  Drive, Kaggle, HuggingFace, yapay zekâ servisleri) aktarılmayacak — taahhütname md. 5.

## 2. Mevcut doğrulanmış model

### Final 20-model ensemble

Kalan aileler:
- `cv10` — 10 fold
- `main_v2` — 5 fold
- `seed99` — 5 fold

OOF-only pruning:
- 5-aile OOF: `0.842924`
- 3-aile / 20-model OOF: **`0.843755`**
- fark: **`+0.000831`**

Final ONNX paket:
- model sayısı: **20**
- boyut: **101.5 MB**
- test_public macro-F1: **0.841204**
- accuracy: **0.845333**
- 750/750 kayıt
- hata: 0

Bu paket güvenli geri dönüş noktasıdır.

## 3. Elenen deneyler — tekrar açılmayacak

### Balanced decoding
- argmax OOF macro-F1: `0.843755`
- balanced macro-F1: `0.834084`
- fark: **`-0.009671`**
- AFIB/AFL doğruluğu: `0.6988 -> 0.6888`

**Karar: REDDET.**

### QRST / residual özellikleri
- referans macro-F1: `0.8369`
- en iyi residual birleşimi: `0.8376`
- kazanç: **`+0.0006`**

**Karar: BIRAK.**

### Inception Fold-0
- main_v2 accuracy: `0.83765`
- Inception accuracy: `0.83765`
- aynı tahmin oranı: **`0.92353`**
- kurtarılabilir: `64`
- AFIB/AFL içinde kurtarılabilir: `49 / 340`

**Karar: KALDI / full 5-fold YOK.**

### Hybrid Fold-0
- main_v2 accuracy: `0.83765`
- Hybrid accuracy: `0.83765`
- aynı tahmin oranı: **`0.92588`**
- kurtarılabilir: `58`
- AFIB/AFL içinde kurtarılabilir: `46 / 340`

**Karar: KALDI / full 5-fold YOK.**

### mixup çift koruması
- McNemar: `p = 0.7552`, `hb_nopair` ve `hb_nomix` ikisi de `0.8000`

**Karar: GÜRÜLTÜ, REDDET.**

## 4. Yarışma teslim mekaniği — en yüksek öncelik

`competition_package`, `prepare_competition_package.py` ile oluşturuldu.

Doğrulama sonucu:
- [x] 20 ONNX bulundu
- [x] manifestte 20 model bulundu
- [x] `modeller / dosya` şeması doğru tanındı
- [x] diskteki ve manifestteki modeller birebir eşleşiyor
- [x] `manifest.json`
- [x] `predict.py`
- [x] `ecg_preprocess.py`
- [x] `wfdb_lite.py`
- [x] `requirements.txt`
- [x] `README_TESLIM.md`
- [x] toplam boyut: 101.5 MB

Kaynak release değiştirilmedi.

### `make_submission.py`

Kullanılacak sürüm: **`887458a`** (`claude/gorev-md-implementation-b2aj7y`
dalının ucu). Bundan sonrası yok — 887458a güncel sürümün kendisidir.

Uygulanmış ve test edilmiş korumalar (14 senaryo / 50 kontrol geçti):
- Bundle API + `predict_record()` API desteği, `hasattr` ile otomatik seçim
- bizim final paketimizde `predict_record(path)` kullanılacak
- `predict.py` değiştirilmeden çalışır — ön işleme TEKRAR YAPILMAZ
- `predict_record` yol biçimi (`.hea` mi, uzantısız mı) ilk kayıtta keşfedilir
- ilk kayıt gerçek inference ön kontrolü; başarısızsa hemen durur
- 5 elemanlı probability kontrolü
- NaN / Inf kontrolü
- negatif probability kontrolü
- toplam > 0 kontrolü, 1'e normalize etme
- MANIFEST'te model listesi anahtarı tanınmazsa sert durur
- `--max-fail` sınırı; aşılırsa **JSON hiç yazılmaz**
- düşen kayıtta eşit olasılık (**Bundle yolunda da** — sıfır satır üzerine)
- id eksik / fazla / tekrar kontrolü
- sınıf adları: `NORMAL`, `AFIB`, `AFL`, `LBBB`, `RBBB`
- `predicted_class` ile argmax tutarlılığı
- `competition_level = LISE`
- JSON yazıldıktan sonra diskten tekrar validate
- sınıf dağılımı çıktısı
- >%90 tek sınıf uyarısı
- toplam süre / ms/kayıt / kayıt/dakika

#### Koşarken çıkabilecek iki bilgilendirme satırı — ikisi de normal

- `NOT: predict_record uzantisiz WFDB taban adi bekliyor`
  → yol biçimi keşfedildi, sorun yok.
- `NOT: N kayitta paketin sinif indeksi ile en yuksek olasilik ayni degil`
  → paket ağırlıklı oylama kullanıyor. JSON'a kılavuz madde 3 gereği **argmax**
  yazılır. `N` büyükse (yüzlerce) incelenmeli.

### Sıradaki kritik test

1. Güncel `make_submission.py` proje köküne konacak.
2. `prepare_competition_package.py` tekrar çalıştırılacak.
3. `competition_package` içine güncel dosya yerleşecek.
4. Yalnız 750 test_public header ID'siyle gerçek JSON üretilecek.
5. JSON diskten tekrar `--validate` edilecek.
6. Süre ölçümü kaydedilecek.

**Prova koşusu** (hemen, KYS beklemeden):

```bat
python make_submission.py --root "D:\TUNA_ISPIR\Documents\Claude\Projects\SYZ" --ids test_public_ids.txt --team-name "tkt-26" --team-id TEAM_TEST --application-id BASVURU_TEST --out TEST_PUBLIC_PROVA.json
```

**Teslim koşusu** (KYS geldikten sonra — bugün 21:00'e gidecek dosya):

```bat
python make_submission.py --root "D:\TUNA_ISPIR\Documents\Claude\Projects\SYZ" --ids test_public_ids.txt --team-name "tkt-26" --team-id <KYS_TAKIM_ID> --application-id <KYS_BASVURU_ID>
```

Çıktı adı otomatik: `TEAM_<KYS_TAKIM_ID>_FINAL.json`

**Doğrulama** (`--ids` verilince `--root` gereksiz):

```bat
python make_submission.py --validate TEAM_<KYS_TAKIM_ID>_FINAL.json --ids test_public_ids.txt
```

Son satır tam olarak `all checks passed` olmalı. Değilse **gönderme.**

Bu adımlar geçmeden submission hattı tamamlanmış sayılmayacak.

## 5. KYS bilgileri — P0, bugün 21:00'i bloke ediyor

Gerçek değerler bugün öğrenilip offline not edilmeli:
- [ ] `team_id`
- [ ] `application_id`
- [ ] takım adının resmi yazımı

Gerçek teslimde `TEAM_TEST` / `BASVURU_TEST` kullanılmayacak.
**Bugünün 21:00 teslimi de gerçek teslim sayılır** — kimlik alanları KYS ile
birebir aynı olmalı.

## 6. Dış veri — güncel indirme durumu

Challenge 2021 kaynakları paralel indiriliyor. **Arka planda devam etsin**,
bugünkü P0 işini engellemiyor.

Son ölçülen durum:

| Kaynak | İnen `.hea` | Toplam | Durum |
|---|---:|---:|---:|
| Ningbo | 34,905 | 34,905 | **%100** |
| Chapman-Shaoxing | 5,015 | 10,247 | **%48.9** |
| PTB-XL | 5,059 | 21,837 | **%23.2** |
| Georgia | 5,024 | 10,344 | **%48.6** |

Toplam:
- indirilen: **50,003**
- hedef: **77,333**
- yaklaşık genel ilerleme: **%64.7**

İlerleme kontrol komutu:

```bat
cd /d D:\TUNA_ISPIR\Documents\Claude\Projects\SYZ\ecg_train_v2 && for %S in (ningbo chapman-shaoxing ptb-xl georgia) do @echo ==== %S ==== & @dir /s /b external_data\%S\*.hea 2>nul | find /c /v ""
```

## 7. Kaynak / provenance bulgusu

Mevcut 5000 kayıt içindeki orijinal header isimlerinden en az şu kaynaklar tespit edildi:
- Chapman-Shaoxing
- Ningbo
- PTB-XL
- MIMIC-IV-ECG

Çözülebilen bölümde:
- AFIB: Chapman ağırlıklı
- AFL: Ningbo ağırlıklı

Bu nedenle dış veri eklerken yalnız sınıf sayısını değil, **kaynak çeşitliliğini**
de artırmak önemli.

> ⚠️ `tools/source_breakdown.py` (5 sn) **hâlâ koşulmadı.** Çıkardığı "dengeli
> kaynaklarda beklenen doğruluk" sayısı §7.2 dış doğrulamasında ne bekleneceğini
> söyler ve sunumun en güçlü dürüstlük cümlesidir. Submission hattı kapandıktan
> sonra ilk iş.

## 8. Dış veri eğitim hattı

Veriler tamamlandığında veya yeterli alt küme hazır olduğunda:

### A. Sızıntı kontrolü
`add_external.py` ile:
- kayıt adı çakışması
- sinyal / şekil imzası
- yeniden adlandırılmış kopya riski
- test_public sızıntısı

kontrol edilecek.

**test_public ile çakışan hiçbir kayıt eğitime girmeyecek.**

### B. Dry-run

```bat
python tools\add_external.py --source <DIS_VERI_KLASORU> --cache cache --out cache_ext --dry-run
```

Kontrol:
- etiket haritası
- çakışan kayıt sayısı
- test_public ile sızıntı
- kaynak × sınıf dağılımı
- sınıf başına gerçekten yeni kayıt sayısı

**Sızıntı satırları sıfırsa DUR** — tarama çalışmıyor demektir, tarama sıfır
bulmuyor demek değil.

### C. Fold-0 kapısı

Yeni dış veriler yalnız training fold'a girecek. Validation / OOF yarışmanın
kendi development verisinden kalacak.

Karar:
- Fold-0 kazanç `<= 0` → **DUR**
- yaklaşık `+0.004 ... +0.01` → ikinci kontrollü Fold-0 düşünülebilir
- `> +0.01` → tam CV için güçlü aday

### D. Tam CV

Yalnız Fold-0 kapısı geçerse:
- tam 5-fold
- OOF karşılaştırması
- McNemar (`p < 0.05` değilse kazanç gürültüdür)
- yeni ensemble değerlendirmesi

Anlamlı avantaj yoksa eski 20-model paket teslim edilir.

**Takvim gerçeği:** dış veri hattının tamamı (indirme bitişi + ön işleme 1–2 sa
+ fold-0 + tam CV) en iyi ihtimalle 12 Eylül'ü bulur. 15 Eylül freeze'ine kadar
McNemar'ı da yetiştirmek gerekir. Kapı geçmezse bu bir başarısızlık değil,
planlanmış sonuçtur.

## 9. Offline prova

Submission JSON hattı doğrulandıktan hemen sonra:

### Prova 1
- Wi-Fi kapat
- ethernet çıkar
- temiz klasöre `competition_package` kopyala
- minimal ortamla inference
- JSON üret
- `--validate`
- süre tut

### Prova 2
Mümkünse başka bilgisayar veya temiz ayrı ortam.

Amaç: **16 Eylül'de internet olmadan ilk denemede çalışmak.**

## 10. Hız

Bilmemiz gereken:
- 750 kayıt toplam süre
- ms/kayıt
- kayıt/dakika

**Beklenti** (bench ölçümlerinden hesaplandı, henüz uçtan uca doğrulanmadı):
659.6 ms ön işleme + 20 × 14.11 ms model ≈ **~942 ms/kayıt**
→ 750 kayıt ≈ **~12 dakika**, ≈ **64 kayıt/dakika**

Gerçek sayı bugünkü koşudan gelecek. **Bu aralığın çok dışındaysa dur ve
sebebini bul** — özellikle çok hızlıysa (kayıtlar sessizce düşüyor olabilir).

Yarışma test kümesi 750'den büyük olabilir; kayıt/dakika sayısı salonda kaç
dakika gerektiğini hesaplamanın tek yoludur.

## 11. USB teslim paketi

```text
competition_package/
├── models/                 (20 ONNX)
├── manifest.json           zorunlu
├── predict.py              zorunlu
├── ecg_preprocess.py       zorunlu
├── wfdb_lite.py            zorunlu
├── make_submission.py
├── preprocess.json         (varsa; zorunlu listede değil)
├── requirements.txt
└── README_TESLIM.md
```

Yarışma test verisi USB paketine kopyalanmayacak.

Ayrıca:
- [ ] ana USB
- [ ] ikinci USB yedeği
- [ ] yarışma verisi hariç bulut yedeği
- [ ] mümkünse ikinci bilgisayar

## 12. Yarışma günü — 16 Eylül

1. 09:30'dan önce alanda hazır ol.
2. Örnek veri ile JSON format kontrolünü hemen yap — ücretsiz kontrol, kaçırma.
3. Şifre açıklanınca:
   - test verisini aç
   - inference
   - JSON üret
   - `--validate`
4. **İlk geçerli JSON oluşur oluşmaz yükle.** Kılavuz madde 2.5: son geçerli
   dosya resmî teslimdir, hatalı bir sonraki yükleme öncekini geçersiz kılmaz.
   Erken bir geçerli dosya en değerli sigortadır.
5. Süre kalırsa daha iyi sürüm çalıştır ve tekrar yükle.
6. USB model paketini teslim et.

## 13. Sunum

Sunum model testinde ilk 10'a girildikten sonra kullanılacak.
10 dakika + 3 dakika soru-cevap. Organizasyonun şablonu bekleniyor.
İçerik hazır: `SUNUM_BRIEF.md`.

Hazırlanacak:
- problem tanımı
- veri / leakage disiplini
- preprocessing
- model
- 5-fold OOF
- pruning
- PyTorch'suz ONNX paket
- dış veri ve genelleme
- elenen deneyler
- canlı demo

Güçlü savunma noktaları:
- fikirler ölçülerek elendi
- QRST: +0.0006 → bırakıldı
- balanced decoding: -0.0097 → bırakıldı
- mixup çift koruması: McNemar p=0.755 → gürültü
- Inception / Hybrid yeterli hata çeşitliliği üretmedi
- 30 → 20 model pruning OOF'u bozmadı
- frozen release + snapshot yaklaşımı
- offline, PyTorch'suz deployment

## 14. Öncelik sırası

### P0 — BUGÜN, 21:00'den önce
- [ ] **prova koşusu (`TEAM_TEST`) — hemen başlat, KYS'yi bekleme**
- [ ] süre / kayıt-dakika ölç
- [ ] sınıf dağılımını kontrol et
- [ ] **gerçek `team_id`** (t3kys.com)
- [ ] **gerçek `application_id`** (t3kys.com)
- [ ] teslim koşusu (gerçek kimliklerle)
- [ ] JSON `--validate` → `all checks passed`
- [ ] **özelden gönder**

### P1 — Yarın
- [ ] offline prova #1
- [ ] offline prova #2
- [ ] format geri bildirimi geldiyse uygula (gelirse P0'a çıkar)

### P2 — Paralel, arka planda
- [ ] Chapman tamamla
- [ ] PTB-XL tamamla
- [ ] Georgia tamamla
- [x] Ningbo tamamlandı

### P3 — Submission güvenli olduktan sonra
- [ ] `source_breakdown.py` (5 sn)
- [ ] `sph_probe.py`
- [ ] `add_external.py --dry-run`
- [ ] leakage kontrolü
- [ ] dış veri Fold-0

### P4 — Yalnız kapı geçerse
- [ ] full CV
- [ ] McNemar
- [ ] yeni paket
- [ ] eski 0.841204 paket ile OOF tabanlı seçim

### P5 — Final
- [ ] sunum
- [ ] USB
- [ ] ikinci USB
- [ ] son offline prova
- [ ] 15 Eylül freeze

## 15. Bugün için tek net hedef

**21:00'e kadar organizasyona geçerli bir örnek JSON göndermek.**

Bunun yan ürünü olarak submission hattı da kapanmış olur. Dış veri indirmeleri
arka planda devam eder; bugün başka model işi başlatılmaz.

Başarı kriteri:

```text
750 test_public
↓
20-model gerçek ONNX inference
↓
TEAM_<KYS_ID>_FINAL.json   (gerçek kimliklerle)
↓
diskten tekrar --validate
↓
0 format hatası
↓
sınıf dağılımı makul  (~%40 NORMAL, tek sınıfta yığılma YOK)
↓
süre kaydedildi
↓
özelden gönderildi   ← 21:00
```

Bu tamamlanmadan yeni model eğitimi başlatma.

## Güvenli geri dönüş

Herhangi bir yeni deney başarısız olursa:

`release/verified_20model_final_2026-09-03/package`

yarışmaya götürülebilecek doğrulanmış temel pakettir.
