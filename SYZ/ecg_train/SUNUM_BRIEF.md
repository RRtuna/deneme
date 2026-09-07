# TEKNOFEST 2026 EKG Projesi — Sunum Hazırlığı Devir Belgesi

> **Bu belge ne işe yarar:** Bu projeyi hiç bilmeyen bir sohbete/kişiye
> devretmek için yazıldı. Amacı **final sunumunu hazırlamak**. Sunumda
> kullanılabilecek her gerçek sayı, her karar, her elenen fikir ve her
> mühendislik bulgusu burada. Tahmin veya süsleme yok — doğrulanmamış olanlar
> ayrıca işaretlendi.
>
> **Kimin projesi:** Takım `tkt-26`, Özel İzmir Bahçeşehir 50. Yıl Fen ve
> Teknoloji Lisesi. Lise seviyesi, Sağlıkta Yapay Zekâ Yarışması.
>
> **Tarih:** Bu belge 7 Eylül 2026'da yazıldı. Yarışma ~11-12 Eylül.

---

## 1. GÖREV

12 derivasyonlu EKG kayıtlarından **5 sınıflı** sınıflandırma:

| indeks | sınıf | açılım |
|---|---|---|
| 0 | Normal | normal sinüs ritmi |
| 1 | AFIB | atriyal fibrilasyon |
| 2 | AFL | atriyal flutter |
| 3 | LBBB | sol dal bloğu |
| 4 | RBBB | sağ dal bloğu |

- Sinyal: 12 derivasyon, 500 Hz, 10 saniye, WFDB formatı (`.hea` + `.mat`/`.dat`)
- Metrik: **macro-F1**
- Kısıt: **Yalnızca CPU**, GPU yok, bulut yok
- Teslim: **ONNX** — kullanıcı makinesinde PyTorch olmayacak

### Veri

TEKNOFEST tarafından sağlanan, gizlilik taahhütnamesine tabi 5.000 kayıt:

| bölüm | kayıt |
|---|---|
| `train.csv` | 3.500 |
| `validation.csv` | 750 |
| `test_public.csv` | 750 |

Her sınıftan tam 1.000 kayıt — mükemmel dengeli.
**Geliştirme kümesi = train + validation = 4.250 kayıt.** Tüm çapraz doğrulama
bunun üzerinde yapıldı.

---

## 2. ŞARTNAMEDEN — SUNUMU DOĞRUDAN ETKİLEYEN MADDELER

Bunlar sunumun çerçevesini belirliyor, mutlaka okunmalı.

### §7.4 ve §7.5 — puanlama

> "Final Yarışmaları (Fiziki) **%90**, Final Sunum Puanlaması **%10**."
> "Proje Sunuş Formu 0, Proje Detay Raporu 0."

**Sonuç:** Sunum tek başına 10 puan. Modeli 0.8412'den 0.85'e çıkarmak 90
puanın içinde ~1 puanlık bir hareket. **Sunumun kaldıracı 10 kat daha büyük.**

> "Yarışma jürisi, finale kalan takımların **kodlarını tekrar çalıştırmasını**
> ve beyan ettikleri sonuçları bulmalarını isteme yetkisine sahiptir."

**Sonuç:** Bu bir puan değil, bir eşik. Yerinde koşturulamayan bir sistem
diğer her şeyi geçersiz kılar. Sunumda "tek komutla, PyTorch olmadan,
internetsiz çalışır" diyebilmek kritik.

### §7.2 — final değerlendirmesi yeni bir veri setinde

> "Model performansının klinik gerçeklik içinde sınanabilmesi amacıyla
> **external validasyon** yapılacaktır. TEKNOFEST için **özgün ve tamamen
> anonimleştirilmiş yeni bir EKG veri seti** kullanılacaktır... farklı hasta
> gruplarında nasıl davrandığını test etmek hem de **genelleme yeteneğini**
> değerlendirmek amacıyla."

**Sonuç:** Sıralamayı belirleyen `test_public` DEĞİL, hiç görülmemiş başka bir
kaynaktan gelen veri. Sunumun ana teması **genelleme** olmalı, "kendi test
setimde şu skoru aldım" değil.

### §3.1.1 — dış veri serbest

> "Yarışmacılar erişime açık olan farklı veri setlerini ve/veya kendi
> oluşturacakları veri setlerini de model eğitimi için kullanabileceklerdir."

Kaynak olarak PhysioNet ECG Arrhythmia Dataset 1.0.0 gösteriliyor (~45.000
kayıt, 12 derivasyon, 500 Hz, SNOMED-CT).

### Veri Seti Kullanım ve Gizlilik Taahhütnamesi

- **md. 2:** "Veri Seti" ham kayıtlarla sınırlı değil — "bunlardan
  türetilebilecek tüm veri, dosya ve içerikleri" de kapsıyor
- **md. 5:** GitHub, Google Drive, Kaggle, Hugging Face vb. platformlara
  **yüklenemez**; "üçüncü taraf yapay zekâ servisleri" de sayılıyor
- **md. 3:** başka bir yarışma/proje kapsamında kullanılamaz
- **md. 4:** ticari kullanım yasak

**Sunumda gösterilirken:** ekran görüntülerinde hasta kimliği ya da ham kayıt
listesi olmasın. Toplu metrikler (karışıklık matrisi, F1 tablosu) sorun değil.

---

## 3. SİSTEM — NE İNŞA EDİLDİ

### Boru hattı

```
WFDB okuma  →  ön işleme  →  cache  →  5-fold CV eğitim  →  ensemble  →  ONNX paket
(wfdb_lite)   (ecg_preprocess)  (prep)      (train)          (ensemble)     (export)
```

**Hiçbir dış kütüphaneye bağımlı değil:** WFDB okuyucu ve tüm sinyal işleme
sıfırdan yazıldı (saf numpy). `scipy` varsa hızlandırıcı olarak kullanılıyor,
yoksa saf-numpy referans yolu devreye giriyor — sonuç aynı.

### Ön işleme (`ecg_preprocess.py`)

- Butterworth yüksek geçiren (0.5 Hz, 2. derece) + alçak geçiren (40 Hz, 4. derece)
- 50/60 Hz çentik filtreleri (Q = 30)
- Sıfır fazlı `sosfiltfilt` (tek sayılı yansıma dolgusu ile)
- 500 Hz → **150 Hz** yeniden örnekleme, 10 sn = 1500 örnek
- Kayıt başına global normalizasyon, ±8σ kırpma
- **37 elle çıkarılmış özellik**: RR aralığı istatistikleri (19), atriyal
  ölçümler (11), QRS morfolojisi (7)

**Kritik tasarım kuralı:** ön işleme **tek kaynaktan** gelir. Eğitim ve
çıkarım aynı fonksiyonu çağırır. İkinci bir filtre kodu yok. Bu, eğitim/çıkarım
tutarsızlığı denen sinsi hata sınıfını baştan imkânsız kılar.

**Doğrulama:** elle kurulan Butterworth katsayıları `scipy.signal.butter` ile
**1e-12 hassasiyetinde** aynı çıkıyor. `tools/test_preprocess.py` 27 kontrol
koşuyor.

### Model (`model.py`)

- **ResNet1D + Squeeze-Excitation** ana omurga
- Yanında **37 özellik için ayrı MLP dalı**, sonra birleştirme
- Özellik ölçekleyici **modülün içinde** (ONNX kendi kendine yeter, ayrı bir
  scaler dosyası ağırlıklardan kopamaz)
- Preset'ler: `r18`, `r34`, `r18k11`, `wide`, `w64`, `w80`
- Ek modül `model_diverse.py`: Inception1D ve CNN+Transformer hibrit

### Eğitim (`train.py`)

- 5-fold katmanlı çapraz doğrulama
- EMA (üstel hareketli ortalama), kosinüs öğrenme oranı, mixup
- Agresif veri artırma: kaydırma, ölçek, derivasyon düşürme, gürültü,
  gezinme, frekans maskesi — her biri ayrı ayrı kapatılabilir
- TTA (`shift3`)
- **Devam edebilir:** her fold bitince `done.json` yazar; makine uyursa bir
  fold kaybedersin, tüm işi değil

### Teslim paketi (`export.py`)

- Her fold için ayrı ONNX grafiği (opset 17, TorchScript ihracatçısı)
- **int8 dinamik nicemleme** — 3.8× küçültme
- `manifest.json`: üyeler, ağırlıklar, skorlar, SHA-256 sağlamaları
- Paket `ecg_preprocess.py` ve `wfdb_lite.py`'nin **birebir kopyasını** içerir
- **Kendi kendini doğrular:** ONNX skoru PyTorch skoruyla karşılaştırılır,
  fark > 0.005 ise ihracat başarısız sayılır

### Sunum uygulaması (`package_src/ecg_demo.py`)

Tarayıcıda çalışan, yalnızca standart kütüphane kullanan yerel sunucu:
- **Tek Kayıt** — bir EKG seç, sinyali gör, tahmini ve sınıf olasılıklarını gör
- **Toplu Skor** — bir CSV üzerinde koş, macro-F1 ve karışıklık matrisi
- **Sistem** — model bilgisi, hız ölçümü, sağlama doğrulaması
- **Karşılaştır** — iki paketi yan yana koy

Sunumda canlı demo için birebir bu.

---

## 4. SONUÇLAR — GERÇEK SAYILAR

### Ana sonuç

| ölçüm | değer |
|---|---|
| **test_public macro-F1** | **0.841204** |
| test_public doğruluk | 0.845333 |
| geliştirme OOF macro-F1 | 0.843755 |
| standart hata (SE) | 0.0129 |
| **%95 güven aralığı** | **[0.8140, 0.8644]** |

Güven aralığı, karışıklık matrisinden 20.000 kez bootstrap örneklemesiyle
hesaplandı.

**Kıyaslar:**
- Taban hedef: 0.8378 → **geçildi**
- Kaggle'daki en iyi referans: 0.8572 → **güven aralığının içinde kalıyor**
  (yani aradaki fark istatistiksel olarak anlamlı değil)

### Sınıf bazında

Normal, LBBB ve RBBB neredeyse mükemmel. Tüm hata **AFIB ↔ AFL** ikilisinde
toplanıyor:

| ölçüm | değer |
|---|---|
| AFIB/AFL ikili doğruluk (geliştirme OOF) | **0.7376** |
| AFIB/AFL ikili doğruluk (test) | 0.751 |
| AFL duyarlılık (recall) | 0.573 |

Model "bu ikisinden biri" demeyi %91.8 doğru yapıyor; ikilinin **içinde**
%74'te kalıyor. İkili doğruluk %90 olsa macro-F1 ≈ 0.90 olurdu.

### Hız (kullanıcının makinesinde ölçüldü)

| bileşen | süre |
|---|---|
| ön işleme (sabit) | 659.6 ms/kayıt |
| model başına çıkarım | 14.11 ms |

Yani toplam sürenin **~%70'i ön işleme**. Model küçültmenin getirisi sınırlı —
bu, sunumda "neden modeli daha da küçültmedik" sorusunun cevabı.

---

## 5. DARBOĞAZIN ANATOMİSİ — SUNUMUN BİLİMSEL ÇEKİRDEĞİ

Bu bölüm sunumun en güçlü kısmı olabilir. Sadece "şu skoru aldık" değil,
**"neden burada durduğunu anladık"**.

### Dört bağımsız yöntem aynı duvara çarptı

| yöntem | AFIB/AFL ikili doğruluk |
|---|---|
| Kaggle en iyi model (500 Hz, base=64, **8.8 M parametre, GPU**) | 0.701 |
| Bizim CNN ensemble (150 Hz, base=32) | 0.760 |
| 37 elle çıkarılmış özellik + GBM | 0.725 |
| Yalnızca bu ikili için eğitilmiş uzman model | 0.744 |

**Kaggle'ın 8.8 milyon parametreli GPU modelinin bizden daha kötü olması**,
bunun bir kapasite sorunu olmadığını gösteriyor.

### Hataların %93.4'ü modelin kararsız olduğu yerde

Geliştirme kümesindeki ~410 ikili hatanın yalnızca **27 tanesi yüksek
güvenli** (p ≥ 0.80). Yani 14:1 oranla hatalar, modelin "emin değilim" dediği
yerlerde.

**Bunun anlamı:** sorun yanlış etiket değil, **girdinin gerçekten belirsiz
olması**. Etiket gürültüsü olsaydı model kendinden emin biçimde yanılırdı
(doğru deseni öğrenir, etiket ona karşı çıkar). Kaba AFIB ile AFL, kardiyologların
da anlaşamadığı gerçek bir klinik gri bölgedir.

### Sinyal işleme açıklaması (jüri için güzel bir detay)

AFIB ile AFL'yi ayıran şey **atriyal dalgadır**: AFL'de ~4-6 Hz'lik düzenli
testere dişi (F dalgası), AFIB'de aynı bantta düzensiz gürültü. Ama bu dalga
QRS'ten **10-20 kat küçüktür** ve 75 bpm'de QRS treninin harmonikleri
(1.25, 2.5, 3.75, **5.0, 6.25** Hz) tam o bandın üzerine düşer.

---

## 6. ELENEN FİKİRLER — PROJENİN EN DEĞERLİ VARLIĞI

**Sunumda bunu "denedik olmadı" diye değil, "her fikri ölçtük ve ölçüme göre
eledik" diye anlat.** Jüri, şansa tutmuş bir 0.86'dan disiplinli bir 0.84'ü
ödüllendirir — çünkü ikincisi tekrarlanabilir.

| # | fikir | ölçüm | karar |
|---|---|---|---|
| 1 | Kapasite artırma (base 32→64→80) | OOF kazancı yok | RED |
| 2 | 250 Hz örnekleme (150 yerine) | kazanç yok | RED |
| 3 | Etiket temizleme (şüpheli kayıtları çıkar) | OOF artmadı | RED |
| 4 | AFIB/AFL uzman modeli | ikili 0.744 (< 0.760) | RED |
| 5 | Sınıf başına eşik kalibrasyonu | OOF +0.003, test −0.0015 | RED |
| 6 | Ölçek TTA | kazanç yok | RED |
| 7 | Daha fazla fold | kazanç yok | RED |
| 8 | **Dengeli kod çözme** (transport dual) | OOF **−0.0097** | RED |
| 9 | **mixup'ta AFIB↔AFL çiftini koruma** | McNemar **p = 0.755** | RED |
| 10 | **QRST iptalli atriyal bant özellikleri** | gerçek veride **+0.0006** (kapı 0.008) | RED |
| 11 | **Inception1D mimarisi** | aynı tahmin oranı **0.9235** | RED |
| 12 | **CNN + Transformer hibrit** | aynı tahmin oranı **0.9259**, kurtarılabilir 58 | RED |

### Ayrıca: 30 → 20 model budaması

Ensemble'dan 10 model çıkarıldı. **750 tahminin hiçbiri değişmedi.** Bu,
"model çeşitliliği" iddiasının bu projede neden çalışmadığının en net kanıtı —
beş ailenin hepsi ResNet'ti ve aynı şeyi öğreniyorlardı.

---

## 7. METODOLOJİ — SUNUMDA ANLATILACAK ASIL ŞEY

Bu proje sıradan bir "model eğittik" projesinden şu araçlarla ayrılıyor:

### 7.1 Seçim yalnızca OOF ile

**Değişmez kural:** mimari, hiperparametre, eşik, ensemble ağırlığı — hepsi
**out-of-fold** skoruyla seçildi. `test_public`'e yalnızca bir deney bittikten
sonra, rapor için, tek sefer bakıldı. Hiçbir seçim ona bakılarak yapılmadı.

### 7.2 Bayes tavanı bilinen sentetik kıyas kümesi

`tools/make_synth_hard.py` — fikirleri gerçek veriyi (ve günleri) harcamadan
elemek için kurulmuş bir test ortamı.

AFIB/AFL ayrımı gizli bir θ değişkenine bağlanıyor: θ=1 saf testere dişi
(klasik AFL), θ=0 saf bant gürültüsü (klasik AFIB). İki sınıfın θ dağılımları
**örtüşüyor** ve örtüşme miktarı ayarlanabiliyor. Örtüşme bilindiği için
**"mükemmel bir model en fazla ne yapabilir" analitik olarak hesaplanabiliyor.**

Bu kümede Bayes tavanı **0.8408**, referans koşu **0.7965** — yani gerçek
projedeki 0.7376 ile aynı rejim, ölçülebilir boşluk var.

### 7.3 Eşleştirilmiş istatistiksel test

`tools/compare_runs.py` — iki koşuyu yan yana koyup "0.7965 vs 0.8033, demek
ki iyi" demek yanlıştır: fark aynı kayıtlar üzerinde ölçüldüğü için bağımsız
değildir.

Doğru test **McNemar**'dır — yalnızca fikrini değiştiren kayıtlara bakar:
düzelen (b01) ve bozulan (b10). Karar için `p < 0.05` isteniyor.

### 7.4 Çeşitlilik kapısı — tek başına skor DEĞİL

`check_diversity.py` — yeni bir model ailesi eklemeden önce sorulan soru
"daha iyi mi" değil, **"farklı mı yanılıyor"**. Basılan asıl sayı
**KURTARILABILIR**: birinin bildiği, diğerinin bilmediği kayıt sayısı —
ensemble'ın kazanabileceği üst sınır.

### 7.5 Sızıntı taraması

`tools/add_external.py` — dış veri eklerken üç kademeli çakışma taraması:
kayıt adı → şekil imzası (kazanç/ofset değişimine duyarsız) → korelasyon
≥ 0.995 (yeniden örneklenmiş/kırpılmış kopyaları yakalar).

**Doğrulama:** `test_public`'ten alınıp yeniden adlandırılmış 12 kaydın
**12'si de** yakalandı, sıfır sızıntı.

### 7.6 Veri kökeni analizi

`tools/data_provenance.py` — kayıt adı önekinden kaynak hastaneyi çözer.
Verinin dört kaynaktan geldiği bu araçla ortaya çıktı.

---

## 8. VERİNİN KÖKENİ — SUNUMDA ANLATILACAK ÖNEMLİ BULGU

Kayıt adlarından çözüldüğünde, TEKNOFEST verisinin **dört ayrı hastaneden**
geldiği görüldü:

| sınıf | MIMIC-IV-ECG | PTB-XL | Chapman-Shaoxing | Ningbo | toplam |
|---|---|---|---|---|---|
| Normal | 333 | 333 | 88 | 246 | 1000 |
| AFIB | 475 | 48 | 477 | **0** | 1000 |
| AFL | 471 | 56 | **22** | 451 | 1000 |
| LBBB | 446 | 445 | 88 | 21 | 1000 |
| RBBB | 333 | 333 | 228 | 106 | 1000 |

**Dikkat çeken yapı:** Ningbo'da hiç AFIB yok; Chapman'da neredeyse hiç AFL
yok. Yani bu iki kaynak, sınıf hakkında neredeyse kesin bilgi taşıyor.

Sinyale hiç bakmayan, yalnızca "bu kayıt hangi hastaneden" diye soran bir
kural AFIB/AFL ikilisinde **0.7295** tutturuyor. Modelin gerçek değeri
**0.7376**.

**BU BİR KANIT DEĞİL** — tesadüf olabilir ve büyük ihtimalle öyledir. Ayırt
edici test `tools/source_breakdown.py` ile yapılıyor: MIMIC-IV-ECG dengeli
(475 AFIB / 471 AFL), yani orada kaynak hiçbir şey söylemiyor. Model gerçekten
sinyalden öğrendiyse orada da ~0.74 tutturur; yalnızca hastane imzasını
öğrendiyse ~0.50'ye düşer.

> ⚠️ **BU TEST HENÜZ KOŞULMADI.** Sonucu bilinmiyor. Sunumda kullanmadan önce
> mutlaka koş: `python tools/source_breakdown.py --cache cache --oof ensemble_oof_prob.npy`
>
> Çıkan "dengeli kaynaklarda doğruluk" sayısı, §7.2 dış doğrulamasında
> beklenecek gerçek değerdir. Sunumda **"geliştirmede 0.84 aldık, kaynak
> dengesi düzeltildiğinde beklentimiz şu"** demek jüri önünde çok güçlü bir
> cümledir — ama önce o sayıyı öğrenmek şart.

---

## 9. MÜHENDİSLİK BULGULARI — SUNUMA RENK KATAR

Bunlar "sistemi gerçekten kurduk ve test ettik" mesajını veren somut örnekler.

| # | bulgu | sonucu ne olurdu |
|---|---|---|
| 1 | **EMA'da ısınma yoktu** → val F1 0.0667 (tek sınıf tahmin ediyordu) | tüm hiperparametre taraması sessizce bozuk olurdu |
| 2 | **ONNX dynamo ihracatçısı** ağırlıkları `.onnx.data` yan dosyasına yazıyordu | sadece `*.onnx` kopyalayan biri bozuk paket alırdı; ayrıca int8 çalışmıyordu |
| 3 | **int8 kabul ölçütü yanlış seviyedeydi** (grafik başına olasılık farkı) | paket 3.8× gereksiz büyük kalırdı |
| 4 | **Demo uygulaması "İlk 100"** alfabetik sıralıyordu → hepsi tek sınıf, macro-F1 0.199 | jüri önünde model bozuk görünürdü |
| 5 | `check_diversity.py` kısmi koşuyu yakalamıyordu | `--only_fold` ile eğitilen koşu `oof_prob.npy` yazmaz; eski dosyayla sessizce yanlış karşılaştırma |
| 6 | Butterworth katsayıları elle kuruldu ve **scipy ile 1e-12'de doğrulandı** | filtre hatası tüm sonuçları görünmez biçimde bozardı |

4 numara özellikle anlatılmaya değer: **stratified örnekleme yapılmadığı için
demo bozuk görünüyordu.** Aynı hata sınıfı sonradan bir test betiğinde tekrar
yakalandı (AUC'ların hepsi tam 0.500 çıkması).

---

## 10. TESLİM ZİNCİRİ DOĞRULAMASI

Sunumda "sistem gerçekten çalışıyor" demenin somut kanıtı:

```
ensemble.py  →  export.py (int8)  →  paketten ham WFDB ile çıkarım
```

- ONNX ile PyTorch arasındaki fark: **0.0000**
- int8 ile float32 arasındaki ensemble skor farkı: **0.0000**
- Paket boyutu: **3.8× küçültme** (int8 sayesinde)
- Paket, hiçbir internet bağlantısı olmadan, PyTorch kurulu olmadan çalışıyor

---

## 11. KALAN GÜNLERİN PLANI (bilgi amaçlı)

Detayı `PLAN_5GUN.md`'de. Özet:

| gün | iş |
|---|---|
| 0 | İndirmeleri başlat, `source_breakdown.py` koş, paketi yedekle |
| 1 | Dış veri ekle, **fold-0 kapısı** → kazanç yoksa DUR |
| 2 | Tam 5-fold (kapı geçtiyse), **sunumu bitir** |
| 3 | McNemar, ensemble, export, paket doğrulama |
| 4 | Dondur, jüri provası, sunum provası |
| 5 | Tampon |

**Kural sıfır:** çalışan 0.8412 paketi hiçbir adımda üzerine yazılmaz. Her
deney ayrı `--tag` ve ayrı cache kullanır.

**Dış veri planı:** PhysioNet/CinC Challenge 2021 havuzundan dengeli ~900/sınıf
(~4.700 kayıt). Eklenen kayıtlar `split="extra"` alır ve **yalnızca eğitim
fold'larına** girer, hiçbir fold'un doğrulamasına girmez — böylece OOF hâlâ
yalnızca yarışma verisinde ölçülür ve önceki tüm deneylerle karşılaştırılabilir
kalır.

---

## 12. SUNUM İÇİN ÖNERİLEN ÇATI

Bu bir öneri, kalıp değil. Süreyi bilmiyorum, ona göre kırpılmalı.

1. **Problem** — 5 sınıflı EKG, macro-F1, CPU-only, ONNX teslim
2. **Sistem** — boru hattı şeması, sıfırdan yazılan WFDB okuyucu ve sinyal
   işleme, tek kaynaklı ön işleme kuralı
3. **Sonuç** — 0.8412, %95 GA [0.8140, 0.8644], sınıf bazında tablo
4. **Darboğazın anatomisi** — tüm hata AFIB/AFL'de; dört bağımsız yöntem aynı
   duvara çarptı; hataların %93.4'ü modelin kararsız olduğu yerde; bunun
   anlamı etiket değil, girdi belirsizliği
5. **Metodoloji** — OOF-only seçim, Bayes tavanlı kıyas kümesi, eşleştirilmiş
   McNemar, çeşitlilik kapısı, sızıntı taraması
6. **Elenen fikirler tablosu** — 12 fikir, her biri bir sayıyla elenmiş
7. **Genelleme** — veri kökeni analizi, kaynak dengesi bulgusu, §7.2 dış
   doğrulaması için dürüst beklenti *(source_breakdown sonucu geldiğinde)*
8. **Canlı demo** — `ecg_demo.py`: bir kayıt seç, tahmini gör; toplu skor koş
9. **Tekrarlanabilirlik** — tek komut, PyTorch yok, internet yok, SHA-256
   sağlamalı manifest, kendi kendini doğrulayan ihracat

### Vurgulanacak üç cümle

- *"Modeli büyütmek çözmüyor: 8.8 milyon parametreli GPU modeli bizden daha
  kötü sonuç veriyor."*
- *"Her fikri bir kapıdan geçirdik; 12 fikri sayıyla eledik ve hepsini kayda
  geçirdik."*
- *"Sistemi PyTorch'suz, internetsiz, tek komutla çalışacak şekilde paketledik
  ve paketin skorunu eğitimdeki skorla birebir doğruladık."*

### Söylenmemesi gerekenler

- ❌ "Kaggle'ı geçtik" — geçilmedi, güven aralığı örtüşüyor
- ❌ "%84 doğrulukla hastalık teşhis ediyoruz" — bu bir tarama/karar destek
  aracı, teşhis aracı değil; klinik iddia kurma
- ❌ Dış doğrulamada 0.84 bekleneceğini ima etme — beklenti daha düşük olabilir,
  `source_breakdown.py` sonucu bunu söyleyecek
- ❌ Doğrulanmamış sayı kullanma — bu belgede ⚠️ işaretli olanlar henüz ölçülmedi

---

## 13. HENÜZ BİLİNMEYENLER — SUNUMDA KESİN KONUŞMA

| soru | durum |
|---|---|
| `source_breakdown.py` sonucu (dengeli kaynaklarda doğruluk) | ⚠️ **koşulmadı** |
| SPH veri setinde kaç AFL var | ⚠️ **ölçülmedi** (`sph_probe.py` bekliyor) |
| Dış veri eklemenin gerçek kazancı | ⚠️ **ölçülmedi** (fold-0 kapısı bekliyor) |
| Final dış doğrulama kümesinin ne olduğu | bilinmiyor, TEKNOFEST açıklamadı |

---

## 14. DOSYA HARİTASI

**Belgeler:**
`GOREV.md` (özgün görev tanımı) · `DENEY_KAYDI.md` (tüm deneylerin kaydı —
sunum için altın değerinde) · `SONUC.md` · `PLAN_5GUN.md` · `VERI_KAYNAKLARI.md`
· `DIS_VERI.md` · `UYGULAMA.md`

**Çekirdek kod:**
`wfdb_lite.py` · `ecg_preprocess.py` · `prep.py` · `model.py` · `train.py` ·
`ensemble.py` · `export.py` · `package_src/predict.py` · `package_src/ecg_demo.py`

**Ölçüm ve kapı araçları:**
`check_diversity.py` · `afib_afl_diag.py` · `resid_probe.py` ·
`tools/compare_runs.py` · `tools/make_synth_hard.py` · `tools/data_provenance.py`
· `tools/source_breakdown.py` · `tools/add_external.py` · `tools/sph_probe.py`

**Depo:** `RRtuna/deneme`, dal `claude/gorev-md-implementation-b2aj7y`.
Depoda **hiçbir yarışma verisi yok** — 42 takipli dosyanın tamamı `.py` ve
`.md`; cache, koşular, paket ve türetilmiş diziler `.gitignore`'da
(taahhütname md. 5 gereği).
