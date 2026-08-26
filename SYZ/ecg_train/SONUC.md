# SONUÇ

## Özet — önce en önemli cümle

**Bu ortamda `test_public` üzerinde macro-F1 ölçülemedi ve 0.8378 geçilemedi,
çünkü SYZ veri kümesi bu makinede yok.** Teslim edilen şey çalışan bir model
değil, **çalıştığı uçtan uca doğrulanmış bir boru hattı**.

Sayıyı süslemek yerine durumu olduğu gibi yazıyorum, çünkü TEKNOFEST jürisinde
sorulacak ilk şey budur.

---

## Neden model eğitilemedi

GOREV.md şu ortamı varsayıyordu:

```
Veri:  D:\TUNA_ISPIR\Documents\Claude\Projects\SYZ     5000 kayıt
Kod:   D:\TUNA_ISPIR\Documents\Claude\Projects\SYZ\ecg_train
       ecg_preprocess.py, prep.py, model.py, train.py, ensemble.py,
       export.py, bench.py, baseline/ (mevcut 10 modelin OOF olasılıkları)
```

Bu oturum bulut tabanlı, geçici bir konteynerde koştu. Somut durum:

| kontrol | bulgu |
|---|---|
| `RRtuna/deneme` deposu | **tamamen boş**, hiç commit yok |
| `RRtuna/resnet1d`, `RRtuna/resnet` | **ikisi de boş**, hiç commit yok |
| Diskte `SYZ/` veya `ecg_train/` | **yok** |
| `baseline/` OOF olasılıkları | **yok** |
| Mevcut `ecg_model_package` | **yok** |
| Donanım | 4 çekirdek Xeon 2.80 GHz, 15 GB RAM, GPU yok (kural 1 ile uyumlu) |
| `physionet.org` | **403** — kurum ağ politikası engelliyor |
| `download.pytorch.org` | **403** — aynı politika |
| PyPI | erişilebilir |

Yani ne senin veri kümen vardı, ne de yerine koyabileceğim herkese açık bir
12-derivasyonlu EKG kümesi indirilebiliyordu. `test_public`'in 750 kaydı
olmadan "macro-F1 > 0.8378" ölçülemez. Bu sayıyı üretmenin tek yolu uydurmak
olurdu.

**Ne yapıldı yerine:** GOREV.md'nin tarif ettiği bütün boru hattı sıfırdan
yazıldı ve sentetik EKG ile uçtan uca koşturularak her aşamanın çalıştığı
kanıtlandı. Sen gerçek veriyle `PLAN_8SAAT.md`'yi koştuğunda sayılar gelecek.

---

## Ne teslim edildi

| dosya | durum | kanıt |
|---|---|---|
| `ecg_preprocess.py` | ✅ çalışıyor | filtreler scipy ile 1e-12 içinde aynı |
| `wfdb_lite.py` | ✅ çalışıyor | format 16/212/`.mat` gidiş-dönüş 1e-7 |
| `prep.py` | ✅ çalışıyor | 1000 kayıt, `hatali=0`, 194 kayıt/s |
| `prep_fs.py` | ✅ yazıldı | `F.npy` yeniden kullanımı doğrulandı |
| `model.py` | ✅ çalışıyor | 8 preset, w64 = 9.07 M param |
| `train.py` | ✅ çalışıyor | 5-fold OOF 0.9977, devam çalışıyor |
| `ensemble.py` | ✅ çalışıyor | flat/weighted/stacked karşılaştırması |
| `export.py` | ✅ çalışıyor | ONNX = PyTorch, fark **0.0000** |
| `predict.py` | ✅ çalışıyor | **PyTorch engelliyken** koştu |
| `bench.py` | ✅ çalışıyor | 8 preset ölçüldü |
| `afib_afl_diag.py` | ✅ çalışıyor | tavan taraması + SVG çizim |
| `tools/test_preprocess.py` | ✅ 27/27 PASS | — |
| **eğitilmiş ağırlıklar** | ❌ **yok** | gerçek veri yok |

### "Bitti tanımı" kontrol listesi

| madde | durum |
|---|---|
| `predict.py --batch` çalışıyor, skor `manifest.json` ile ±0.001 eşleşiyor | ✅ fark 0.0000 (sentetik) |
| PyTorch kurulu olmayan ortamda çalışıyor | ✅ `import torch` engellenerek doğrulandı |
| test macro-F1 > 0.8378 | ❌ **ölçülemedi — veri yok** |
| `DENEY_KAYDI.md` her denemeyi komutuyla içeriyor | ✅ |
| `SONUC.md` yazıldı, işe yaramayanlar da orada | ✅ bu dosya |
| Hiçbir seçim `test_public`'e bakılarak yapılmadı | ✅ kodla zorlanıyor |

---

## Sentetik doğrulama — ne kanıtlar, ne kanıtlamaz

Boru hattını koşturmak için `tools/make_synth.py` ile 1000 kayıtlık 12
derivasyonlu, 500 Hz, 10 saniyelik bir WFDB kümesi üretildi. Sınıflar onları
gerçekten ayıran özelliklerle çizildi: Normal'de düzenli RR ve her QRS'ten önce
P dalgası; AFIB'de düzensiz-düzensiz RR ve P yok; AFL'de düzenli RR ve
diyastolde ~5 Hz testere dişi; LBBB'de geniş QRS, V5/V6'da geniş monofazik R;
RBBB'de V1/V2'de rSR'.

**Sonuç: sentetik test macro-F1 0.9933.**

Bu sayı **model kalitesi hakkında hiçbir şey söylemez.** Sentetik veri gerçek
EKG'den çok daha ayrılabilir: `hard` modunda AFIB/AFL örtüşmesi (değişken
bloklu flutter, kaba AFIB, derivasyon düşmesi, artefakt) eklendikten sonra bile
ikili doğruluk 0.97'de kaldı — gerçek veride bu sayı **0.76**.

Sahte veriyi daha da zorlaştırmak tiyatro olurdu; zorluğu uydurmak model
hakkında bilgi üretmez. Sentetik küme **boru hattı testi** olarak bırakıldı.

Kanıtladığı şey şu: `prep → train → ensemble → export → predict` zincirinin her
halkası çalışıyor, aralarındaki veri sözleşmeleri tutuyor, ONNX PyTorch'la
aynı sayıyı veriyor ve paket PyTorch'suz koşuyor.

---

## İşe yaramayanlar ve bulunan hatalar

GOREV'in son sözü: "Bir deney işe yaramadığında bunu açıkça yaz." Bu bölüm o
yüzden var.

### Denendi, işe yaramadı

**1. Sentetik veriyi gerçekçi zorlukta yapmak — başarısız.**
İki tur denendi. `easy` modda 37 özellik + lojistik regresyon **1.0000** verdi.
`hard` modda AFIB/AFL örtüşmesi, artefakt, derivasyon düşmesi eklendi; sonuç
**0.9867**. Gerçek verideki 0.8378'lik zorluk seviyesine yaklaşılamadı.
**Neden:** gerçek AFIB/AFL zorluğu etiket belirsizliğinden ve gerçek
fizyolojik sürekliliktendedir; parametreli bir üreteçte bunu taklit etmek,
üreteçteki gizli değişkeni modele sızdırmak demektir. Vazgeçildi ve sınırı
açıkça yazıldı.

**2. Yeni dynamo ONNX dışa aktarıcısı — kullanılamadı.**
İki bağımsız şekilde bozuktu: (a) ağırlıkları `*.onnx.data` yan dosyalarına
yazıyor, (b) `quantize_dynamic` onun grafiğinde şekil çıkarımı hatası veriyor.
TorchScript dışa aktarıcısına (`dynamo=False`) dönüldü, ikisi de çözüldü.

**3. int8'i olasılık farkıyla yargılamak — yanlış ölçüttü.**
Grafik başına 2e-3 eşiğiyle int8 modellerin **hepsi** reddediliyordu
(sapmalar 5.6e-3 – 1.8e-2). Ama kuantalama olasılıkları tasarımı gereği ~%1
oynatır; önemli olan skorun değişip değişmediğidir. Ölçüt skora bağlandı,
int8 kabul edildi, paket 66.9 MB → **18.1 MB**.

### Bulunan ve düzeltilen 6 hata

| # | hata | düzeltilmeseydi |
|---|---|---|
| 1 | EMA ısınması yok — 54 adım sonra ağırlıkların %95'i hâlâ rastgele başlangıç | kısa tek-fold taramaları sessizce çöp üretirdi; FAZ 2'nin bütün kapasite kapısı buna dayanıyor |
| 2 | ONNX ağırlıkları yan dosyada, sha256 sadece grafiği kapsıyor | `models/*.onnx` kopyalayan biri bozuk paket alırdı |
| 3 | int8 dynamo grafiğinde çalışmıyor | teslim int8 olamazdı — GOREV şartı |
| 4 | int8 kabul ölçütü yanlış | paket 3.7x gereksiz büyük kalırdı |
| 5 | `ensemble_oof_prob.npy` yanlış uzunlukta (850 yerine 1000 olmalı) | FAZ 2.5 teşhisi hiç koşamazdı |
| 6 | `LogisticRegression(multi_class=)` sklearn 1.9'da kaldırılmış | stacking hiç koşamazdı |

Hata 1 en tehlikeliydi: 40 epoch'luk gerçek koşuda kendini göstermezdi
(`0.999^4280 ≈ 0.014`), sadece kısa taramalarda. FAZ 2'nin tek-fold kapasite
taraması tam olarak öyle bir koşu — yani bu hata fark edilmeseydi, bütün
genişlik kararı gürültü üzerine kurulurdu.

---

## GOREV.md'de bulunan çelişki

**Kural 3** diyor ki: *"`test_public` üzerinde HİÇBİR seçim yapılmaz. Mimari,
hiperparametre, eşik, ensemble ağırlığı — hepsi OOF skoruyla seçilir."*

**FAZ 2'nin kapısı** diyor ki: *"en iyi fold 0 **test** skoru > 0.855 ise
kazanan yapılandırmayı FAZ 4'e taşı."*

İkisi aynı anda tutulamaz. FAZ 2'nin kapısı, tanımı gereği, `test_public`
skoruna bakarak mimari seçmektir.

**Alınan karar:** OOF kullanıldı. `train.py` her tek-fold koşusunda iki sayıyı
da basıyor ve hangisiyle karar verileceğini açıkça yazıyor:

```
val (OOF) macro-F1  : 0.xxxx   <- kararlari BUNUNLA ver
test_public macro-F1: 0.xxxx   (sadece rapor icin)
```

**Gerekçe:** fold'lar arası oynaklık ±0.015. 750 kayıtlık bir test kümesinde
fold 0 test skoruna göre genişlik seçmek, gürültüye uyum sağlamaktır ve
seçilen genişliğin gerçekten daha iyi olduğunu göstermez. Jüriye
savunulabilir olan OOF'tur.

Bu, GOREV'in kendi 1. tuzağıyla da tutarlı: *"`test_public`'e bakarak ayar
yapmak. En yıkıcı hata."*

---

## `test_public` sızıntısına karşı alınan kod düzeyi önlemler

Kural 3 ve 4 yorum satırı olarak değil, kodun yapısı olarak uygulandı:

- `ensemble.choose_rule(oof_mats, y_dev, ...)` — **test dizilerini parametre
  olarak hiç almaz.** Kural seçimi bittikten sonra `apply_rule` test matrisine
  uygulanır. Sızıntı için fonksiyon imzasını değiştirmek gerekir.
- `train.py` özellik ölçekleyiciyi **fold'un eğitim bölümünden** hesaplar,
  geliştirme kümesinin tamamından değil — fold içi sızıntı da kapalı.
- `afib_afl_diag.py` `test_public` satırlarını baştan dışarıda bırakır; hangi
  kaydın şüpheli olduğuna, o kaydı eğitimde görmemiş fold'un tahminiyle karar
  verilir (OOF'un tanımı gereği).
- `train.py --exclude` **yalnızca eğitim** bölümünden kayıt çıkarır; val ve
  test fold'ları dokunulmaz, yani öncesi/sonrası OOF karşılaştırması dürüst.
- `ensemble.py` stacking'i **iç çapraz doğrulamayla** ölçer. OOF matrisine
  stacker uydurup aynı matriste skorlamak, tam olarak stacking'in kazandığı
  kadar iyimserdir — "stacking kazanıyor" sonucunun tekrarlanamamasının
  sebebi budur. (GOREV'in kendi tablosu zaten OOF 0.8356 vs 0.8345 farkının
  gürültü içinde olduğunu söylüyor.)
- Kural seçiminde **basitlik tercihi**: `weighted`, `flat`'i 0.002'den az
  geçiyorsa `flat` seçilir; `stacked`, diğerlerini 0.005'ten az geçiyorsa
  atılır. Gürültü kadar bir OOF farkı için ikinci bir model taşımaya değmez.

---

## Ölçülen donanım bütçesi

4 çekirdek Xeon @2.80 GHz, AVX512F var, **AVX512-BF16 ve AMX yok**.
4250 geliştirme kaydı, batch 32, 150 Hz giriş, 40 epoch:

| preset | param | s/epoch | tam 5-fold |
|---|---|---|---|
| r18 | 2.32 M | 22.5 | 1.25 sa |
| inception | 1.00 M | 27.9 | 1.55 sa |
| hybrid | 1.25 M | 34.0 | 1.89 sa |
| wide (b48) | 5.14 M | 41.3 | 2.29 sa |
| r34 | 4.33 M | 42.8 | 2.38 sa |
| **w64** | 9.07 M | 62.6 | **3.48 sa** |
| w80 | 14.11 M | 93.2 | 5.18 sa |

GOREV'in referansı (bulut, AMX + AVX512-BF16) r18 için 14 s/epoch diyordu;
bu makine 22.5 s/epoch, yani **1.6x yavaş** — GOREV'in "2-3 kat" tahmini
doğru mertebede.

**bf16 uyarısı:** bu CPU'da AVX512-BF16 ve AMX yok, bf16 autocast
hızlandırmaz, **yavaşlatır**. `bench.py` bunu saptayıp uyarıyor, `train.py`
kendiliğinden kapatıyor.

`w64` = 9.07 M parametre, GOREV'in bahsettiği Kaggle `best.pt`'nin
(base=64, 8.8 M) çok yakınında — genişlik ölçeklendirmesinin karşılaştırılabilir
olduğuna dair bağımsız bir doğrulama.

---

## AFIB/AFL — tavan taraması aracı

GOREV'in "ikili doğruluk %90 olsa macro-F1 ≈ 0.90 olurdu" iddiası artık bir
his değil, ölçülen bir sayı. `afib_afl_diag.py` ikili dışındaki her tahmini
olduğu gibi bırakıp yalnızca ikili içi kararı değiştirerek tarama yapar:

```
tavan taramasi -- ikili dogruluk su olsaydi macro-F1 ne olurdu:
  0.70 -> ...   0.80 -> ...   0.90 -> ...   1.00 -> ...
```

Sentetik veride mekanizmanın çalıştığı doğrulandı (0.70 → 0.90 arasında
**+0.079**). Gerçek `baseline/r18_feat/oof_prob.npy` ile koştuğunda, bütçeyi
AFIB/AFL'ye kaydırmanın değip değmeyeceğini bu tablo söyleyecek.

Araç ayrıca yüksek güvenli ters tahminleri listeler ve şüpheli kayıtların
II/III/aVF/V1 derivasyonlarını **SVG** olarak çizer (hiçbir çizim kütüphanesi
gerekmez). Dört bağımsız yöntemin aynı duvara çarpması (Kaggle 0.701, CNN
ensemble 0.760, GBM 0.725, uzman model 0.744) bunun kapasite değil veri/etiket
sorunu olduğunu güçlü biçimde gösteriyor; bu araç o hipotezi test etmek için.

---

## Sıradaki adım

1. Bu depoyu klonla, `SYZ/ecg_train/` içeriğini kendi `ecg_train` klasörüne koy.
2. `python tools/test_preprocess.py` → `all checks passed` görmelisin.
3. `python prep.py` → **`hatali=0`** ve **3500/750/750**.
4. `python bench.py --json bench.json` → kendi bütçeni ölç.
5. `PLAN_8SAAT.md`'yi sırayla koş.
6. Her deneyi `DENEY_KAYDI.md`'nin altındaki boş tabloya yaz.
7. Bu dosyayı gerçek sayılarla güncelle.

Mevcut paketin (0.8378) Kaggle'ın tek GPU modelini (0.8220) zaten geçiyor,
üstelik 20 kat hafif ve dizüstünde çevrimdışı çalışıyor. **Onu silme.**
`baseline/` klasörüne OOF olasılıklarını koy; `ensemble.py` yeni modeli onunla
birleştirip katkı verip vermediğini OOF'la söyleyecek.
