# GÖREV: Bu bilgisayarda mümkün olan en iyi EKG sınıflandırma modelini üret

Sen bir makine öğrenmesi araştırma mühendisisin. Bu bir tartışma değil, bir **yürütme
görevi**. Aşağıdaki fazları sırayla uygula, her fazın kapısındaki sayıya bakarak
kendin karar ver, sonunda çalışan bir model paketi ve bir rapor teslim et.

Kullanıcıya her adımda soru sorma. Sadece **FAZ 1'in sonunda bir kez** zaman bütçesini
teyit et, sonra plana sadık kal.

---

## Başarı tanımı

`test_public` (750 kayıt) üzerinde macro-F1'i **0.8378'in üstüne** çıkar ve bunu
`onnxruntime` ile çalışan, PyTorch gerektirmeyen bir pakette teslim et.

Referans noktaları (hepsi aynı `test_public` üzerinde ölçüldü):

| model | macro-F1 | AFIB/AFL ikili iç doğruluk |
|---|---|---|
| Saf NumPy lojistik regresyon | 0.6429 | — |
| Kaggle `best.pt` — tek model, T4 GPU, 500 Hz, base=64, 8.8 M param | 0.8220 | 0.7014 |
| **Mevcut paket** — 10 model, CPU, 150 Hz, base=32 | **0.8378** | **0.7600** |
| Kaggle 25 model ensemble (ağırlıkları elimizde yok) | 0.8572 | — |

Hedef sırası: önce 0.8378'i geç, sonra 0.8572'yi hedefle.

---

## DEĞİŞMEZ KURALLAR

Bunları ihlal eden bir sonuç geçersizdir.

1. **Sadece CPU, sadece bu bilgisayar.** GPU yok, bulut yok, Colab yok.
2. **Teslim ONNX olacak**, kullanıcı makinesinde PyTorch olmayacak.
3. **`test_public` üzerinde HİÇBİR seçim yapılmaz.** Mimari, hiperparametre, eşik,
   ensemble ağırlığı — hepsi OOF (out-of-fold) skoruyla seçilir. `test_public`'e
   yalnızca bir deney bittikten sonra, rapor için, tek sefer bakılır.
4. **`test_public` kayıtları eğitime asla girmez.** Ne fold'a, ne kalibrasyona.
5. **Ön işleme tek kaynaktan:** `ecg_preprocess.py`. Eğitim ve çıkarım aynı fonksiyonu
   çağırır. İkinci bir filtre kodu yazma.
6. **`ecg_preprocess.py`'yi değiştirirsen cache'i yeniden üret** (`python prep.py`),
   yoksa eğitim eski veriyle çalışır ve fark etmezsin.
7. **Her deneyi `DENEY_KAYDI.md`'ye yaz:** komut, süre, OOF, test, karar. İstisnasız.

---

## Ortam

```
Veri:   D:\TUNA_ISPIR\Documents\Claude\Projects\SYZ
Kod:    D:\TUNA_ISPIR\Documents\Claude\Projects\SYZ\ecg_train
```

5000 kayıt, 5 sınıf × tam 1000: **Normal(0) AFIB(1) AFL(2) LBBB(3) RBBB(4)**.
12 derivasyon, 500 Hz, 10 sn, WFDB (`.hea` + `.dat` veya `.mat`).
`train.csv` 3500 · `validation.csv` 750 · `test_public.csv` 750.

**Geliştirme kümesi = train + validation = 4250 kayıt.** Tüm CV bunun üzerinde.

| dosya | işi |
|---|---|
| `ecg_preprocess.py` | ön işlemenin tek kaynağı: filtre, R tepe, 37 özellik, z-skor |
| `wfdb_lite.py` | `.hea`/`.dat`/`.mat` okuyucu, saf numpy |
| `prep.py` | ham → 150 Hz `X.npy` + `F.npy` (37 özellik) + `index.csv` |
| `prep_fs.py` | başka hızda X üretir: `python prep_fs.py 250` |
| `model.py` | `r18` `r34` `r18k11` `wide` `inception` `hybrid` + özellik dalı |
| `train.py` | 5-fold CV, mixup, EMA, cosine, bf16, **fold-atlamalı devam** |
| `ensemble.py` | ağırlık araması + stacking + uzman entegrasyonu |
| `export.py` | ONNX + int8 + manifest + **kendi kendini doğrular** |
| `bench.py` | donanım ölçümü, veri gerekmez |
| `baseline/` | mevcut modelin OOF/test olasılıkları — yeni modeli buna ekleyip ölç |

---

## FAZ 0 — Kurulum ve taban çizgisini doğrula

```bash
set ECG_ROOT=D:\TUNA_ISPIR\Documents\Claude\Projects\SYZ
set ECG_WORK=D:\TUNA_ISPIR\Documents\Claude\Projects\SYZ\ecg_train

pip install numpy scipy pandas scikit-learn
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install onnx onnxruntime

cd %ECG_WORK%
python prep.py
```

`prep.py` çıktısında **`hatali=0`** görmelisin ve split dağılımı 700/150/150 olmalı.
Değilse dur ve nedenini bul — bozuk veriyle eğitim boşa gider.

Sonra mevcut paketin gerçekten çalıştığını doğrula:

```bash
cd ..\ecg_model_package
python predict.py --batch %ECG_ROOT%\test_public.csv --root %ECG_ROOT%
```

**Kapı:** çıkan macro-F1 0.8377 ± 0.001 olmalı. Olmuyorsa ön işleme uyuşmuyordur;
devam etme, önce onu çöz.

---

## FAZ 1 — Donanımı ölç, bütçeyi kes

```bash
cd %ECG_WORK%
python bench.py
```

Bu, her mimarinin bu makinede epoch başına kaç saniye sürdüğünü ve 5-fold × 40 epoch'un
kaç saat edeceğini yazar. **bf16 satırlarına dikkat et:** AVX512-BF16 yoksa bf16
hızlanma getirmez, o zaman `--no_bf16` ile koş.

Referans (bulut makinesi, 2 çekirdek, AMX + AVX512-BF16): r18 14 s/epoch,
wide 21 s/epoch. Bu makinede muhtemelen **2-3 kat yavaş** olacak.

**Şimdi kullanıcıya tek bir soru sor:** "Elimde kaç saat var?" Cevaba göre aşağıdaki
fazlardan kaçını koşacağına karar ver ve planı yaz. Sonra bir daha sorma.

Kaba bütçe rehberi (bench.py çıktısıyla ölçekle):

| bütçe | kapsam |
|---|---|
| ~4 saat | FAZ 2 + FAZ 4 (kapasite → tam 5-fold) + FAZ 7 |
| ~8 saat | + FAZ 3 (çözünürlük) + FAZ 6 (tarif) |
| ~16 saat | + FAZ 5 (mimari çeşitliliği), 10-fold |

---

## FAZ 2 — Kapasite taraması *(en yüksek öncelik, kanıt var)*

**Neden:** tek değişkenli ölçüm yapıldı — base 32 → fold 0 test **0.8288**,
base 48 → **0.8468**. Sadece genişlikten **+0.018**. Bu en güçlü sinyal.

`model.py` içindeki `PRESETS` sözlüğüne ekle:

```python
"w64": dict(base=64, blocks=(2,2,2,2), mults=(1,2,4,8), k=7,  stem_k=15),
"w80": dict(base=80, blocks=(2,2,2,2), mults=(1,2,4,8), k=7,  stem_k=15),
```

Sadece fold 0'da koş (5-fold maliyetinin 1/5'i):

```bash
python train.py --preset w64 --tag cap_b64 --only_fold 0 --epochs 40 --patience 99
python train.py --preset w64 --tag cap_b64_d3 --only_fold 0 --epochs 40 --patience 99 --dropout 0.3
python train.py --preset w80 --tag cap_b80 --only_fold 0 --epochs 40 --patience 99 --dropout 0.3
```

**Kapı:** en iyi fold 0 test skoru **> 0.855** ise kazanan yapılandırmayı FAZ 4'e taşı.
Hiçbiri 0.8468'i **0.01'den fazla** geçmiyorsa base=48'de kal ve FAZ 3'e geç.

**Dikkat:** genişlik arttıkça aşırı öğrenme artar. Val eğrisi erken tepe yapıp
düşüyorsa `--dropout 0.3` ve `--wd 3e-4` dene — augmentasyonu artırma, o zaten agresif.

---

## FAZ 3 — Zaman çözünürlüğü *(test edilmedi, yüksek öncelik)*

**Neden:** Kaggle 500 Hz kullandı. 150 Hz'in AFL'in testere dişi F dalgalarının ince
morfolojisini kırpıp kırpmadığı bilinmiyor — bu deney bulut makinesi çöktüğü için
hiç tamamlanamadı.

```bash
python prep_fs.py 250
python train.py --preset <FAZ2 kazanani> --tag res_250 --cache %ECG_WORK%\cache_250 --only_fold 0 --epochs 40 --patience 99
```

`prep_fs.py` sadece `X.npy` üretir; 37 özellik zaten 500 Hz'den hesaplandığı için
`F.npy` aynen kullanılır — yani karşılaştırma tek değişkenli ve temiz.

**Kapı:** 250 Hz fold 0, 150 Hz karşılığını **0.01'den fazla** geçiyorsa çözünürlük
gerçek kaldıraçtır → `python prep_fs.py 500` ile 500 Hz'i de dene, bütçeyi buraya
yatır. Fark bundan azsa 150 Hz'de kal (3 kat ucuz) ve bir daha dönme.

**Bellek:** 500 Hz'de `X.npy` 600 MB. RAM 8 GB'ın altındaysa `train.py` içindeki
`X = np.ascontiguousarray(X)` satırını kaldır, `mmap_mode="r"` ile bırak.

---

## FAZ 4 — Kazananı tam 5-fold koş

FAZ 2 ve 3'ün kazananıyla:

```bash
python train.py --preset <kazanan> --tag main_v2 --epochs 40 --patience 99 [--cache <kazanan cache>]
python ensemble.py
```

`ensemble.py` yeni modeli `baseline/` içindeki mevcut modellerle birleştirir ve
düz ortalama / ağırlıklı / stacking'i OOF'a göre karşılaştırır.

**Kapı:** ensemble OOF **0.8345'in üstünde** mi? Değilse yeni model katkı vermiyordur —
`DENEY_KAYDI.md`'ye yaz ve neden olduğunu düşün, körlemesine bir sonraki faza geçme.

**Bu noktada elinde teslim edilebilir bir sonuç var.** FAZ 7'yi şimdi koş, paketi
kaydet, sonra kalan bütçeyle FAZ 5-6'ya devam et. Yarım kalırsan elinde bir şey olsun.

---

## FAZ 5 — Gerçek mimari çeşitliliği

**Neden:** iki ResNet varyantını ensemble'lamak kazanç vermedi (0.8337 ve 0.8290 →
düz ortalama 0.8317). Çünkü aynı hataları yapıyorlar. Kaggle'ın +0.017'si
ResNet + Inception + Transformer karışımından geldi.

`model.py` içinde `inception` (paralel 11/21/41 kerneller) ve `hybrid` (CNN kodlayıcı +
2 katmanlı Transformer) hazır ama **hiç eğitilmedi.**

```bash
python train.py --preset inception --tag div_inc --only_fold 0 --epochs 30 --patience 99 --lr 0.002
python train.py --preset hybrid    --tag div_hyb --only_fold 0 --epochs 30 --patience 99 --lr 0.001
```

**Kapı — tek başına skor DEĞİL.** Ölçüt: hata korelasyonu düşük mü?

```python
import numpy as np
a = np.load("baseline/r18_feat/oof_prob.npy").argmax(1)
b = np.load("runs/div_inc/oof_prob.npy").argmax(1)
print("ayni tahmini verme orani:", (a == b).mean())
```

**0.85'in altındaysa** bu model gerçekten farklı bakıyor → tam 5-fold koş, ensemble'a ekle.
Üstündeyse ResNet'in kopyası demektir, atla.

Tek başına 0.82 alıp ensemble'a +0.01 katan model, tek başına 0.84 alıp hiçbir şey
katmayandan iyidir.

---

## FAZ 6 — Tarif rafinesi *(ucuz, birikimli)*

Bütçe kalırsa, sırayla. Her biri küçük, toplamı anlamlı:

1. **CV sonrası tüm veriyle yeniden eğit.** Fold'ların medyan en-iyi-epoch sayısını al,
   4250 kaydın tamamıyla o kadar epoch eğit, ensemble'a ek model olarak koy.
   Maliyet 1/5, beklenen +0.005.
2. **10-fold.** Her model %90 veriyle eğitilir (%80 yerine). Maliyet 2×, beklenen +0.005.
3. **Farklı seed'lerle tekrar.** Aynı yapılandırma, `--seed 99`, `--seed 7`.
   En ucuz çeşitlilik kaynağı.
4. **TTA genişletme.** `train.py:evaluate()` şu an 3 kaydırma yapıyor; ölçek ve
   derivasyon-alt-kümesi TTA'sı ekle.
5. **Augmentasyon ablasyonu.** Mevcut set agresif: kaydırma, genel ölçek,
   derivasyon-ölçek, gürültü, taban kayması, derivasyon düşürme %20, frekans maskesi %25,
   mixup 0.3. `Aug.__call__` içinde tek tek kapatıp fold 0'da ölç — bazıları zarar
   veriyor olabilir.

---

## FAZ 2.5 — AFIB/AFL etiket teşhisi *(ucuz, yüksek bilgi değeri — FAZ 2 ile paralel koş)*

Bu model eğitmez, mevcut OOF olasılıklarını kullanır, dakikalar sürer. **Mutlaka yap.**

**Durum:** kalan hatanın neredeyse tamamı AFIB↔AFL'de. Model "bu ikisinden biri"
demeyi %91.8 doğru yapıyor, ikilinin **içinde** %76'da kalıyor. İkili doğruluk %90
olsa macro-F1 ≈ 0.90 olurdu. Yani tüm oyun burada.

**Dört bağımsız yöntem aynı duvara çarptı:**

| yöntem | AFIB/AFL ikili doğruluk |
|---|---|
| Kaggle best.pt (500 Hz, base=64, 8.8 M param) | 0.701 |
| Bizim CNN ensemble (150 Hz, base=32) | 0.760 |
| 37 elle çıkarılmış özellik + GBM | 0.725 |
| Sadece bu ikili için eğitilmiş uzman model | 0.744 |

Kaggle'ın büyük GPU modelinin de çözememesi, bunun **kapasite değil veri/etiket sorunu**
olduğunu güçlü biçimde gösteriyor. Bunu kesinleştir:

1. `baseline/r18_feat/oof_prob.npy` ile: gerçek etiketi AFL olup yüksek güvenle
   (>0.80) AFIB denen kayıtları listele, ve tersini.
2. Bunlardan 20-30 tanesinin EKG'sini çiz (II, III, aVF, V1 — flutter dalgaları
   burada görünür). Gerçekten testere dişi F dalgası var mı?
3. Şüpheli kayıtları **eğitimden** çıkarıp (test_public'ten ASLA) yeniden eğit,
   OOF'un ne olduğuna bak.

**Karar:** OOF belirgin artıyorsa yol budur, bütçeyi buraya kaydır ve bulguyu rapora
yaz — bilimsel olarak değerli bir sonuç. Artmıyorsa sinyal tabanlı bir tavana
çarpılmış demektir; AFIB/AFL'yi bırak, FAZ 2-3-5 ile genel skoru yükselt.

**Uyarı:** temizleme yaparken fold sızıntısına dikkat — bir kaydın şüpheli olup
olmadığına, o kaydı eğitimde görmemiş fold'un tahminiyle karar ver.

---

## FAZ 7 — Paketle, doğrula, teslim et

```bash
python ensemble.py      # nihai birleştirme kuralini OOF'a gore secer
python export.py        # ONNX + int8 + manifest + kendi kendini dogrular
```

`export.py` paketi `test_public` üzerinde koşturup ONNX skorunu PyTorch skoruyla
karşılaştırır. **Fark > 0.005 ise bir şey bozuktur — teslim etme, önce onu bul.**

Sonra son kontrol:

```bash
cd package
python predict.py --batch %ECG_ROOT%\test_public.csv --root %ECG_ROOT%
python predict.py %ECG_ROOT%\Normal\NORM_000508\48090046.hea
```

Paket şunları içermeli: `models/*.onnx` (int8), `ecg_preprocess.py` (eğitimdekinin
birebir kopyası), `wfdb_lite.py`, `predict.py`, `manifest.json` (ağırlıklar +
doğrulama skorları), `preprocess.json`, `README.md`.

---

## FAZ 8 — Rapor

`SONUC.md` yaz, şunlar olsun:

- **Karşılaştırma tablosu:** eski paket 0.8378 → yeni paket X. Ne değişti, neden.
- **Sınıf başına F1** ve **karışıklık matrisi**.
- **AFIB/AFL ikili iç doğruluğu** — asıl ölçüt bu, ayrıca raporla.
- **Deney tablosu:** her deney, değişen tek şey, fold 0 skoru, 5-fold OOF, 5-fold test,
  karar. `DENEY_KAYDI.md`'den derle.
- **İşe yaramayanlar** — nedeniyle birlikte. Bunu atlama.
- Seçimlerin **OOF ile** yapıldığının açık beyanı.

---

## Bitti tanımı

- [ ] `predict.py --batch` çalışıyor, skor `manifest.json` ile ±0.001 eşleşiyor
- [ ] PyTorch kurulu olmayan ortamda çalışıyor
- [ ] test macro-F1 > 0.8378
- [ ] `DENEY_KAYDI.md` her denemeyi komutuyla birlikte içeriyor
- [ ] `SONUC.md` yazıldı, işe yaramayanlar da orada
- [ ] Hiçbir seçim `test_public`'e bakılarak yapılmadı

---

## Tuzaklar

1. **`test_public`'e bakarak ayar yapmak.** En yıkıcı hata. Her seçim OOF ile.
2. **Tek fold sonucuna güvenmek.** Fold'lar arası oynaklık ±0.015. Karar için
   **≥0.01 fark** iste, altındakine gürültü de ve geç.
3. **`ecg_preprocess.py`'yi değiştirip `prep.py`'yi yeniden koşmamak.**
4. **Fold sızıntısı.** Özellik ölçekleyici sadece geliştirme kümesinden hesaplanır —
   `train.py` doğru yapıyor, bozma.
5. **Erken durdurmayla cosine'i kesmek.** `--patience 99` kullan, LR tam sönümlensin.
   60 epoch + erken durdurma denendi: model ep24'te tepe yapıp düştü, LR hiç
   sönümlenmedi. 40 epoch tam sönümlü daha iyi.
6. **Aşırı öğrenmeyi augmentasyonla çözmeye çalışmak.** Önce dropout ve weight decay.
7. **Yarım kalan koşuyu sonuç saymak.** `summary.json` yoksa sonuç yoktur.
8. **Uzun koşuyu kurtarmayı unutmak.** `train.py` fold-atlamalı devam eder —
   bilgisayar kapanırsa aynı komutu tekrar çalıştır, biten fold'ları geçer.

---

## Zaman kaybetme — bunlar denendi ve elendi

Sayılar gerçek, hepsi koşuldu:

| yaklaşım | sonuç |
|---|---|
| AFIB/AFL için ayrı uzman model | ensemble'a katkı **+0.001** (α=0.05) — hataları ana modelle örtüşüyor |
| Sınıf başına eşik/bias kalibrasyonu | OOF +0.003, **test −0.0015** — eşik sorunu değil |
| İki ResNet varyantını ensemble'lamak | 0.8337 + 0.8290 → **0.8317** — çeşitlilik yok |
| Stacking vs ağırlıklı ortalama | OOF 0.8356 vs 0.8345 — fark gürültü içinde |
| 60 epoch + erken durdurma | ep24'te tepe, sonra düşüş — 40 epoch tam cosine daha iyi |

---

## Son söz

Bir deney işe yaramadığında bunu açıkça yaz. Sayıyı süslemek TEKNOFEST jürisinde
sorulacak ilk şeydir. "Denedik, işe yaramadı, nedeni şu" cümlesi "0.89 aldık"
cümlesinden daha savunulabilirdir — çünkü doğrudur.

Mevcut paket zaten Kaggle'ın tek GPU modelini geçiyor (0.8378 vs 0.8220), üstelik
20 kat hafif ve dizüstünde çevrimdışı çalışıyor. Bunu koru, üstüne koy.
