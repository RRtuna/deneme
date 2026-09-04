# QRST-artık özellikleri — uygulama yönergesi

Bu belge tek bir fikri uygular: **AFIB/AFL ayrımı için QRS'i çıkarıp kalan
atriyal sinyali ölçmek.** Toplam 4 adım. İlk 2 adım 1 dakika sürer ve fikrin
senin verinde işe yarayıp yaramadığını **eğitim yapmadan** söyler.

Değişmez kurallar korunuyor: `ecg_preprocess.py`'ye dokunulmuyor, ön işleme
davranışı değişmiyor, `test_public` hiçbir seçime girmiyor.

---

## Dosyalar — nereye koyacaksın

```
ecg_train/
  resid_features.py          <- YENİ   ölçümlerin tek kaynağı
  resid_probe.py             <- YENİ   eğitimsiz karar kapısı
  train.py                   <- DEĞİŞTİ  kanal/özellik sayısını cache'ten okuyor
  export.py                  <- DEĞİŞTİ  ağırlıklardan çıkarım yapıyor
  package_src/predict.py     <- DEĞİŞTİ  çıkarımda 25 sayıyı kendisi üretiyor
  tools/
    test_resid.py            <- YENİ   uyumluluk testi
    make_resid_features.py   <- YENİ   F.npy 37 -> 62
```

**Önce yedek al:**

```
copy train.py train.py.yedek
copy export.py export.py.yedek
copy package_src\predict.py package_src\predict.py.yedek
```

`model.py`, `ensemble.py`, `prep.py`, `ecg_preprocess.py` **değişmedi** —
onlara dokunma.

---

## ADIM 1 — uyumluluk testi (5 saniye)

```
python tools/test_resid.py --cache cache
```

Senin `ecg_preprocess.py`'n benimkinden farklı. Bu betik hangi API'yi
bulduğunu çalışma anında kontrol eder. **17 kontrolün hepsi geçmeli**, son
satır tam olarak `all checks passed` olmalı.

Dikkat edeceğin çıktılar:

| satır | anlamı |
|---|---|
| `kullanilan yol: detect_rpeaks` veya `detect_r` | senin R bulucun kullanılıyor — **iyi** |
| `kullanilan yol: fallback` | seninki tutmadı, betiğin kendi bulucusu devrede. Çalışır ama mevcut 37 özellikle aynı vuruşları kullanmaz. Sorun değil, sadece bil. |
| `4 derivasyon indeksi bulundu` | II, III, aVF, V1 doğru bulunmuş mu |
| `kayit basina maliyet: ~4 ms` | çıkarıma eklenecek yük |
| bölüm 7'deki AUC listesi | **kaba** ön bakış, karar değil |

Bölüm 7'deki AUC'lar 0.5 civarındaysa panik yapma — o sadece 200 kayıtlık
kaba bir bakış. Asıl karar Adım 2'de.

**Kalırsa:** çıktıyı bana at, düzeltirim. Devam etme.

---

## ADIM 2 — karar kapısı (~20 saniye, eğitim YOK)

```
python resid_probe.py --cache cache --oof ensemble_oof_prob.npy --run runs/<ana koşun>
```

`--run` kısmına 5-fold OOF üreten ana koşunun klasörünü yaz (ör. `runs/main_v2`
veya `runs/cv10`). Bu, probun ağınla **aynı fold'ları** kullanmasını sağlar;
vermezsen kendi fold'unu kurar ve kazanç bir miktar iyimser çıkar.

Betik üç bloğu ayrı ayrı raporlar:

- **bant-artık (20 sayı)** — QRST iptali + 2.5–12 Hz spektral ölçümler
- **iletim oranı (5 sayı)** — RR'ler atriyal döngünün tam katı mı
- **ikisi birden (25 sayı)**

Her blok için bakacağın satırlar:

```
    ayni tahmin orani   : 0.82xx   FARKLI bakiyorlar     <- 0.85'in altı iyi
    sadece prob dogru   :   NN     <- yeni bilgi         <- 0'dan büyük olmalı
    w          ikili dog.     macro-F1         fark
    0.5            0.8xxx       0.9xxx      +0.00xx      <- KARAR bu sütunda
```

### Kapı

| en iyi `fark` | ne yapacaksın |
|---|---|
| **> 0.02** | Adım 3'e geç, yeniden eğit |
| **0.008 – 0.02** | sınırda. Eğitim yapmadan harmanı kullanabilirsin (aşağıda) veya Adım 3'e geçebilirsin |
| **< 0.008** | **DUR.** Fikir senin verinde işe yaramıyor. `DENEY_KAYDI.md`'ye yaz ve bırak. 20 saniye kaybettin. |

Çıktının tamamını `DENEY_KAYDI.md`'ye yapıştır — jüriye "denedik, şu sonucu
aldık" diyebilmek için.

### Sınırda çıkarsa: eğitimsiz harman

`resid_probe.py` seçtiği `w` ile ağın AFIB/AFL olasılığını probunkiyle
karıştırıyor. Bunu kalıcı hale getirmek istersen bana söyle, `ensemble.py`'ye
sabitleyen yamayı veririm — **yeniden eğitim gerektirmez**, sadece karar
kuralını değiştirir.

**Uyarı:** `w` OOF üzerinde seçiliyor. `test_public`'e ancak karar verdikten
sonra, tek sefer bakılır (DEĞİŞMEZ KURAL 3).

---

## ADIM 3 — özellikleri cache'e ekle (~1 dakika)

```
python tools/make_resid_features.py --in cache --out cache_f62 --link
```

- `X.npy`'ye **dokunmaz** (`--link` ile sembolik bağlar; Windows'ta izin
  sorunu çıkarsa `--link` olmadan çalıştır, kopyalar)
- `F.npy` 37 → 62 olur
- `y.npy`, `index.csv`, `meta.json` kopyalanır

Ön işleme değişmediği için `prep.py`'yi yeniden çalıştırmana **gerek yok**
(DEĞİŞMEZ KURAL 6 ihlal edilmiyor — `ecg_preprocess.py` aynı).

---

## ADIM 4 — yeniden eğit

Ana koşunda kullandığın komutun **birebir aynısını** kullan, sadece
`--cache` ve `--tag` değişsin:

```
python train.py --cache cache_f62 --tag f62 --preset <mevcut preset> --folds 5 --epochs 40 --patience 99
```

`train.py` artık özellik sayısını `F.npy`'den okuyor. Başlangıçta şu satırı
görmelisin:

```
cache 62 ozellikli (37 + 25 ek olcum)
```

Bu satır çıkmıyorsa yanlış cache'i veriyorsun.

**Tek şey değişsin kuralı:** `--preset`, `--epochs`, `--lr`, `--folds` — hepsi
ana koşunla aynı olmalı. Yoksa kazancın özellikten mi hiperparametreden mi
geldiğini bilemezsin.

### Karşılaştırma

```
python tools/compare_runs.py runs/<ana koşun> runs/f62 --cache cache_f62
```

Eşleştirilmiş McNemar testi yapar. **`p < 0.05` ve `ANLAMLI` yazmıyorsa
kazanç seed gürültüsüdür**, kabul etme.

---

## ADIM 5 — paketle

```
python ensemble.py --cache cache_f62 --members runs/f62 <diğer üyeler>
python export.py --cache cache_f62 --int8
```

`export.py` artık girdi genişliğini ağırlıklardan çıkarıyor, `manifest.json`'a
gerçek özellik sayısını yazıyor ve `resid_features.py`'yi pakete kopyalıyor.
Paketin `predict.py`'si manifest 37'den fazla istediğini görünce 25 sayıyı
çıkarımda kendisi üretir.

`export.py` kendi kendini doğrular: ONNX skoru PyTorch skoruyla
karşılaştırılır, **fark > 0.005 ise teslim etme.**

Son kontrol — paketi ham veriyle koştur:

```
cd package
python predict.py --batch <yol>\test_public.csv --root <veri kökü>
```

Basılan macro-F1, `runs/f62/summary.json`'daki `test_macro_f1` ile aynı
olmalı. Farklıysa eğitim ile çıkarım arasında tutarsızlık var demektir — dur
ve bana söyle.

---

## Karışırsa: geri dönüş

Hiçbir şey üzerine yazılmadı. Eski cache (`cache/`), eski koşular
(`runs/...`) ve eski paket olduğu gibi duruyor. Geri dönmek için:

```
copy train.py.yedek train.py
copy export.py.yedek export.py
copy package_src\predict.py.yedek package_src\predict.py
```

`cache_f62/` klasörünü silebilirsin.

---

## Beklenti — dürüst hali

Sentetik kıyas kümesinde ölçülen: AFIB/AFL 0.7965 → 0.8282, macro-F1
0.9181 → 0.9313 (p = 0.0036).

**Ama o küme benim ürettiğim bir kümeydi** ve atriyal dalgayı bu özelliklerin
tam olarak ölçtüğü biçimde kodluyordu. Gerçek veride kazanç daha küçük
olacak. Benim tahminim **+0.005 ile +0.015 arası** macro-F1, yani
0.8412 → ~0.85. Test SE'n 0.0129 olduğu için bu bir standart hata civarı.

Adım 2 bunu 20 saniyede kesinleştirir. Oradaki sayı benim tahminimden
üstündür.
