# 8 SAATLİK PLAN — kendi makinende koşacağın sıra

FAZ 1'in tek sorusuna verdiğin cevap: **~8 saat.** GOREV'in orta bütçe kapsamı
FAZ 2 + FAZ 3 + FAZ 4 + FAZ 6 + FAZ 7-8. Bu dosya onu somut komutlara çeviriyor.

FAZ 2.5 (AFIB/AFL teşhisi) bütçe dışı sayılıyor — model eğitmiyor, dakikalar
sürüyor, GOREV "mutlaka yap" diyor. Plana dahil.

---

## ÖNCE: bütçeni kendi makinende ölç

Aşağıdaki saatler **bu bulut makinesinde** ölçüldü (4 çekirdek Xeon 2.80 GHz,
AVX512F, BF16 yok). Senin makinen farklı olacak. İlk iş:

```bat
cd %ECG_WORK%
python bench.py --dev 4250 --epochs 40 --json bench.json
```

**Ölçek katsayını hesapla:** çıkan `r18` s/epoch değerini 22.5'e böl.

- Katsayı **< 1** → makinen daha hızlı, plan rahat sığar, FAZ 5'i de ekle.
- Katsayı **1-1.5** → plan olduğu gibi geçerli.
- Katsayı **> 1.5** → aşağıdaki "kısaltma" notlarını uygula.

Aynı komut AVX512-BF16/AMX olup olmadığını da yazar. **Yoksa bütün eğitimlere
`--no_bf16` ekle** (`train.py` zaten kendiliğinden kapatıyor, ama açıkça
yazmak koşu kaydını netleştirir).

### Bu makinede ölçülen süreler (referans)

| preset | param | s/epoch | tek fold × 40 ep | 5-fold × 40 ep |
|---|---|---|---|---|
| r18 | 2.32 M | 22.5 | 0.25 sa | 1.25 sa |
| inception | 1.00 M | 27.9 | 0.31 sa | 1.55 sa |
| hybrid | 1.25 M | 34.0 | 0.38 sa | 1.89 sa |
| wide (b48) | 5.14 M | 41.3 | 0.46 sa | 2.29 sa |
| **w64** | 9.07 M | 62.6 | **0.70 sa** | **3.48 sa** |
| **w80** | 14.11 M | 93.2 | **1.04 sa** | 5.18 sa |

250 Hz'e geçince giriş 1500 → 2500 olur, süre kabaca **1.6x** artar.
Yeniden ölçmek için: `python bench.py --length 2500`.

---

## Saat 0:00 — FAZ 0: kurulum ve taban çizgisi (~20 dk)

```bat
set ECG_ROOT=D:\TUNA_ISPIR\Documents\Claude\Projects\SYZ
set ECG_WORK=%ECG_ROOT%\ecg_train
cd %ECG_WORK%

pip install numpy scipy pandas scikit-learn
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install onnx onnxruntime onnxscript

python tools/test_preprocess.py
python prep.py
```

**Kapı 1:** `tools/test_preprocess.py` → `all checks passed`. Geçmezse dur;
ön işleme bozuksa geri kalan her şey boşa gider.

**Kapı 2:** `prep.py` → **`hatali=0`** ve dağılım **3500 / 750 / 750**.
Sınıf başına 700/150/150 olmalı. Değilse dur ve nedenini bul.

> `prep.py` CSV sütun adlarını varsaymaz; diskteki kayıtlarla eşleşme oranına
> bakarak kayıt ve etiket sütununu kendisi bulur. Yine de çıktıdaki
> `split_info` bloğunu `cache/meta.json` içinde bir kez gözden geçir — yanlış
> sütun seçtiyse orada görürsün.

Mevcut paketin hâlâ 0.8378 verdiğini doğrula:

```bat
cd ..\ecg_model_package
python predict.py --batch %ECG_ROOT%\test_public.csv --root %ECG_ROOT%
cd %ECG_WORK%
```

**Kapı 3:** 0.8377 ± 0.001. Tutmuyorsa ön işleme uyuşmuyordur — bu depodaki
`ecg_preprocess.py` senin eski paketininkiyle **aynı değil** (bu yeniden
yazılmış bir uygulama). O zaman eski paketi referans olarak bırak, yeni boru
hattını kendi içinde tutarlı bir sistem olarak değerlendir ve karşılaştırmayı
`test_public` skorları üzerinden yap.

---

## Saat 0:20 — FAZ 2: kapasite taraması (~2.2 sa)

En yüksek öncelik, çünkü tek kanıtlı sinyal bu: base 32 → 48 tek başına
**+0.018** getirmiş.

```bat
python train.py --preset wide --tag base_b48 --only_fold 0 --epochs 40 --patience 99 --no_bf16
python train.py --preset w64  --tag cap_b64  --only_fold 0 --epochs 40 --patience 99 --no_bf16
python train.py --preset w64  --tag cap_b64_d3 --only_fold 0 --epochs 40 --patience 99 --dropout 0.3 --no_bf16
python train.py --preset w80  --tag cap_b80  --only_fold 0 --epochs 40 --patience 99 --dropout 0.3 --no_bf16
```

Bu makinede: 0.46 + 0.70 + 0.70 + 1.04 = **2.9 saat**. Bütçe sıkışırsa
`cap_b80`'i at (aşağıya bak), 1.9 saate iner.

`base_b48` referans koşusu **şart**: mevcut 0.8468 sayısı başka bir kod
tabanından geliyor, bu boru hattıyla kıyaslanamaz. Aynı kodda ölçülmüş bir
taban çizgisi olmadan "+0.018" iddiası test edilemez.

**Kapı — OOF ile karar ver, test ile değil.**

`train.py` her tek-fold koşusunda iki sayı yazar:

```
val (OOF) macro-F1  : 0.xxxx   <- kararlari BUNUNLA ver
test_public macro-F1: 0.xxxx   (sadece rapor icin)
```

En yüksek **val** skorunu veren yapılandırmayı seç. `base_b48`'i **0.01'den
fazla** geçen yoksa `wide` (base=48) ile devam et ve FAZ 3'e geç.

> **GOREV'de bir çelişki var.** Kural 3 "`test_public` üzerinde hiçbir seçim
> yapılmaz" diyor, FAZ 2'nin kapısı ise "fold 0 **test** skoru > 0.855" diyor.
> İkisi aynı anda tutulamaz. Bu plan OOF'u kullanıyor: 750 kayıtlık test
> kümesine ±0.015 fold gürültüsüyle uyum sağlayarak genişlik seçmek, jüriye
> savunulamaz. Her iki sayı da `DENEY_KAYDI.md`'ye yazılıyor, karar OOF'la.

**Aşırı öğrenme belirtisi:** val eğrisi erken tepe yapıp düşüyorsa
`--dropout 0.3 --wd 3e-4` dene. **Augmentasyonu artırma** — mevcut set zaten
agresif ve GOREV'in 6. tuzağı tam olarak bu.

---

## Saat 2:30 — FAZ 2.5: AFIB/AFL teşhisi (~15 dk, model eğitmez)

FAZ 4'ten sonra da koşabilirsin ama mevcut `baseline/` OOF'un varsa **şimdi**
koş — sonucu bütçe dağılımını değiştirebilir.

```bat
python afib_afl_diag.py --oof baseline\r18_feat\oof_prob.npy --plot 30 --write-exclude suspects.npy
```

Üç çıktı verir:

1. **Tavan taraması.** İkili doğruluk şu olsaydı macro-F1 ne olurdu. GOREV'in
   "%90 olsa macro-F1 ≈ 0.90 olurdu" iddiasını sayıya çevirir. Sentetik veride
   bu mekanizmanın çalıştığı doğrulandı (0.70 → 0.90 arası +0.079).
2. **Yüksek güvenli ters tahminler.** Gerçek etiketi AFL olup >%80 güvenle
   AFIB denen kayıtlar ve tersi. Bunlar etiket hatası adayları — model, etiketi
   doğru olan bir kayıt hakkında nadiren yüksek güvenle yanılır.
3. **SVG çizimler** (`diag/afib_afl/`). Derivasyon **II, III, aVF, V1** —
   flutter dalgaları burada görünür. Herhangi bir çizim kütüphanesi gerekmez.

**BUNLARA GERÇEKTEN BAK.** Düzenli testere dişi F dalgası görüyorsan etiket AFL
olmalı. Düzensiz ince dalgalanma varsa AFIB. Ayırt edemiyorsan etiket zaten
tartışmalı demektir — ve bu, rapora yazılacak bilimsel bir bulgudur.

Sonra temizlenmiş eğitimi dene:

```bat
python train.py --preset <FAZ2 kazanani> --tag clean_v1 --only_fold 0 --epochs 40 --patience 99 --exclude suspects.npy
```

`--exclude` **yalnızca eğitim** bölümünden çıkarır; val ve test fold'ları
dokunulmaz kalır, yani OOF karşılaştırması dürüst olur. `test_public` zaten
teşhise hiç girmiyor.

**Karar:** OOF belirgin artıyorsa (≥0.01) yol budur, bütçeyi buraya kaydır ve
bulguyu rapora yaz. Artmıyorsa sinyal tabanlı bir tavana çarpılmış demektir;
AFIB/AFL'yi bırak, FAZ 2-3 ile genel skoru yükselt. **Her iki sonuç da
raporlanır** — "denedik, işe yaramadı, nedeni şu" savunulabilir bir cümledir.

---

## Saat 2:45 — FAZ 3: zaman çözünürlüğü (~1.2 sa)

```bat
python prep_fs.py 250
python train.py --preset <FAZ2 kazanani> --tag res_250 --cache cache_250 --only_fold 0 --epochs 40 --patience 99 --no_bf16
```

`prep_fs.py` yalnızca `X.npy` üretir; 37 özellik zaten 500 Hz'den hesaplandığı
için `F.npy` aynen kopyalanır. Karşılaştırma **tek değişkenli**: sadece ağın
gördüğü zaman çözünürlüğü değişir.

Bu makinede w64 @250 Hz tek fold ≈ **1.1 saat**.

**Kapı:** 250 Hz fold 0 **val** skoru, 150 Hz karşılığını **0.01'den fazla**
geçiyor mu?

- **Evet** → çözünürlük gerçek kaldıraç. `python prep_fs.py 500` ile 500 Hz'i
  de dene ve bütçeyi buraya yatır. **Ama dikkat:** 500 Hz'de tek fold bu
  makinede ~2.2 saat, `X.npy` 600 MB. 8 saatlik bütçede 500 Hz'li tam 5-fold
  **sığmaz** (~11 saat). O durumda 250 Hz'de kal, 500 Hz'i bir sonraki
  oturuma bırak.
- **Hayır** → 150 Hz'de kal (3 kat ucuz) ve bir daha dönme.

**Bellek:** 500 Hz denersen ve RAM 8 GB'ın altındaysa `train.py --mmap` kullan;
`X.npy` diskte kalır.

---

## Saat 4:00 — FAZ 4: kazananla tam 5-fold (~3.5 sa)

```bat
python train.py --preset <kazanan> --tag main_v2 --epochs 40 --patience 99 --no_bf16 [--cache cache_250]
python ensemble.py
```

w64 @150 Hz: **3.5 saat**. w64 @250 Hz: ~5.6 saat — 8 saatlik bütçede
FAZ 6'yı yer, planla.

**Bilgisayar kapanırsa aynı komutu tekrar çalıştır.** `train.py` biten
fold'ları `fold<k>/done.json` işaretinden tanır ve atlar. Bu doğrulandı.

`ensemble.py` yeni modeli `baseline/` içindekilerle birleştirir ve düz
ortalama / ağırlıklı / stacking'i **sadece OOF'a göre** karşılaştırır.
Stacking'i iç çapraz doğrulamayla ölçer — OOF matrisine stacker uydurup aynı
matriste skorlamak, tam olarak stacking'in kazandığı kadar iyimserdir ve
"stacking kazanıyor" sonucunu tekrarlanamaz kılar.

**Kapı:** ensemble OOF **0.8345'in üstünde** mi? Değilse yeni model katkı
vermiyordur — `DENEY_KAYDI.md`'ye yaz ve nedenini düşün, körlemesine devam etme.

`ensemble.py` ayrıca **hata korelasyonunu** basar (FAZ 5'in ölçütü):

```
ayni tahmini verme orani (dusuk = gercek cesitlilik):
  main_v2   r18_feat   0.87
```

---

## Saat 7:30 — FAZ 7: paketle ve doğrula (~20 dk)

**Bu noktada elinde teslim edilebilir bir sonuç var. Önce bunu kaydet.**

```bat
python export.py
cd package
python predict.py --batch %ECG_ROOT%\test_public.csv --root %ECG_ROOT%
python predict.py %ECG_ROOT%\Normal\NORM_000508\48090046.hea
cd ..
```

`export.py` kendi kendini doğrular:

- her grafiği onnxruntime'da yeniden koşup PyTorch'la karşılaştırır
  (fp32 için eşik 2e-3 — bu bir dışa aktarım hatasını yakalar);
- int8'i **skor** düzeyinde yargılar: fp32 ve int8 ensemble macro-F1'leri
  0.005'ten fazla ayrılırsa int8 atılır ve float32 teslim edilir;
- yan ağırlık dosyası (`*.onnx.data`) oluşursa dışa aktarımı **durdurur** —
  aksi halde `models/*.onnx` kopyalayan biri bozuk paket alır;
- paketi `test_public` üzerinde koşup PyTorch skoruyla karşılaştırır.

**Fark > 0.005 ise teslim etme, önce onu bul.** Çıkış kodu da sıfırdan farklı
olur.

Paket içeriği: `models/*.onnx` (int8), `ecg_preprocess.py` (eğitimdekinin
birebir kopyası), `wfdb_lite.py`, `predict.py`, `manifest.json`,
`preprocess.json`, `README.md`.

**PyTorch'suz doğrulama** (bu ortamda test edildi, çalışıyor):

```bat
mkdir C:\temp\notorch\torch
echo raise ImportError("blocked") > C:\temp\notorch\torch\__init__.py
cd package
set PYTHONPATH=C:\temp\notorch
python predict.py --batch %ECG_ROOT%\test_public.csv --root %ECG_ROOT%
set PYTHONPATH=
```

---

## Saat 7:50 — FAZ 6: kalan bütçeyle tarif rafinesi

Sıra önemli — üsttekiler ucuz ve birikimli:

1. **Farklı seed** (en ucuz çeşitlilik, 5-fold maliyeti):
   ```bat
   python train.py --preset <kazanan> --tag seed99 --seed 99 --epochs 40 --patience 99
   ```
2. **Tüm veriyle yeniden eğitim** (maliyet 1/5, beklenen +0.005):
   ```bat
   python train.py --preset <kazanan> --tag main_v2 --epochs 40 --patience 99 --full
   ```
   Fold'ların medyan en-iyi-epoch sayısını alır, 4250 kaydın tamamıyla o kadar
   epoch eğitir. **OOF'u yoktur** — ensemble'a ek üye olarak konur, ağırlık
   araması onu OOF'suz değerlendiremez, o yüzden düz ortalamaya girer.
3. **TTA genişletme** (eğitim gerektirmez, sadece yeniden değerlendirme):
   `--tta shift3,scale` veya `--tta shift5,scale,leads`
4. **Augmentasyon ablasyonu** (her biri tek fold):
   ```bat
   python train.py --preset <kazanan> --tag abl_nofreq --only_fold 0 --epochs 40 --aug_off freq_mask
   python train.py --preset <kazanan> --tag abl_nodrop --only_fold 0 --epochs 40 --aug_off lead_drop
   ```
   Kapatılabilir isimler: `shift scale lead_scale noise baseline lead_drop freq_mask`
5. **10-fold** (maliyet 2x, beklenen +0.005): `--folds 10`. 8 saatlik bütçeye
   w64 ile sığmaz; ancak `wide` kazandıysa düşün.

---

## Saat 8:00 — FAZ 8: rapor

`SONUC.md`'yi doldur. Şunlar olmalı:

- karşılaştırma tablosu: eski paket 0.8378 → yeni paket X, ne değişti, neden
- sınıf başına F1 ve karışıklık matrisi (`predict.py --batch` ikisini de basar)
- **AFIB/AFL ikili iç doğruluğu** — asıl ölçüt bu, ayrıca raporla
  (`predict.py` ve `ensemble.py` ikisi de basar)
- deney tablosu — `DENEY_KAYDI.md`'den derle
- **işe yaramayanlar**, nedeniyle birlikte — bunu atlama
- seçimlerin **OOF ile** yapıldığının açık beyanı

---

## Bütçe sıkışırsa — ne atılır, hangi sırayla

| at | kazanılan | risk |
|---|---|---|
| `cap_b80` (FAZ 2) | 1.0 sa | düşük — w64 zaten aşırı öğrenmeye yakın |
| FAZ 3 (250 Hz) | 1.2 sa | orta — test edilmemiş bir kaldıraç kaçar |
| FAZ 6 | kalan ne varsa | düşük — hepsi +0.005 mertebesinde |
| FAZ 4'ü `wide` ile koş | 1.2 sa | orta — kapasite kazancı kaçar |

**Asla atma:** FAZ 0 kapıları, FAZ 4'ün tam 5-fold'u, FAZ 7'nin doğrulaması.
Yarım kalan bir koşu sonuç değildir: `summary.json` yoksa sonuç yoktur.
