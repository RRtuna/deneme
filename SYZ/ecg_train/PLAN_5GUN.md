> # ⚠️ BU PLAN GEÇERSİZ
>
> Final Uygulama ve Sonuç Teslim Kılavuzu okundu; planın dayandığı iki varsayım
> da yanlış çıktı:
>
> 1. **Takvim:** final 16–18 Eylül (Dicle Üniversitesi), 5 gün değil 9 gün.
> 2. **Öncelik:** sunum yalnızca **model testinde ilk 10'a girenlere** açık.
>    Yani sunum, model performansının arkasında kilitli — bu belgedeki
>    "sunumun kaldıracı 10 kat" ifadesi YANLIŞTIR.
>
> Geçerli plan: **`PLAN_FINAL.md`**
>
> (Aşağıdaki içerik yalnızca kayıt amacıyla duruyor.)

---

# 5 günlük plan — zaman/kazanç sırasına göre

## Kural sıfır: çalışan paketi asla bozma

Elinde 0.8412 alan, doğrulanmış, ONNX'e çıkmış bir paket var. Aşağıdaki hiçbir
adım onun üzerine yazmaz. Her yeni deney **ayrı bir `--tag` ve ayrı bir cache**
kullanır. 11 Eylül sabahı elinde ne olursa olsun, teslim edilecek bir şey
kesin var.

Yedeği bugün al: `package/` klasörünü ve `runs/<ana koşun>/` klasörünü
harici diske kopyala.

---

## Zaman/kazanç tablosu — neye ne kadar değer

| iş | süre | beklenen kazanç | oran |
|---|---|---|---|
| **Sunum + rapor** | ~1 gün | **10 puanın tamamı** | ⭐⭐⭐⭐⭐ |
| `source_breakdown.py` | **5 sn** | 0 puan ama finali öngörür | ⭐⭐⭐⭐⭐ |
| Jüri önünde kodu koşturabilmek | ~2 saat | diskalifiye riskini sıfırlar | ⭐⭐⭐⭐ |
| Dış veri + fold-0 kapısı | ~1 gün | +0.005…0.02 macro-F1 ≈ 0.5–2 puan | ⭐⭐⭐ |
| Tam 5-fold + paketleme | ~1 gün | yukarıdakini kalıcı kılar | ⭐⭐ |
| SPH dönüştürücü | ~4 saat | belirsiz, proba bağlı | ⭐ |

**§7.4:** "Final Yarışmaları (Fiziki) %90, Final Sunum Puanlaması %10."
**§7.5:** "Yarışma jürisi, finale kalan takımların kodlarını tekrar
çalıştırmasını ve beyan ettikleri sonuçları bulmalarını isteme yetkisine
sahiptir." — bu ikincisi bir puan değil, bir **eşik**. Koşturamazsan diğer
her şey boşa gider.

---

## GÜN 0 — bugün akşam (~30 dk dikkat, gerisi arka planda)

Amaç: uzun süren her şeyi başlat, sonra bırak çalışsın.

1. **İndirmeleri başlat** (aşağıdaki linkler). Gece boyunca sürecek, sen
   uyurken inecek. Başlatmadan yatma.
2. **`source_breakdown.py` koş** — 5 saniye:
   ```
   python tools/source_breakdown.py --cache cache --oof ensemble_oof_prob.npy
   ```
   Çıkan "dengeli kaynaklarda doğruluk" sayısı, §7.2 dış doğrulamasında
   beklenecek gerçek değerdir. Bunu bil ki sunumda dürüst konuşabilesin.
3. **SPH üst verisini indir** (sadece 2 küçük CSV) ve prob koş:
   ```
   python tools/sph_probe.py --dir <o klasör>
   ```
4. **Yedek al**: `package/` + `runs/` → harici disk.

---

## GÜN 1 — dış veri, fold-0 kapısı

**Sabah** (indirmeler bitmiş olmalı):

```
python tools/add_external.py --source <indirdiğin kök> --cache cache --out cache_ext --per-class 900 --dry-run
```

İki şeye bak:
- **kaynak × sınıf tablosu** — her sınıf birden fazla kaynaktan geliyor mu?
- **sızıntı satırları** — `sekil imzasi ayni` ve `korelasyon` sayıları
  sıfırdan büyük mü? Sıfırsa tarama çalışmıyor, DUR.

Sorun yoksa `--dry-run`'ı kaldır (~1–2 saat ön işleme).

**Öğlen:** fold-0 koşusunu başlat, sonra bilgisayardan uzaklaş.

```
python train.py --cache cache_ext --tag ext900 --preset <mevcut> --only_fold 0 --epochs 40 --patience 99
```

**Fold-0 koşarken sunuma başla.** Bu, planın en verimli saati: makine
çalışırken sen 10 puanlık işi yapıyorsun.

**Akşam — KARAR NOKTASI 1.** Mevcut koşunun fold-0 OOF'uyla kıyasla:

| fold-0 farkı | ne yap |
|---|---|
| **> +0.01** | Gün 2'de tam 5-fold koş |
| **+0.004 … +0.01** | `--per-class 2000` ile bir fold-0 daha; artıyorsa devam |
| **≤ 0** | **DUR.** Dış veri yardım etmiyor. Deney kaydına yaz, kalan 3 günü sunuma ve paketlemeye ayır. |

Son satır ciddi: gerçekten durabilmek bu planın en değerli parçası. Kazanç
yoksa kovalamak 2 gün yakar ve elinde hiçbir şey kalmaz.

---

## GÜN 2 — tam koşu (kapı geçtiyse)

```
python train.py --cache cache_ext --tag ext900 --preset <mevcut> --folds 5 --epochs 40 --patience 99
```

Gece boyunca koşar. **Sen bu sırada sunumu bitir.**

Kapı geçmediyse: Gün 2 ve 3 tamamen sunum + paketleme + prova olur. Bu bir
başarısızlık değil, doğru karar.

---

## GÜN 3 — ölç, birleştir, paketle

```
python tools/compare_runs.py runs/<ana koşun> runs/ext900 --cache cache_ext
```

**KARAR NOKTASI 2:** `p < 0.05` ve `ANLAMLI` yazmıyorsa kazanç gürültüdür —
eski paketi teslim et, yenisini bırak.

Anlamlıysa:
```
python ensemble.py --cache cache_ext --members runs/ext900 <diğer üyeler>
python export.py --cache cache_ext --int8
cd package && python predict.py --batch <test_public.csv> --root <veri kökü>
```

Paketten çıkan macro-F1, `runs/ext900/summary.json`'daki `test_macro_f1` ile
**aynı olmalı**. Değilse teslim etme, dur.

---

## GÜN 4 — dondur ve prova et

- Paketi harici diske ve buluta yedekle (yarışma verisi hariç — taahhütname md. 5)
- **Jüri provası:** temiz bir klasörde, sıfırdan, paketi çalıştır. Kaç dakika
  sürüyor? Hangi komut? Not al.
- Sunum provası: 3 kez, yüksek sesle, süre tutarak
- Deney kaydını (`DENEY_KAYDI.md`) son haline getir

**Bu günden sonra kod değişmez.**

---

## GÜN 5 — tampon

Sadece beklenmedik durumlar için. Plan buraya taşmamalı.

---

## Sunumda anlatacağın şey (10 puan burada)

Elinde çoğu takımda olmayan bir şey var: **elenmiş fikirlerin kaydı.**

| fikir | nasıl elendi |
|---|---|
| QRST iptalli atriyal bant özellikleri | gerçek veride +0.0006, kapının altında |
| Inception1D mimarisi | çeşitlilik kapısını geçemedi |
| CNN+Transformer hibrit | aynı |
| Dengeli kod çözme (transport dual) | OOF −0.0097 |
| mixup çift koruması | McNemar p = 0.755, gürültü |

Bunu "denedik olmadı" diye değil, **"her fikri ölçtük ve ölçüme göre eledik"**
diye anlat. Bir kıyas kümesi kurup Bayes tavanını hesapladığını, eşleştirilmiş
McNemar testi kullandığını, sızıntı taraması yazdığını anlat. Jüri şansa
tutmuş bir 0.86'dan çok, disiplinli bir 0.84'ü ödüllendirir — çünkü ikincisi
tekrarlanabilir.

Bir de dürüstlük puanı: `source_breakdown.py`'nin verdiği "dengeli
kaynaklarda beklenen doğruluk" sayısını sunumda söyle. "Geliştirmede 0.84
aldık ama kaynak dengesi düzeltildiğinde beklentimiz şu" demek, jüri önünde
çok güçlü bir cümledir.

---

## İNDİRME LİNKLERİ

### 1. PhysioNet/CinC Challenge 2021 — ÖNCELİK

Ana sayfa: https://physionet.org/content/challenge-2021/1.0.3/

**Tamamı** (büyük, ~20+ GB — muhtemelen gereksiz):
```
wget -r -N -c -np https://physionet.org/files/challenge-2021/1.0.3/
```

**Sadece ihtiyacın olan alt kümeler** (tavsiye edilen):
```
wget -r -N -c -np https://physionet.org/files/challenge-2021/1.0.3/training/ningbo/
wget -r -N -c -np https://physionet.org/files/challenge-2021/1.0.3/training/chapman_shaoxing/
wget -r -N -c -np https://physionet.org/files/challenge-2021/1.0.3/training/ptb-xl/
wget -r -N -c -np https://physionet.org/files/challenge-2021/1.0.3/training/georgia/
```

AWS S3 üzerinden (genelde daha hızlı, hesap gerekmez):
```
aws s3 sync --no-sign-request s3://physionet-open/challenge-2021/1.0.3/training/ningbo/ ./ningbo/
```

**Not:** Alt klasör adlarını sayfadaki dizin listesinden doğrula — yukarıdakiler
beklenen adlar, ama physionet.org bana kapalı olduğu için birebir teyit
edemedim. `training/` altına girip gördüğün adları kullan.

Windows'ta `wget` yoksa: sayfadaki "Download the ZIP file" bağlantısı ya da
`curl` kullan. Tarayıcıdan tek tek indirme — binlerce dosya var.

### 2. SPH — sadece üst veri (Gün 0'da, birkaç MB)

https://data.mendeley.com/datasets/dvb5mnhfc4/1

Oradan **yalnızca** `attributes.csv` ve `code.csv` dosyalarını indir. Tüm
veri setini (25.770 HDF5 dosyası) prob sonucu iyi çıkarsa indirirsin.

### 3. İndirmeyeceklerin — ve nedeni

| veri seti | neden hayır |
|---|---|
| MIMIC-IV-ECG | credentialed erişim, CITI eğitimi + referans + manuel onay → haftalar |
| CODE-15% | 345 bin kayıt ama **AFL sınıfı yok**, darboğazına dokunmaz |
| CPSC 2018 tek başına | AFL yok; Challenge 2021 içinde zaten geliyor |
| PTB-XL tek başına | Challenge 2021 içinde zaten var, ayrıca indirme |

---

## Disk ve süre uyarısı

- Challenge 2021'in 4 alt kümesi ≈ 10–15 GB indirme
- `cache_ext/` ≈ mevcut `cache/` boyutunun 2–2.5 katı
- Ön işleme (~4.700 yeni kayıt) ≈ 1–2 saat, `prep.py` işçileriyle
- Tam 5-fold koşu, 2.4× veriyle ≈ mevcut koşunun 2–2.5 katı süre

İndirmeye başlamadan önce boş disk alanını kontrol et. Yer yoksa `ningbo` +
`chapman_shaoxing` ile yetin — AFL'nin neredeyse tamamı o ikisinde.
