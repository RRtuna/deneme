# Dış Veri ile Model Geliştirme Planı

**Tarih:** 9 Eylül 2026 · **Takım:** tkt-26 · **Freeze:** 15 Eylül · **Yarışma:** 16 Eylül

> Bu plan iki öneriyi harmanlar: dış veriyi doğru kullanma tekniği (multi-label
> temizliği, kaynak dengeleme, pretrain/fine-tune) ve ölçüme dayalı kapı
> disiplini (fold-0 → McNemar → sert abort). Teknik iyi olsa da takvime
> sığmayan bir program işe yaramaz; kapılar bunun için var.

---

## 0. Nerede duruyoruz

| | değer |
|---|---|
| test_public macro-F1 | **0.841204** |
| accuracy | 0.845333 |
| Ensemble | 20 ONNX (`cv10`×10 + `main_v2`×5 + `seed99`×5) |
| Inference | MP11, **6.21 dk / 750 kayıt**, bit-birebir doğrulandı |
| Altın referans | `TEST_PUBLIC_PROVA.json` |

Sınıf bazlı F1:

```
NORMAL 0.942 | AFIB 0.746 | AFL 0.667 | LBBB 0.961 | RBBB 0.891
```

Darboğaz net: **AFL ve AFIB.**

İndirilen dış veri (8 Eylül itibarıyla ~50.003 / 77.333 `.hea`):

```
Ningbo           34.905 / 34.905   %100
Chapman-Shaoxing  5.015 / 10.247   %49
PTB-XL            5.059 / 21.837   %23
Georgia           5.024 / 10.344   %49
```

---

## 1. Ne kadar gelişebiliriz — dürüst aralık

**Beklenti: +0.005 … +0.02 macro-F1.** `>+0.02` ihtimali yaklaşık %25.
`≤0` ihtimali azımsanmayacak düzeyde.

### Neden daha iyimser değilim

Dört bağımsız yöntem AFIB/AFL çiftinde aynı duvara çarptı:

| yöntem | AFIB/AFL doğruluğu |
|---|---|
| Kaggle 8.8 M parametreli GPU modeli | 0.701 |
| Bizim CNN ensemble | 0.760 |
| 37 özellik + GBM | 0.725 |
| Yalnız o çifte eğitilmiş uzman model | 0.744 |

Farklı kapasite, mimari ve özellik kümesi — hepsi ~0.70–0.76. **8.8 M
parametreli model küçüğünden daha kötü.** Bu, veri kıtlığının değil
**sinyal/etiket sınırının** imzası.

Ek olarak: çift hatalarının **%93.4'ü düşük güvenli.** Model kendinden emin
biçimde yanılmıyor, gerçekten kararsız — belirsiz ya da tutarsız etiketlenmiş
vakalarla uyumlu bir tablo.

### Ama asıl kazanç test_public'te değil

Şartname §7.2: final, **görmediğimiz yeni bir veri setinde** dış doğrulama.

Provenance analizimiz **AFIB → Chapman ağırlıklı, AFL → Ningbo ağırlıklı**
çıkmıştı. Model sınıfı ayırt etmek yerine kısmen *kaynağı* tanıyor olabilir.
16 Eylül'de o koltuk değneği yok.

**Dış verinin en yüksek getirili işlevi bu koltuk değneğini kırmak** — her
sınıfı birden fazla kaynaktan göstermek. Bu, test_public'te belki +0.01 olarak
görünür ama gerçek finalde çok daha fazlasını kurtarabilir.

---

## 2. Riskler — ve her birinin karşılığı

| risk | neden ciddi | karşılık |
|---|---|---|
| **Sızıntı** | `test_public` kayıtları açık veri setlerinin içinde. Sızarsa OOF'ta GÖRÜNMEZ; skor yükselir, gerçek başarı düşer | 3 kademeli tarama: ad → şekil imzası → korelasyon ≥0.995. Sıfır sızıntı raporu = tarama bozuk, DUR |
| **Multi-label** | Challenge 2021 çoklu SNOMED taşır; bir kayıt AF + RBBB olabilir. Rastgele birini seçmek karar sınırını bozar | `--single-label` **varsayılan açık**; birden fazla hedef tanı taşıyan kayıt atlanır |
| **Ningbo baskınlığı** | 34.905 vs 5.000 = 7:1. Model "Ningbo cihazı nasıl görünür" öğrenebilir | `--per-class` + **`--balance-sources`** (yeni): kota sınıf içinde kaynaklar arasında sırayla dağıtılır |
| **Etiket konvansiyonu** | AFL/AFIB ayrımı merkezler arasında meşhur biçimde tutarsız. Kod haritası bunu ÇÖZMEZ | Fold-0 kapısı. AFL F1 düşerse dış veri zarar veriyor demektir |
| **Takvim** | Ensemble 20 checkpoint. Artı export + int8 + doğrulama + prova | Sert abort: 12 Eylül akşamı |

---

## 3. Kural sıfır

Bu adımların hiçbiri şunların üzerine yazmaz:

```
release/verified_20model_final_2026-09-03/     ← temel geri dönüş
competition_package/  (MP11)                   ← şu anki yarışma paketi
TEST_PUBLIC_PROVA.json                         ← altın referans
```

Yeni her şey **ayrı klasöre**: `cache_ext/`, `runs/ext*/`, `package_ext/`.

`test_public` hiçbir seçimde kullanılmaz. Dış veri `split="extra"` alır ve
**her fold'un eğitimine** girer, **hiçbir fold'un doğrulamasına** girmez —
OOF yalnız yarışma verisinde ölçülmeye devam eder, önceki tüm deneylerle
karşılaştırılabilir kalır.

---

## 4. GÜN 0 — bugün (9 Eylül)

### 4.1 Önce ölç: kaynak bağımlılığı gerçek mi (5 saniye)

```bat
python tools\source_breakdown.py --cache cache --oof ensemble_oof_prob.npy
```

**Bu ölçüm tüm planın girdisidir ve hâlâ yapılmadı.**

| "dengeli kaynaklarda doğruluk" | anlamı | dış veri |
|---|---|---|
| ~0.84'e yakın | model kaynağa dayanmıyor | değeri düşük, +0.005 civarı |
| ~0.75 veya altı | model kaynağa dayanıyor | **yüksek değer**, gerçek finalde +0.02–0.04 |

Bu sayı, dış veriyi hiç yapmasanız bile **sunumun en güçlü cümlesi** olur:

> "Geliştirmede 0.8412 aldık. Kaynak dengesi düzeltildiğinde beklentimiz şu —
> ve bunu ölçtük."

### 4.2 Envanter + sızıntı taraması (hiçbir şey yazmaz)

```bat
python tools\add_external.py --source <DIS_VERI_KOKU> --cache cache ^
    --out cache_ext --dry-run
```

**Çıktıda üç şeye bak:**

1. **Sızıntı satırları** — `sekil imzasi ayni` ve `korelasyon` sayıları
   sıfırdan büyük mü? **Sıfırsa DUR** — tarama çalışmıyor demektir, tarama
   temiz bulmuyor demek değil.
2. **Kaynak × sınıf tablosu** — her sınıf birden fazla kaynaktan geliyor mu?
   AFL yalnız Ningbo'dan geliyorsa koltuk değneğini kırmıyoruz, değiştiriyoruz.
3. **Etiket haritası** — "senin verinden türetildi" mi yazıyor, yoksa yerleşik
   tabloya mı düştü? İkincisi yüksek sesle uyarır; öyleyse örtüşme az demektir,
   dikkatli ol.

### 4.3 Cache'i yaz — kaynak dengeli

```bat
python tools\add_external.py --source <DIS_VERI_KOKU> --cache cache ^
    --out cache_ext --per-class 900 --balance-sources
```

`--balance-sources` yeni. Olmadan `--per-class 900`, AFL'nin 900'ünü de
Ningbo'dan alabilir:

```
balance YOK  ->  {Ningbo: 900}
balance VAR  ->  {Ningbo: 740, Chapman: 120, PTB-XL: 40}
```

Küçük kaynaklardan hepsi alınır, kalan kota en büyüğünden tamamlanır.

~1–2 saat ön işleme (scipy kurulu olduğu için hızlı).

### 4.4 Fold-0'ı başlat, bilgisayardan uzaklaş

```bat
python train.py --cache cache_ext --tag ext900 --preset <mevcut preset> ^
    --only_fold 0 --epochs 40 --patience 99
```

**Bu koşarken sunuma başla.** Planın en verimli saati: makine çalışırken sen
10 puanlık işi yapıyorsun.

---

## 5. GÜN 1 (10 Eylül) — KAPI 1: fold-0

Mevcut koşunun fold-0 OOF'uyla karşılaştır. **Yalnız macro-F1'e değil,
AFL ve AFIB F1'lerine ayrı ayrı bak.**

| fold-0 farkı | karar |
|---|---|
| **≤ 0** | **DUR.** Deney kaydına yaz, kalan günleri sunum + provaya ayır. MP11 teslim edilir. |
| **+0.004 … +0.01** | `--per-class 2000 --balance-sources` ile bir fold-0 daha. Artıyorsa devam, artmıyorsa DUR. |
| **> +0.01** | Tam CV'ye gir. |

> AFL F1 **düşerse** — macro-F1 artsa bile — dış veri etiket konvansiyonu
> sorunu getiriyor demektir. `--exclude-source` ile şüpheli kaynağı çıkarıp
> tekrar dene, ya da DUR.

**Durabilmek bu planın en değerli parçası.** Kazanç yoksa kovalamak 3 gün
yakar ve elinizde hiçbir şey kalmaz.

---

## 6. GÜN 2–3 (11–12 Eylül) — tam CV

Kapı geçtiyse, üç aileyi de yeniden eğit:

```bat
python train.py --cache cache_ext --tag ext_main --preset <main_v2 preseti> --folds 5  --epochs 40 --patience 99
python train.py --cache cache_ext --tag ext_cv10 --preset <cv10 preseti>    --folds 10 --epochs 40 --patience 99
python train.py --cache cache_ext --tag ext_seed99 --preset <seed99 preseti> --folds 5 --seed 99 --epochs 40 --patience 99
```

**20 checkpoint.** Gece koşuları. Bu, planın en pahalı kısmı — ve tam da bu
yüzden Kapı 1'i geçmeden başlanmaz.

---

## 7. 12 EYLÜL AKŞAMI — SERT ABORT

**Elde McNemar-anlamlı bir tam CV yoksa, MP11 + mevcut 20 model teslim edilir.
Tartışma yok.**

Gerekçe: elinizde 14.8× hızlanmış, bit-birebir doğrulanmış, 750/750 çalışan
bir paket var. Aceleyle takas edilen bir ensemble'ın riski — bozuk export,
doğrulanmamış ONNX, offline provaya vakit kalmaması — +0.01 macro-F1'den
**çok daha büyük.**

---

## 8. GÜN 4 (13 Eylül) — KAPI 2: McNemar + export

```bat
python tools\compare_runs.py runs\<mevcut ana kosu> runs\ext_main --cache cache_ext
```

**`p < 0.05` ve `ANLAMLI` yazmıyorsa kazanç gürültüdür — eski paketi teslim et.**

Anlamlıysa zincirin tamamı:

```bat
python ensemble.py --cache cache_ext --members runs\ext_main runs\ext_cv10 runs\ext_seed99 --out ensemble_ext.json
python export.py --ensemble ensemble_ext.json --cache cache_ext --out package_ext --int8
```

`export.py` ONNX çıktısının PyTorch ile eştiğini kendisi doğruluyor
(`--tol-prob`, `--tol-prob-int8`). **Bu doğrulama geçmezse teslim etme.**

Sonra yeni paketi MP11 ile koştur:

```bat
python make_submission.py --root <TEST_ROOT> --ids test_public_ids.txt ^
    --team-name "tkt-26" --team-id 807466 --application-id 4997133 ^
    --threads 1 --model-parallel 11 --out EXT_PROVA.json
```

**Kabul koşulları — hepsi:**

- [ ] `export.py` PyTorch↔ONNX doğrulaması geçti
- [ ] 750/750 kayıt, 0 hata
- [ ] `all checks passed`
- [ ] Yeniden hesaplanan macro-F1 > 0.841204
- [ ] McNemar `p < 0.05`
- [ ] Sınıf dağılımı makul, tek sınıfa yığılma yok
- [ ] Süre kabul edilebilir (~6–8 dk)

Biri bile tutmazsa: **eski paket.**

---

## 9. GÜN 5–6 (14–15 Eylül) — dondur

- Nihai paketi seç (OOF + McNemar'a bak, test_public'e DEĞİL)
- Offline prova × 2 (Wi-Fi kapalı, temiz klasör, mümkünse ikinci bilgisayar)
- USB × 2 + bulut yedeği (yarışma verisi hariç — taahhütname md. 5)
- `test_public_ids.txt`'yi USB paketinden **çıkar** (türetilmiş dosya, md. 2)
- Deney kaydını son haline getir
- **15 Eylül'den sonra kod değişmez**

---

## 10. Denemeyeceğimiz iyi fikir: pretrain → fine-tune

Aşama 1'de tüm temiz dış veriyle pretrain, aşama 2'de yarışma verisiyle
fine-tune — domain kaymasına karşı doğrudan birleştirmeden genelde daha iyi.

**Ama her modeli iki kez eğitmek demek: 40 koşu.** 6 güne sığmaz.

Kapı çok güçlü geçerse (`> +0.02`) ve 11 Eylül'de vakit görünüyorsa yeniden
değerlendirilir. Aksi halde bu, yarışmadan sonrası için bir not.

---

## 11. Özet takvim

| gün | iş | kapı |
|---|---|---|
| **9 Eyl** | `source_breakdown` → `--dry-run` → `cache_ext` → fold-0 | sızıntı taraması çalışıyor mu |
| **10 Eyl** | fold-0 sonucu | **KAPI 1:** ≤0 → DUR |
| **11–12 Eyl** | tam CV, 20 checkpoint | — |
| **12 Eyl akşam** | — | **SERT ABORT** |
| **13 Eyl** | McNemar → ensemble → export → MP11 prova | **KAPI 2:** p<0.05 |
| **14–15 Eyl** | prova, USB, freeze | — |
| **16 Eyl** | yarışma | — |

---

## 12. Sunuma ne yazacağız — her iki sonuçta da

Kapı geçse de geçmese de anlatacak bir şey var:

**Geçerse:** "Dış veriyi ölçerek ekledik. Sızıntı taraması yazdık, çoklu
etiketli kayıtları ayıkladık, kaynakları dengeledik, fold-0 kapısı koyduk,
McNemar ile doğruladık."

**Geçmezse — bu da güçlü:** "Dış veri ekledik, fold-0'da kazanç çıkmadı,
**durduk.** 13. elenen fikir bu oldu."

Elenmiş fikirlerin kaydı çoğu takımda yok:

| fikir | nasıl elendi |
|---|---|
| Balanced decoding | OOF −0.0097 |
| QRST artık özellikleri | gerçek veride +0.0006 |
| Inception1D | çeşitlilik kapısını geçemedi (0.9235) |
| CNN+Transformer hibrit | aynı (0.9259) |
| mixup çift koruması | McNemar p = 0.755 |
| Batch inference | olasılıklar 0.006 saptı |
| 2×MP5 hibrit paralellik | %1 kazanç, 2 kat operasyon riski |

Jüri şansa tutmuş bir 0.86'dan çok, **disiplinli bir 0.84'ü** ödüllendirir —
çünkü ikincisi tekrarlanabilir.

---

## Güvenli geri dönüş

Herhangi bir adım başarısız olursa:

```
competition_package/  +  MP11  →  6.21 dk, macro-F1 0.841204, bit-birebir doğrulanmış
```

Bu paket yarışmaya götürülebilir durumda ve hiçbir deney onu değiştirmiyor.
