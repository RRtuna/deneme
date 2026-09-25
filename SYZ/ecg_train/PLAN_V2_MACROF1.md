# ext_v2 — macro-F1 artırma planı

**Tarih:** 25 Eylül 2026 · **Girdi:** `ecg_train_v2\EGITIM_RAPORU.md` (3/5 fold, 07:54)
**Kural:** her deneyde tek şey değişir; karar OOF'la verilir, `test_public` yalnız rapordur.

---

## 0. Yapılacaklar

**(S)** = siz yaparsınız · **(C)** = bilgisayarınızdaki Claude oturumunda ben yaparım

### Hazırlık

- [x] **(S)** ext_v2 eğitiminin bitmesini bekleyin — 10:34'te bitti, 9,2 sa, hata yok.
- [ ] **(S)** Branch'i çekin: `git fetch origin` → `git checkout claude/egitim-raporu-macro-f1-ictmn0`
- [ ] **(S)** `SYZ\ecg_train\tools\` içindeki `v2_analiz.py` ve `v2_karsilastir.py` dosyalarını `ecg_train_v2\tools\` klasörüne kopyalayın.
- [ ] **(S)** `ecg_train_v2` klasöründe bir terminal açıp `claude remote-control` çalıştırın (ya da Claude Desktop'u açın).
- [ ] **(S, isteğe bağlı)** v2 `train.py`'yi push edin. T4'ü ancak kodu görünce ekleyebilirim.

### Analiz (eğitim yok, dakikalar sürer)

- [ ] **(C)** Adım 0: `v2_analiz.py` biten 3 fold'la koşar. Eğitim sürerken `OMP_NUM_THREADS=2` ile koşulur.
- [ ] **(C)** Adım 1: ext_v2 bitince aynı analiz 5 fold'la tekrarlanır. Bu sayılar yeni taban olur.
- [ ] **(C)** D2 (sınıf bias'ı) ve E2 (yığma) kapıyı geçtiyse final tahmine eklenir.
- [ ] **(S)** C sonucu Ningbo hipotezini doğrularsa hedefi siz seçersiniz: aynı hastanelerden gelen test mi, yeni veri seti mi? (§3)

### Tarama eğitimleri (fold 0, sırayla)

- [ ] **(C)** T1 `--label_smoothing 0.1`, ~2,3 sa → `v2_karsilastir.py`
- [ ] **(C)** T2 `--epochs 60`, ~3,5 sa → `v2_karsilastir.py`
- [ ] **(C)** T3 `--preset w64`, ~3,5 sa → `v2_karsilastir.py`
- [ ] **(C)** T4 yumuşatılmış örnekleme (`train.py` gelirse)

### Final

- [ ] **(C)** Kapıyı geçen ayarla tam 5-fold koşu yapılır (gece, `--seed 42`).
- [ ] **(C)** `v2_karsilastir.py runs\ext_v2 runs\<kazanan>` çalıştırılır; final sistem iki koşunun 10 modelinin ortalaması olur (+ D2 bias'ı, kapıyı geçtiyse).
- [ ] **(C)** Her sonuç §4'teki tabloya yazılır, negatifler dahil.

---

## 1. Rapordan teşhis

### 1.1 Kayıp nerede

Kısmi OOF macro-F1 **0.8426**. Macro-F1 beş sınıfın ortalaması olduğu için her
sınıfın kaybı (1 − F1) doğrudan toplanır:

| sınıf | OOF F1 | kayıp | payı |
|---|---|---|---|
| AFIB | 0.6255 | 0.3745 | **%48** |
| AFL | 0.7809 | 0.2191 | **%28** |
| LBBB | 0.8976 | 0.1024 | %13 |
| RBBB | 0.9256 | 0.0744 | %9 |
| Normal | 0.9834 | 0.0166 | %2 |

**Kaybın %75'i AFIB/AFL çiftinde.** Test karışıklığında AFIB'in %34'ü AFL,
AFL'nin %21'i AFIB sanılıyor. Diğer her şeyi mükemmel yapsak bile AFIB/AFL
düzelmeden macro-F1 çok az kıpırdar.

### 1.2 Hipotezler

**H1 — etiket konvansiyonu (en önemlisi).** Challenge 2021 resmî sayımlarına göre
Ningbo'da **0 AFIB, 7.615 AFL** var (`tools/data_provenance.py` başlığında da
yazılı). ext_v2'deki AFL'nin büyük çoğunluğu buradan geliyor. Genel popülasyonda
AFL, AF'den yaklaşık 10 kat nadir; bir hastanede %22 AFL ve %0 AF olması
epidemiyolojik olarak beklenmez. Makul açıklama: **Ningbo'nun "AFL" etiketi AF'yi
de kapsıyor.** Doğruysa, AF'ye benzeyen düzensiz bir ritim Ningbo'dan geliyorsa
"AFL", başka hastaneden geliyorsa "AFIB" etiketi alıyor. O zaman model bu ayrımı
ritimden değil **kaynaktan** yapmak zorunda. Bu tablo v1'in üç bulgusuyla da
tutarlı:

- Dört bağımsız yöntem AFIB/AFL'de 0.70–0.76'da aynı duvara çarptı.
- Dış AFIB eklemek her seferinde kötüleştirdi.
- Georgia AFIB'in %22'si AFL sanıldı.

**Yeni kanıt (5 fold, `kaynak_raporu.txt`):** Chapman-Shaoxing'deki gerçek AFIB
kayıtlarının yalnız **%25–31'i** doğru tahmin ediliyor; çoğu AFL sanılıyor.
Chapman ile Ningbo aynı veri tabanından geliyor ve kayıt düzenleri aynı (ikisi de
JS önekli). Ningbo'daki binlerce "AFL" gerçekte AF ise, model "Chapman/Ningbo
düzeninde bir kayıtta AF'ye benzer ritim → AFL" kuralını öğrenmiştir ve
Chapman'ın AFIB'ini tam da bu yüzden AFL sanar. Chapman'ın AF:AFL oranı gerçekçi
(1780:445 ≈ 4:1), Ningbo'nunki değil (0:7615). Bu yüzden şüpheli etiket büyük
olasılıkla **Chapman'ınki değil, Ningbo'nunki**.

→ **Test:** `v2_analiz.py`, bölüm C. Kayıtların RR düzenliliğine (sınıf ve kaynak
bazında) bakıyor. Eğitim gerekmiyor. Beklenen: Chapman AFIB ve Ningbo AFL'nin
"düzenli %" değerleri ikisi de düşük, Chapman AFL'ninki belirgin biçimde yüksek.

**H2 — eğitim bütçesi.** Doğrulama skoru eğitimin sonuna kadar yükseliyor (en iyi
epoch 36, 40, 40). Epoch başına 10.000 örnek gösteriliyor, bu da eğitim kümesinin
yaklaşık yarısı. Daha uzun eğitim fayda sağlayabilir.

**H3 — dengeli örnekleme.** Her epoch'ta her sınıftan 2.000 örnek alınıyor.
LBBB'nin ~450 eğitim kaydının her biri 40 epoch'ta ~180 kez görülüyor, Normal
kayıtlarının ise her epoch'ta yalnız %20'si. Model eşit sınıf ağırlıklarıyla
eğitiliyor ama değerlendirme verisi dengesiz (Normal %49, AFL %24, AFIB %14,
RBBB %11, LBBB %2,3). Bu yüzden karar eşiği kaymış olabilir.

→ **Test:** `v2_analiz.py`, bölüm D. Eşiği eğitimsiz olarak ölçüyor.

Test karışıklığında tahmin sayıları gerçek sayılara çok yakın (AFIB 606/597,
AFL 1024/1054, LBBB 102/100). Bu yüzden D'den büyük bir kazanç beklemiyorum,
ama ölçmenin maliyeti yok.

**H4 — varyans.** Test'te 3 fold'un ortalaması 0.8473 verdi, tek fold'lar ise
0.8352–0.8416. Model varyansı büyük, dolayısıyla birden fazla koşunun
ortalamasını almak en güvenilir kaldıraç olabilir. `v2_karsilastir.py` bunu
OOF'ta dürüstçe ölçer: iki model de o kaydı eğitimde görmemiş oluyor.

---

## 2. Adımlar

### Adım 0 — şimdi, eğitim sürerken (~2 dk, eğitim yok)

Önce `SYZ\ecg_train\tools\` altındaki `v2_analiz.py` ve `v2_karsilastir.py`
dosyalarını `ecg_train_v2\tools\` klasörüne kopyalayın. İkisi aynı klasörde
durmalı.

```bat
cd <...>\ecg_train_v2
set OMP_NUM_THREADS=2
python tools\v2_analiz.py --cache cache_extonly_v2 --run runs\ext_v2
```

`OMP_NUM_THREADS=2`, yığma bölümünün çalışan eğitimden çekirdek çalmasını önler.

Bölümlere göre ne yapılacak:

| bölüm | sonuç | yapılacak |
|---|---|---|
| **C** | Ningbo AFL'nin düzenli oranı, diğer kaynakların AFIB'i kadar düşük | H1 doğrulandı → **§3'teki karar** |
| **B** | kaynak kuralı modele ≤ 0.03 yakın | AFIB/AFL ayrımı büyük ölçüde hastane ayrımı |
| **D2** | UYGULA | bias'ı final tahmine ekle (maliyeti 0) |
| **E2** | UYGULA | yığıcıyı finale ekle (+37 özellik) |

### Adım 1 — ext_v2 bitince (~10:40)

Adım 0'daki komutu tekrar çalıştırın; bu sefer 5 fold kullanılır. Bu sayılar yeni
**taban**. Hepsini `DENEY_KAYDI`'na yazın.

### Adım 2 — tek-fold taramalar (fold 0)

ext_v2 komutunun **aynısı**, yalnızca bir şey değişir. Ayrıca iki kural var:

- `--seed 42` kalmalı, çünkü fold bölünmesi seed'e bağlı (`make_folds(..., args.seed)`).
- `--only_fold 0` eklenir.

```bat
set BASE=--cache cache_extonly_v2 --preset wide --folds 5 --epochs 40 --patience 10 --batch 64 --balanced_epoch 10000 --only_fold 0
python train.py %BASE% --tag t1_ls01 --label_smoothing 0.1
python tools\v2_karsilastir.py --cache cache_extonly_v2 runs\ext_v2 runs\t1_ls01
```

`v2_karsilastir` iki şey söyler:

- **(a)** Değişiklik tek model olarak daha iyi mi? (eşleştirilmiş bootstrap + McNemar)
- **(b)** ext_v2 ile ortalaması ne kadar kazandırıyor? (ensemble)

| # | değişen tek şey | hipotez | süre (fold 0) |
|---|---|---|---|
| T1 | `--label_smoothing 0.1` | H1: gürültülü AFIB/AFL etiketine karşı dayanıklılık | ~2,3 sa |
| T2 | `--epochs 60` | H2 | ~3,5 sa |
| T3 | `--preset w64` | v1'deki kapasite eleme kararı 3.400 eğitim kaydıyla verilmişti; şimdi 19.907 kayıt var, yeniden test | ~3,5 sa |
| T4 | yumuşatılmış örnekleme (sınıf ağırlığı ∝ n^0,5) | H3 | kod gerekir* |

\* v2 `train.py` repoya push edilirse `--balance_power` bayrağını ben eklerim.

**Sıra:** T1 → T2 → T3. T1 en ucuz olanı ve H1'e doğrudan dokunuyor.

### Adım 3 — kazanan tam 5-fold + ensemble (gece)

1. Kapıyı geçen ayarla tam 5-fold koşun. `--only_fold` olmadan, aynı `--seed 42` ile.
2. Ardından `v2_karsilastir.py runs\ext_v2 runs\<kazanan>` çalıştırın. Final sistem
   = iki koşunun 10 modelinin ortalaması (+ D2 bias'ı, eğer kapıyı geçtiyse).

Hiçbir tek-model değişikliği kapıyı geçmezse bile ikinci bir koşu **yalnız
ensemble için** değerlidir. Bunun için en ucuz aday T1 (aynı süre).

---

## 3. H1 doğrulanırsa karar sizin

Bu durumda "macro-F1'i artırmak" iki farklı şey olabilir:

| hedef | ne demek | ne yapılır |
|---|---|---|
| **Aynı dağılım** (test aynı hastanelerden) | kaynak ipucu gerçekten bilgi taşıyor | modelin kaynağı tanıması meşru; E2 yığma bunu kullanabilir |
| **Yeni veri seti** (TEKNOFEST §7.2 gibi) | kaynak ipucu yeni hastanede çöker | kaynak bazında skor raporu + kaynak-dışı (leave-one-source-out) doğrulama; hedef metrik değişir |

İkisi farklı modeller seçtirir. Hangisini optimize ettiğimizi netleştirmeden
AFIB/AFL'ye yatırım yapmak, yanlış sayıyı yükseltme riski taşır.

---

## 4. Karar kapısı ve kayıt

- **Kabul:** Δ macro-F1 ≥ +0,005 **ve** eşleştirilmiş bootstrap P(Δ > 0) ≥ 0,95.
  Altındaki sonuç gürültü sayılır.
- Tek fold'da geçen bir değişiklik, 5-fold'da da geçmeden finale girmez. v1'deki
  Waveform-SVM dersi: fold 0/1'de +0,008/+0,011, fold 2/3/4'te −0,004/−0,006/−0,024.

| # | komut / değişiklik | fold | OOF F1 | Δ | P(Δ>0) | ensemble Δ | karar |
|---|---|---|---|---|---|---|---|
| A0 | `v2_analiz` (3 fold) | 0–2 | 0.8426 | — | — | — | |
| B0 | ext_v2 `summary.json` (test_public 5-fold ens. **0.8510**, AFIB 0.622) | 0–4 | **0.8402** | — | — | — | yeni taban |
| A1 | `v2_analiz` (5 fold) | 0–4 | | — | — | — | |
| D2 | sınıf bias'ı | 0–4 | | | | — | |
| E2 | yığma + 37 özellik | 0–4 | | | | — | |
| T1 | `--label_smoothing 0.1` | 0 | | | | | |
| T2 | `--epochs 60` | 0 | | | | | |
| T3 | `--preset w64` | 0 | | | | | |
