# Kullanılabilir açık EKG veri setleri

Şartname §3.1.1 dış veriye açıkça izin veriyor. Aşağıdaki liste, **senin 5 sınıfın
ve 5 günlük takvimin** için değerlendirildi — genel bir katalog değil.

Sıralama ölçütü: **AFL kaydı sayısı.** Darboğazın orası ve açık veri setlerinde
en kıt sınıf o.

---

## Tier 1 — hemen kullanılabilir, dönüştürme gerektirmez

### PhysioNet/CinC Challenge 2021 eğitim kümesi ⭐ ÖNCELİK

Sekiz veri tabanını tek bir yapıda toplar. **Senin `prep.py`'n bunu doğrudan
okur** — `.hea` + `.mat`, WFDB, SNOMED-CT etiketleri, `#Dx:` satırında kodlar.
Her veri tabanı ayrı klasörde geldiği için kaynak seçimi kolay.

https://physionet.org/content/challenge-2021/1.0.3/

Sınıflarının veri tabanı bazında dağılımı (resmî `dx_mapping_scored.csv`):

| sınıf | CPSC | CPSCx | PTB-XL | Georgia | Chapman | Ningbo | **havuz** |
|---|---|---|---|---|---|---|---|
| Normal | 918 | 4 | 18092 | 1752 | 1826 | 6299 | 28891 |
| AFIB | 1221 | 153 | 1514 | 570 | 1780 | 0 | 5238 |
| **AFL** | 0 | 54 | 73 | 186 | 445 | **7615** | **8373** |
| LBBB | 236 | 38 | 536 | 231 | 205 | 248 | 1494 |
| RBBB | 1857 | 114 | 542 | 570 | 454 | 1291 | 4828 |

Senin cache'inde bu kaynaklardan zaten olanlar düşülünce **eklenebilir**:

| sınıf | eklenebilir | kaç farklı kaynaktan |
|---|---|---|
| Normal | ~28.200 | 6 |
| AFIB | ~4.700 | 5 |
| AFL | ~7.800 | 5 |
| **LBBB** | **~940** | 6 |
| RBBB | ~4.200 | 6 |

**Dengeli kalmak istersen sınırı LBBB koyuyor: ~940/sınıf ≈ 4.700 kayıt.**
Eğitim kümen 3.400 → 8.100 olur (2.4×). Bu, 5 güne rahat sığan ve dengeyi
bozmayan tek hamle. Tavsiyem bu.

Alt veri tabanlarını ayrı ayrı da indirebilirsin — hepsi gerekmez:

| alt küme | kayıt | ne için indirilir |
|---|---|---|
| `ningbo` | 34.905 | **AFL'nin tek büyük kaynağı** (7.615) |
| `chapman_shaoxing` | 10.646 | AFIB 1780 + AFL 445, ikisi de bol |
| `ptb_xl` | 21.837 | LBBB 536, Normal bol, AFIB 1514 |
| `georgia` | 10.344 | dengeli, her sınıftan var |
| `cpsc_2018` | 6.877 | RBBB 1857, AFIB 1221 |
| `cpsc_2018_extra` | 3.453 | küçük ama AFL 54 + AFIB 153 |

Kaynak çeşitliliği için **en az üç** alt küme al. `ningbo` tek başına AFL'yi
çözer ama hepsi tek hastaneden gelir.

### PTB-XL (tek başına)

https://physionet.org/content/ptb-xl/1.0.3/ — 21.837 kayıt, 18.885 hasta,
500 Hz ve 100 Hz, 10 sn, WFDB, açık erişim. Challenge 2021 içinde zaten var;
ayrı indirmenin tek avantajı SCP-ECG alt etiketlerine erişim. **AFL yalnızca
73 kayıt** — darboğazın için yetersiz.

---

## Tier 2 — değerli ama dönüştürme işi var

### Shandong Provincial Hospital (SPH)

https://data.mendeley.com/datasets/dvb5mnhfc4/1 — 25.770 kayıt / 24.666 hasta,
500 Hz, 10–60 sn, **açık erişim** (Mendeley, kayıt gerektirmez).

Ama: etiketler **AHA/ACC/HRS ifadeleri**, SNOMED değil (44 ana ifade + 15
niteleyici). Format WFDB değil. Yani hem etiket eşlemesi hem dosya dönüştürücü
yazman gerekir. AFL sayısını doğrulamadım.

**5 günde yapılacak iş değil.** Yarışma sonrası için not al.

### MIMIC-IV-ECG

Senin verinde zaten var (kayıtlarının %40'ı), ama PhysioNet'te **credentialed**
erişimli: CITI "Data or Specimens Only Research" eğitimi + başvuru onayı.
Süreç haftalar sürer. **5 güne sığmaz.**

---

## Tier 3 — senin problemin için işe yaramaz

### CODE-15%

https://zenodo.org/records/4916206 — 345.779 kayıt, devasa, açık erişim. Ama
yalnızca **6 sınıf**: 1dAVb, RBBB, LBBB, SB, AF, ST. **AFL YOK.**

Darboğazın AFIB/AFL olduğu için ana problemine dokunmaz. RBBB/LBBB'yi
güçlendirmek istersen bakılabilir, ama 400 Hz + HDF5 formatı ve dönüştürme
maliyeti var. Öncelik değil.

### CPSC 2018 (tek başına)

AFL sınıfı **yok** (0 kayıt). Challenge 2021 içinde RBBB/AFIB için zaten
alınıyor; ayrı indirmenin anlamı yok.

---

## Karar

**İndir:** Challenge 2021'den `ningbo`, `chapman_shaoxing`, `ptb_xl`, `georgia`
(gerekirse `cpsc_2018` ve `cpsc_2018_extra`).

**Ekle:** dengeli, ~900/sınıf, en az üç kaynaktan.

```
python tools/add_external.py --source <indirdiğin kök> --cache cache --out cache_ext \
    --per-class 900 --dry-run
```

Kuru çalıştırmanın **kaynak × sınıf** tablosuna bak: her sınıf birden fazla
kaynaktan geliyor mu? Geliyorsa `--dry-run`'ı kaldır.

**Ölç:** önce fold 0.

```
python train.py --cache cache_ext --tag ext --preset <mevcut> --only_fold 0 --epochs 40 --patience 99
```

Mevcut koşunun fold-0 OOF'uyla kıyasla. Artıyorsa `--per-class`'ı yükselt ve
tekrar fold 0; kazanç düzleşene kadar. Sonra tam 5-fold.

---

## Sızıntı — atlanamaz

Senin `test_public` kayıtların bu havuzların içinde. `add_external.py` üç
kademeli tarama yapıyor (kayıt adı → şekil imzası → korelasyon ≥ 0.995) ve
takılanı eklemiyor. Kuru çalıştırma çıktısında `sekil imzasi ayni` ve
`korelasyon` satırlarındaki sayılar **sıfırdan büyük olmalı** — sıfırsa tarama
çalışmıyor demektir, dur ve söyle.

## Taahhütname

İndirdiğin açık veri taahhütnameye tabi değil, ama `cache_ext/` yarışma
verisiyle karışık — **git'e girmez**. `.gitignore` `cache_*` kapsıyor, öyle
kalsın.
