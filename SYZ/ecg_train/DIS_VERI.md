# Dış veri ekleme — yönerge

## Kural dayanağı

Şartname bunu açıkça serbest bırakıyor, sormamıza bile gerek yokmuş:

> **§3.1.1:** "Yarışmacılar erişime açık olan farklı veri setlerini ve/veya kendi
> oluşturacakları veri setlerini de model eğitimi için kullanabileceklerdir."

> **§1:** "takımlar modellerini... verilen veri setleri ile birlikte **açık veri
> kaynaklarını da kullanarak** geliştirecektir."

Ve kaynağın adını şartname kendisi veriyor:

> **§3.1.1:** "PhysioNet tarafından paylaşılan **ECG Arrhythmia Dataset (1.0.0)**...
> 12 derivasyonlu, 500 Hz, yaklaşık kırk beş bin bireyin sinyali, SNOMED-CT
> etiketleri, mat ve hea formatında."
> https://physionet.org/content/ecg-arrhythmia/1.0.0/

Bu önemli: yarışma kümesi bu kaynakla **aynı etiketleme düzenini** kullanıyor.
Farklı veri setlerini birleştirirken en büyük risk olan "etiket konvansiyonu
uyuşmuyor" sorunu burada yok.

## Neden bu, kalan en değerli iş

> **§7.2:** "final aşamasında... **external validasyon** yapılacaktır. TEKNOFEST
> için özgün ve tamamen anonimleştirilmiş **yeni bir EKG veri seti**
> kullanılacaktır... **genelleme yeteneğini** değerlendirmek amacıyla."

Sıralamayı belirleyecek olan senin `test_public`'in değil, hiç görmediğin başka
bir kaynaktan gelen veri. Bu yüzden:

- Şimdiye kadar elediğimiz kaldıraçlar (QRST +0.0006, Inception, Hybrid) tek bir
  750 kayıtlık kümede son kırıntıyı kovalıyordu.
- Dış veri ise doğrudan **cross-dataset genellemeyi** iyileştiren şey. Farklı
  hastanelerden (Chapman / Ningbo) gelen kayıt çeşitliliği, tam da final
  aşamasında ölçülecek olan yeteneği besler.

---

## ASIL TEHLİKE: sızıntı

Senin `test_public` kayıtların da o açık veri setinin içinde. Dışarıdan eklenen
bir kayıt onlardan biriyle aynıysa **test üzerinde eğitmiş olursun** — ve bunu
OOF'ta göremezsin. Skor yükselir, gerçek başarı düşer.

`tools/add_external.py` üç kademeli tarama yapar:

| kademe | ne yakalar |
|---|---|
| kayıt adı | aynı isimle duran kopyalar |
| şekil imzası | yeniden adlandırılmış kopyalar (kazanç/ofset değişimine duyarsız) |
| korelasyon ≥ 0.995 | yeniden örneklenmiş / kırpılmış kopyalar |

Herhangi birine takılan kayıt **eklenmez**.

Sentetik sızıntı testiyle doğrulandı: `test_public`'ten alınıp yeniden
adlandırılmış 12 kaydın **12'si de** yakalandı, hiçbiri sızmadı.

## Etiket haritası tahmin edilmiyor

SNOMED kodlarını elle yazmak riskli (RBBB için 59118001 mi, 713427006 mi?).
Betik önce senin cache'indeki kayıtların kaynaktaki karşılıklarını bulur ve
"senin hangi etiketin kaynakta hangi kodla görünüyor" tablosunu **senin
verinden** çıkarır. Ortüşme < 50 kayıtsa yerleşik tabloya düşer ve bunu yüksek
sesle söyler — o durumda kodları kendin doğrula.

---

## Adımlar

### 0. İndir

https://physionet.org/content/ecg-arrhythmia/1.0.0/ — birkaç GB. Diskte yerin
olduğundan emin ol.

### 1. Kuru çalıştırma (hiçbir şey yazmaz)

```
python tools/add_external.py --source <indirdiğin klasör> --cache cache --out cache_ext --dry-run
```

Bakacakların:

- **etiket haritası** — "harita SENIN verinden turedi" yazıyor mu? Yazmıyorsa
  kodları doğrula.
- **sızıntı satırları** — `sekil imzasi ayni` ve `korelasyon >= 0.995` kaç kayıt?
  Sıfırdan büyükse iyi, tarama işini yapıyor demektir.
- **sınıf tablosu** — hangi sınıftan kaç kayıt eklenebilir?

### 2. Ne kadar ekleyeceğine karar ver

Kuru çalıştırmanın sınıf tablosuna bakarak:

| strateji | komut | ne zaman |
|---|---|---|
| dengeli, hepsinden | `--per-class N` (N = en az bulunan sınıfın sayısı) | **önce bunu dene** — final dış doğrulama olduğu için genelleme önemli |
| sadece darboğaz | `--only AFIB,AFL` | ikinci deney; AFIB/AFL'yi hedefler ama dengeyi bozar |

Sınıf dengesini korumak önemli: senin kümen tam dengeli (850/sınıf), tek bir
sınıfa binlerce kayıt eklemek modeli o sınıfa kaydırır.

### 3. Yaz

```
python tools/add_external.py --source <klasör> --cache cache --out cache_ext --per-class N
```

`X.npy` yeniden yazılır (mevcut cache'e **dokunulmaz**), dış kayıtlar
`split="extra"` alır.

### 4. Eğit

Ana koşunun komutunun **birebir aynısı**, sadece `--cache` ve `--tag` değişsin:

```
python train.py --cache cache_ext --tag ext --preset <mevcut> --folds 5 --epochs 40 --patience 99
```

Başlangıçta şu satırları görmelisin:

```
dis veri: NNNN kayit (yalnizca egitim fold'larina girer)
  dis veri: egitime NNNN kayit eklendi (dogrulama MMM kayit, yalnizca yarisma verisi)
```

**Kritik tasarım:** dış kayıtlar her fold'un **eğitimine** girer, hiçbir fold'un
**doğrulamasına** girmez. Yani OOF skoru hâlâ yalnızca yarışma verisinde
ölçülüyor ve **mevcut koşunla doğrudan karşılaştırılabilir**.

### 5. Karşılaştır

```
python tools/compare_runs.py runs/<ana koşun> runs/ext --cache cache_ext
```

Eşleştirilmiş McNemar. `p < 0.05` ve `ANLAMLI` yazmıyorsa kazanç gürültüdür.

---

## Kapı

| OOF farkı | karar |
|---|---|
| > +0.01 ve p < 0.05 | **uygula** — `ensemble.py` + `export.py` ile paketle |
| +0.004 – +0.01 | sınırda; `--per-class`'ı artırıp tekrar dene |
| ≤ 0 | dış veri yardım etmiyor. `DENEY_KAYDI.md`'ye yaz, geri dön. |

Not: OOF senin kendi verinde ölçülüyor, ama asıl hedef **dış doğrulama**. Dış
veriyle eğitilen bir model, OOF'u aynı kalsa bile final aşamasında daha iyi
olabilir. Bu yüzden OOF'ta belirgin bir **düşüş** yoksa dış veriyi tutmak
savunulabilir bir tercihtir — ve raporda bunu gerekçelendirebilirsin.

---

## Taahhütname — unutma

> **md. 5:** "Veri seti... GitHub, GitLab, Bitbucket... platformlara yüklenemez."
> **md. 2:** "Veri Seti: ...ve bunlardan **türetilebilecek tüm veri, dosya ve
> içerikleri** ifade eder."

- `cache/`, `cache_ext/`, `runs/`, `package/`, `*.npy` — hepsi `.gitignore`'da,
  öyle kalsın.
- İndirdiğin PhysioNet verisi **açık kaynak**, taahhütname onu kapsamıyor. Ama
  `cache_ext/` yarışma verisiyle karışık — o yüzden **o da git'e girmez**.
- **md. 3:** yarışma verisi başka bir yarışma/proje kapsamında kullanılamaz.
