# Final planı — 16–18 Eylül, Dicle Üniversitesi

> **`PLAN_5GUN.md` geçersiz.** Final Uygulama ve Sonuç Teslim Kılavuzu okundu ve
> planın dayandığı iki varsayım da yanlış çıktı: takvim ve öncelik sırası.

---

## 1. NE DEĞİŞTİ

### Takvim: 5 gün değil, 9 gün

| tarih | ne olacak |
|---|---|
| **16 Eylül, 09:30** | **Yarışma alanında hazır bulunmak ZORUNLU** |
| 16 Eylül sabah | **Lise seviyesi model testi** — şifreli USB, süre şifreyle başlar |
| 16 Eylül öğleden sonra | Üniversite seviyesi |
| 17 Eylül | Değerlendirme, **her seviyede ilk 10 takım duyurulur** |
| 18 Eylül öğleden sonra | **Yalnızca ilk 10** sunum yapar (10 dk + 3 dk soru-cevap) |

Yer: **Dicle Üniversitesi Konferans Salonu** (Diyarbakır). Yol ve konaklama
planı da bu takvime dahil — 16 Eylül sabahı 09:30'da orada olmak zorunlu.

### Öncelik: önceki tavsiyem yanlıştı

Sana "sunum %10, modeli 0.85'e çıkarmak ~1 puan, sunumun kaldıracı 10 kat"
demiştim. **Bu yanlıştı.** Kılavuz madde 5:

> "Her seviyede **model testi sonucunda ilk 10'a giren takımlar** sunum yapmaya
> hak kazanır."

Yani sunum, model performansının **arkasında kilitli**. İlk 10'a giremezsen
sunum diye bir şey yok. Doğru sıralama:

1. **Model performansı** — hem %90 ağırlık hem de sunumun kapısı
2. **Teslim mekaniği** — bozuk JSON = sıfır, ne kadar iyi model olursa olsun
3. **Sunum** — ilk 10'a girersen 10 puan

---

## 2. YENİ ZORUNLULUKLAR — bunlar pazarlık konusu değil

### 2.1 Çıktı JSON, CSV değil

Mevcut `predict.py` CSV yazıyordu. Kılavuz madde 3 çok katı; tek hatalı alan
dosyayı geçersiz kılıyor ve **süre durmuyor**.

Bunun için `package_src/make_submission.py` yazıldı ve test edildi:
- Kılavuzdaki tüm kuralları uyguluyor (sınıf adları BÜYÜK HARF, olasılık
  toplamı 1 ±0.001, id eksik/fazla/tekrar yok, NaN/Infinity yok)
- Yazdıktan sonra dosyayı **diskten geri okuyup tekrar doğruluyor**
- Yuvarlama kayması en yüksek sınıfa yediriliyor, toplam 1'de kalıyor
- Ön işlemede hata veren kayda eşit olasılık yazıyor — **id asla eksik kalmıyor**

**Bilerek bozulmuş sekiz dosyayla test edildi, sekizi de yakalandı:**
olasılık toplamı, tekrar eden id, eksik id, küçük harf sınıf adı, NaN,
eksik sınıf, yanlış `competition_level`, `predicted_class` ile en yüksek
olasılığın tutmaması.

```
# üretme
python make_submission.py --root <test klasörü> \
    --team-name "tkt-26" --team-id TEAM_XXX --application-id BASVURU_XXX

# var olan bir dosyayı denetleme
python make_submission.py --validate TEAM_XXX_FINAL.json --root <test klasörü>
```

> ⚠️ `team_id` ve `application_id` **KYS'deki bilgilerle aynı olmalı**.
> Gerçek değerleri şimdiden öğren ve not et.

### 2.2 İnternet yasak

> "Yarışma Test aşamasında internet erişimi yasaktır. Takımlar, model
> ağırlıkları, yazılım paketleri ve çalıştırma ortamları dâhil olmak üzere test
> için gerekli tüm bileşenleri kendi bilgisayarlarında hazır bulundurmalıdır."

Paket zaten offline çalışıyor (`onnxruntime` + `numpy`, PyTorch yok). Ama
**kanıtlanması gerekiyor** — aşağıdaki prova maddesine bak.

### 2.3 Olasılıkları ezme — eşitlik bozucu PR-AUC

> "Takımlar arası eşitlik olması durumunda... sınıf olasılıklarından hesaplanan
> **PR-AUC** metriği dikkate alınacaktır."

Sert 0/1 olasılık yazma. Modelin gerçek ensemble olasılıklarını ver.
`make_submission.py` bunu zaten yapıyor.

### 2.4 USB ile model paketi

Kılavuz madde 6, önerilen içerik:
- eğitilmiş ağırlıklar / checkpoint + yapılandırma
- tahmin üreten çalıştırma kodu
- `requirements.txt` veya eşdeğer bağımlılık listesi
- **README**: çalıştırma komutu, giriş klasörü, çıktı JSON'un nerede üretildiği
- **Yarışma test verisi paket içine KOPYALANMAYACAK**

Paketin `requirements.txt` ve README'si yok — yazılacak.

### 2.5 Birden fazla yükleme serbest

> "Sistem kapanış anına kadar başarıyla doğrulanmış **son** dosya resmî teslim
> kabul edilir. Hatalı/eksik son yükleme, daha önceki geçerli dosyayı geçersiz
> kılmaz."

**Strateji:** modeli koştur, JSON'u üret, **hemen yükle**. Sonra vakit kalırsa
iyileştir ve tekrar yükle. Erken bir geçerli dosya, elindeki en değerli
sigortadır.

### 2.6 Sunum şablonu

> "Sunumlar, organizasyon tarafından paylaşılacak **güncel sunum şablonuna**
> uygun hazırlanmalı ve **yalnızca takım üyeleri** tarafından gerçekleştirilmelidir."

Şablon henüz paylaşılmadıysa takip et. 10 dakika + 3 dakika soru-cevap,
100 puan üzerinden, her jüri üyesinin puanı ortalamaya giriyor.

---

## 3. 9 GÜNLÜK PLAN

### 7–8 Eylül — teslim mekaniğini kilitle (EN YÜKSEK ÖNCELİK)

Model iyileştirmesinden **önce** gelir, çünkü bozuk teslim modeli sıfırlar.

1. `make_submission.py`'yi pakete koy, kendi `test_public`'inde koştur, üretilen
   JSON'u `--validate` ile denetle
2. Gerçek `team_id` ve `application_id` değerlerini KYS'den al, not et
3. `requirements.txt` + README yaz (çalıştırma komutu, giriş, çıkış)
4. **Süre ölçümü:** 750 kayıtta ne kadar sürüyor? Kayıt başına ms'yi bil.
   Test kümesi daha büyük olabilir — kaç kayıt/dakika işleyebildiğini bilmen şart.

### 8–9 Eylül — kuru prova (kritik)

Yarışma gününü birebir taklit et:

- **Wi-Fi'yi kapat**, ethernet'i çıkar
- Temiz bir klasöre paketi USB'den kopyala
- Bilmediğin bir klasördeki kayıtlar üzerinde koştur
- JSON üret, doğrula, süreyi tut
- **Bir şey kırılırsa şimdi kırılsın, 16 Eylül'de değil**

Bu provayı en az iki kez yap, ikincisini farklı bir bilgisayarda yapabilirsen
daha iyi.

### 9–12 Eylül — dış veri (model iyileştirmesi)

Artık zaman var. `VERI_KAYNAKLARI.md` ve `DIS_VERI.md`'deki plan geçerli:

1. Challenge 2021'den `ningbo`, `chapman_shaoxing`, `ptb_xl`, `georgia` indir
2. `source_breakdown.py` koş (5 sn) — dış doğrulamada ne bekleyeceğini söyler
3. `add_external.py --per-class 900 --dry-run` → kaynak × sınıf ve sızıntı
   satırlarını kontrol et → yaz
4. **Fold-0 kapısı.** Kazanç ≤ 0 ise DUR, mevcut modeli koru
5. Kapı geçerse tam 5-fold (gece boyu), `compare_runs.py` ile McNemar

**Kural sıfır hâlâ geçerli:** çalışan 0.8412 paketi hiçbir adımda üzerine
yazılmaz. Yeni paket ayrı klasöre çıkar; 15 Eylül'de hangisini USB'ye
koyacağına OOF sayılarına bakarak karar verirsin.

### 13–14 Eylül — sunumu hazırla

Sunum şablonu geldiyse ona göre. `SUNUM_BRIEF.md` içeriği hazır.
10 dakikaya sığdır, en az 3 kez süre tutarak prova et.

Soru-cevap 3 dakika — muhtemel sorulara hazırlan:
- "Neden AFIB/AFL'de düşük?" → darboğaz anatomisi, dört bağımsız yöntemin
  aynı duvara çarpması, hataların %93.4'ünün kararsız bölgede olması
- "Modeli neden büyütmediniz?" → 8.8 M parametreli GPU modeli daha kötü
- "Genelleme?" → `source_breakdown.py` sayısı, dış veri deneyi

### 15 Eylül — dondur ve paketle

- Nihai paketi seç (eski mi yeni mi — OOF'a bak)
- USB'yi hazırla: model paketi + README + requirements.txt
- **İkinci bir USB yedeği** ve bulut yedeği (yarışma verisi hariç)
- Son bir offline prova
- **Bu tarihten sonra kod değişmez**

### 16 Eylül — yarışma günü

- 09:30'dan önce alanda ol
- Örnek veriyle JSON format testini yap — bu ücretsiz bir kontrol, kaçırma
- Şifre duyurulunca: koştur → JSON üret → `--validate` → **hemen yükle**
- Vakit kalırsa iyileştir ve tekrar yükle
- USB model paketini teslim et

---

## 4. YAPILACAKLAR LİSTESİ

**Teslim mekaniği (önce bunlar):**
- [ ] `make_submission.py` paketin içinde, gerçek veriyle test edildi
- [ ] `team_id` ve `application_id` KYS'den alındı
- [ ] `requirements.txt` yazıldı
- [ ] README yazıldı (komut, giriş, çıkış)
- [ ] Offline prova yapıldı (Wi-Fi kapalı, temiz klasör)
- [ ] Süre ölçüldü, kayıt/dakika biliniyor
- [ ] USB hazır, ikinci yedek var

**Model:**
- [ ] `source_breakdown.py` koşuldu
- [ ] Dış veri indirildi
- [ ] Fold-0 kapısı sonucu alındı
- [ ] (kapı geçtiyse) tam 5-fold + McNemar
- [ ] Nihai paket seçildi

**Sunum:**
- [ ] Organizasyonun şablonu alındı
- [ ] 10 dakikaya sığdırıldı
- [ ] 3 kez süre tutularak prova edildi
- [ ] Soru-cevap hazırlığı

---

## 5. RİSKLER

| risk | önlem |
|---|---|
| JSON formatı reddedilir | `--validate` her yüklemeden önce; örnek veri testini kaçırma |
| Süre yetmez | kayıt/dakika hızını önceden bil; erken bir geçerli dosya yükle |
| Bilgisayar/ortam sorunu | **süreyi durdurmaz** (madde 7) — ikinci bilgisayar ve USB yedeği |
| Test verisi beklenmedik formatta | `make_submission.py` `.hea` tarıyor; farklı gelirse hızlı uyarlama gerekir |
| İnternet gerektiren bir bağımlılık | offline provada yakalanır |
| Sunum şablonuna uymama | şablonu erken al |
