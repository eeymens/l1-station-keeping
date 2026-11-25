# Faz 9 notları: gerçek görevlerle kıyas ve LSTM'li PPO

Faz 8'i okumadıysan önce `docs/faz8_notlar.md`'ye bak. Bu not onun devamı.

Faz 9'da iki iş var:

1. **Gerçek görev kıyası** (`scripts/17_mission_benchmark.py`): Bulduğumuz yakıt değerleri gerçekçi mi?
2. **LSTM'li PPO** (`scripts/16_lstm.py`): Faz 8'deki hafıza denemesi neden başarısız oldu, ağdan mı yoksa bilgiden mi?

---

## 1. Gerçek görev kıyası

### Soru

Jüri büyük ihtimalle şunu soracak: "Yılda 2 m/s buldun. Bu çok mu, az mı? Gerçek uydular ne kadar harcıyor?" Bu bölüm o sorunun cevabı.

### Kaynaklar ve sayılar

| Görev | Yıllık Δv [m/s] | Nereden |
| --- | --- | --- |
| SOHO 1996–98 | 2.4 | Roberts (2003): "devir başına ortalama 1.2 m/s, yani yılda 2.4 m/s" |
| SOHO 2004–11 | 0.36* | Roberts (2011): manevra başına ortalama 0.09 m/s, yılda ~4 manevra |
| WIND 2008–11 | 0.68* | Roberts (2011): manevra başına ortalama 0.17 m/s, yılda ~4 manevra |
| ACE 2003–11 | 1.0 | Roberts (2011): "yılda yaklaşık bir metre/saniye" |
| JWST (L2, planlanan) | 2.43* | Dichmann vd. (2014): 10.5 yılda 25.5 m/s |

\* Bu değerleri **biz hesapladık** (manevra büyüklüğü × yılda 4, ya da toplam ÷ yıl). Makalede yıllık değer olarak doğrudan yazmıyor. Jüriye de böyle söyle, kaynağa olmayan bir şeyi yüklemiş olmayız.

Bizim değerler (k = 1, yani gerçek eksantriklik ve bizim SRP varsayımımız, 20 test seed'i, tutma senaryosu):

| Bizim kontrolcü | Yıllık Δv [m/s] | L1'e ortalama uzaklık |
| --- | --- | --- |
| En ucuz klasik tasarım | 0.40 | ~222 km |
| 10 km içinde kalan en ucuz klasik | 1.85 | 9.2 km |
| DR-PPO | 2.02 | 1.4 km |
| RES-PPO | 1.87 | 6.4 km |

### Nasıl yorumlanır?

**Mertebe tutuyor.** Gerçek görevler yılda 0.36–2.43 m/s harcıyor, bizimkiler 0.40–2.02 m/s. Sayılar aynı aralıkta. Bu, simülatörümüzün ve SRP varsayımımızın gerçekçi olduğunun bağımsız bir kontrolü.

**Ama bu bir yarış değil.** Görevler aynı değil, jüriye bunu sen söyle, onlar sormadan:

- Gerçek uydular L1'in **etrafında** 100.000 km'lerce genişlikte halo ya da Lissajous yörüngeleri çiziyor (SOHO: A_y ≈ 670.000 km). Biz noktanın **kendisini** birkaç km içinde tutuyoruz.
- Onlar yılda yaklaşık 4 büyük manevra yapıyor. Biz her gün küçük, sürekli bir itki veriyoruz.
- Bizim modelimiz düzlemsel (2 boyutlu). Ay ve diğer gezegenler modelde yok.

**Neden tam L1'i tutmak daha pahalı?** SRP aracı sürekli Güneş'ten uzağa itiyor. Tam L1'de durmak istersen bu itmeyi her gün iptal etmen gerekiyor, maliyeti de yılda ~1.9 m/s (5.9×10⁻⁸ m/s² × 1 yıl). Halo yörüngesi ise SRP'yi yörüngenin şekline katıyor, onu iptal etmek zorunda kalmıyor. Bizim en ucuz klasik tasarımın da 0.40 m/s harcaması aynı sebepten: o tasarım L1'i tutmaya çalışmıyor, SRP'nin kaydırdığı yeni denge noktasına doğru (~170–220 km Güneş tarafına) kayıyor.

**Bonus bulgu (Roberts 2011):** Bir libration noktası yörüngesini düzeltmenin maliyeti, son manevradan bu yana geçen süreyle **üstel** artıyor, yaklaşık 16 günde bir ikiye katlanıyor. Bizim kontrolcülerin her gün küçük itki vermesinin ucuz olmasının sebebi bu. Kararsızlık büyümeden müdahale ediliyor. Faz 2'deki kararsız özdeğer (λ = 2.533) ile aynı fizik.

### Rapora nasıl yazılır?

- "Sonuçlarımız SOHO, ACE ve WIND gibi gerçek görevlerle aynı mertebede" de. "Gerçek görevlerden daha iyi" deme, görevler farklı.
- Yerli bağlantı için **Giriş** kısmına bir motivasyon paragrafı koyabilirsin: Türkiye'nin Ay Araştırma Programı ve ileride gelecek derin uzay görevleri, L1 benzeri noktalarda düşük yakıtlı ve otonom kontrolün neden önemli olduğu.
- **Gelecek çalışma** kısmına bir cümle: "Yöntem, yerdurağan (GEO) haberleşme uydularının konum koruma problemine (ör. Türksat) uyarlanabilir." Türksat'ın verisini kullanmadık, kullandık gibi de yazma.

---

## 2. LSTM'li PPO (LSTM-PPO)

### Soru

Faz 8'de MEM-PPO (son 10 günü gören PPO) DR-PPO'dan **kötü** çıktı. İki açıklama mümkün:

- **(a) Ağ sorunu:** Basit bir MLP, 100 sayılık gürültülü bir pencereden yavaş değişen bir kuvveti (SRP) çıkarmayı bu eğitim bütçesiyle öğrenemiyor.
- **(b) Bilgi sorunu:** 10 günlük gürültülü navigasyon verisinde bu bilgi zaten çok zayıf. Hiçbir ağ onu kullanamaz.

LSTM bu ikisini ayırıyor.

### LSTM nedir?

Tekrarlayan (recurrent) bir ağ. Her adımda sadece **o günkü** gözlemi alıyor, ama içinde bir **hafıza vektörü** (128 sayı) taşıyor ve her gün bunu güncelliyor. Neyi hatırlayıp neyi unutacağını kendisi öğreniyor. Bölüm boyunca, yani 365 gün boyunca hafızası devam ediyor. MEM-PPO'nun 10 günlük penceresi gibi bir sınırı yok.

Kalman filtresiyle benzerliği: Filtre de her gün yeni ölçümü alıp içindeki kestirimi (x̂, d̂) güncelliyor. LSTM, filtrenin yaptığını **kendi kendine öğrenmeye** çalışıyor.

### Adil karşılaştırma

Her şey DR-PPO ve MEM-PPO ile aynı:

- aynı hiperparametreler (lr 3e-4, γ 0.99, 8 ortam, 512 adım, 10 epoch, 128–128 başlıklar),
- aynı domain randomization: k ~ U(0, 4),
- aynı model seçimi: seed 0–4'te, k = 0, 1, 2, 4,
- 5 eğitim seed'i, her biri 3M adım,
- aynı test: seed 100–119, k = 0, 0.5, 1, 2, 4, aynı 345 klasik tasarım.

Tek fark: politika ağının önüne 128 birimlik bir LSTM eklendi (`sb3-contrib` kütüphanesindeki `RecurrentPPO`).

### Kod

- `scripts/08_train_ppo.py --lstm 128`: LSTM'li eğitim.
- `src/cr3bp/training.py` → `RecurrentPredictor`: LSTM'in hafızasını bölüm boyunca taşıyor. Yeni bölüm başlayınca hafızayı sıfırlıyor.
- `PolicyController.reset()` artık `RecurrentPredictor.reset()`'i de çağırıyor. Bunu iki test kontrol ediyor (`tests/test_env.py`):
  - `test_lstm_state_carries_over_and_resets`: Aynı girdi verildiğinde hafıza yüzünden farklı aksiyon çıkıyor mu, sıfırlayınca ilk aksiyona dönüyor mu? Aynı seed'lerle iki değerlendirme birebir aynı sonucu veriyor mu?
  - `test_lstm_policy_round_trip`: Kaydet → yükle → uçur.

Jüri "LSTM'in hafızası bir sonraki bölüme sızıyor mu?" diye sorarsa cevap: Hayır, bölüm başında sıfırlanıyor. Bunu yukarıdaki test kontrol ediyor.

### Neden GitHub Actions'ta eğitildi?

LSTM'li PPO, MLP'li PPO'dan ~10 kat yavaş: Bir seed için MLP ~25 dk sürerken LSTM ~4 saat sürüyor. 5 seed'i paralel eğitmek için GitHub'ın ücretsiz sunucularını kullandık (`.github/workflows/train-lstm.yml`, her seed ayrı bir iş). Eğitim adımları, seed'ler ve hiperparametreler o dosyada açıkça yazıyor. İsteyen aynısını tekrar çalıştırabilir.

`--max-minutes 330` bir güvenlik sınırı: GitHub bir işi 6 saatte kesiyor. Eğitim 3M adımdan önce bu sınıra takılırsa, o ana kadarki en iyi model kaydediliyor ve eğitim logunda kaçıncı adımda durduğu görünüyor. Böyle bir durum olursa rapora yaz.

### Sonuçlar

*(Eğitim bitince doldurulacak: `python scripts/16_lstm.py`.)*

### Olası sonuçlar nasıl yorumlanır?

- **LSTM-PPO, DR-PPO'dan iyi çıkarsa:** Sorun ağdaydı (a). Bilgi gözlemlerde varmış, MLP'li pencere onu çıkaramamış.
- **LSTM-PPO da DR-PPO'dan kötü ya da ona eşit çıkarsa:** Bilgi tek başına gürültülü gözlemlerden kolayca öğrenilemiyor (b). Bu durumda Faz 8'deki sonuç güçleniyor: En iyi yol klasik kestiriciyi (Kalman) tutup RL'e onun üstüne düzeltme öğretmek (RES-PPO).

İki sonuç da raporda kullanılabilir. Negatif sonuç da bir sonuçtur, yeter ki deney adil olsun.
