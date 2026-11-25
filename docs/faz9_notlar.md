# Faz 9 notları: gerçek görevlerle kıyas ve LSTM'li PPO

Faz 8'i okumadıysan önce `docs/faz8_notlar.md`'ye bak. Bu not onun devamı.

Faz 9'da üç iş var:

1. **Gerçek görev kıyası** (`scripts/17_mission_benchmark.py`): Bulduğumuz yakıt değerleri gerçekçi mi?
2. **LSTM'li PPO** (`scripts/16_lstm.py`): Faz 8'deki hafıza denemesi neden başarısız oldu, ağdan mı yoksa bilgiden mi?
3. **Ay** (`scripts/18_moon.py`): Sabit olmayan, periyodik bir bozucu kuvvet. RL'in klasik tarafı geçebileceği ilk gerçekçi senaryo.

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

5 seed'in hepsi 3M adımı tamamladı (GitHub Actions, seed başına ~4 saat). Test seed'leri 100–119, sayılar 5 seed'in ortalaması. "× en iyi klasik": aynı doğrulukta en ucuz klasik tasarımın Δv'sine oran.

| k | DR-PPO | MEM-PPO | LSTM-PPO | LSTM / DR-PPO | Domine edilmeyen LSTM seed'i |
| --- | --- | --- | --- | --- | --- |
| 0 (tutma) | 5.92 | 12.20 | 8.06 | 1.50 | 0/5 |
| 0.5 | 1.095 | 1.332 | 1.008 | 0.90 | 2/5 |
| 1 | 1.057 | 1.151 | 0.996 | 0.94 | 4/5 |
| 2 | 1.028 | 1.071 | 0.998 | 0.97 | 4/5 |
| 4 | 1.017 | 1.045 | 1.000 | 0.98 | 4/5 |

Kurtarmada (acquisition) LSTM-PPO yine DR-PPO'dan biraz ucuz ama 25 durumun 25'inde domine ediliyor.

**Cevap (a): sorun ağdaydı.**
- SRP varken (k ≥ 0.5) LSTM-PPO, aynı seed'li DR-PPO'dan 40 durumun 33'ünde daha az yakıt harcıyor. Tutmada %2–10 tasarruf.
- Tutmada 20 seed/k durumunun **14'ünde hiçbir klasik tasarım onu domine etmiyor**. k ≥ 1'de klasik zarfın 0.992–1.007 katında, L1'e 6 km'den yakın.
- Yani klasik bir taban olmadan, **tamamen öğrenilmiş** bir kontrolcü ilk kez Pareto sınırına ulaştı. Bilgi gözlemlerde varmış, 10 günlük yığılmış pencere onu çıkaramamış, LSTM çıkarıyor.

**İki sınır (rapora dürüstçe yaz):**
- SRP yokken (k = 0) hafıza pahalıya geliyor: DR-PPO'nun 1.5 katı yakıt (yılda 0.24 m/s, mutlak olarak çok küçük). "Kuvvet yok" cevabını öğrenememiş.
- RES-PPO gibi sınıra **ulaşıyor ama onu ileri itmiyor**. LSTM, Kalman filtresinin yaptığını kendi öğrendi, daha iyisini değil.

Jüri sorusu: "LSTM klasikten iyi mi?" → Hayır. Klasik kadar iyi. Ama bunu kimse ona söylemeden, sadece deneyimle öğrendi. Faz 8'deki MEM-PPO başarısızlığını açıklıyor.

Ay altında ise LSTM-PPO (Ay'ı hiç görmedi) domine ediliyor: Eğitimde gördüğü kuvveti (SRP) kestirmeyi öğrenmiş, her kuvveti değil. Ay bölümüne bak.

### Olası sonuçlar nasıl yorumlanır?

- **LSTM-PPO, DR-PPO'dan iyi çıkarsa:** Sorun ağdaydı (a). Bilgi gözlemlerde varmış, MLP'li pencere onu çıkaramamış.
- **LSTM-PPO da DR-PPO'dan kötü ya da ona eşit çıkarsa:** Bilgi tek başına gürültülü gözlemlerden kolayca öğrenilemiyor (b). Bu durumda Faz 8'deki sonuç güçleniyor: En iyi yol klasik kestiriciyi (Kalman) tutup RL'e onun üstüne düzeltme öğretmek (RES-PPO).

İki sonuç da raporda kullanılabilir. Negatif sonuç da bir sonuçtur, yeter ki deney adil olsun.

---

## 3. Ay: sabit olmayan bir bozucu

### Soru

Faz 7–8'in sonucu: SRP altında klasik taraf hep Pareto sınırında kaldı, çünkü SRP **neredeyse sabit** bir kuvvet ve Kalman filtresi (Bias-LQR) onu kolayca kestiriyor. Makalenin Tartışma bölümünde de şu iddia vardı: "RL, eksik fizik yavaş değişen bir sapma olarak özetlenemediğinde avantaj kazanabilir."

Ay tam olarak böyle bir kuvvet. O zaman deneyelim.

### Fizik: Ay'ı modele nasıl ekledik?

Üç cisim modelinde (CR3BP) "Dünya" dediğimiz nokta aslında **Dünya + Ay'ı tek bir nokta kütle** olarak temsil ediyor. Gerçekte ise bunlar iki ayrı cisim ve ortak ağırlık merkezlerinin (barycenter) etrafında ayda bir kez dönüyorlar.

Ay'ı eklerken şunu yaptık:
- Modeldeki "Dünya" kütlesini (μ) ikiye böldük: Dünya (1 − q)μ ve Ay qμ, burada q = M_Ay / (M_Dünya + M_Ay) = 0.01215.
- İkisini aralarında 384.400 km olacak şekilde ağırlık merkezinin etrafına yerleştirdik.
- Dönen çerçevede Ay, Güneş–Dünya çizgisine göre her **29.53 günde** bir tur atıyor (sinodik ay).

**Toplam kütle değişmiyor**, o yüzden L1 noktası yerinden oynamıyor. Eklenen şey sadece "iki ayrı cisim" ile "tek nokta kütle" arasındaki **fark**. Buna gelgit (tidal) ivmesi deniyor.

**Büyüklüğü:** L1 civarında ortalama ~1×10⁻⁷ m/s², tepe değerde ~6×10⁻⁷ m/s². Kıyaslarsak SRP 0.6×10⁻⁷ m/s², itki sınırımız 10⁻⁶ m/s². Yani Ay, SRP'den daha güçlü ama itki sınırının altında kalıyor.

**Şekli:** Kuvvet çoğunlukla **14.8 günlük** periyotla (sinodik ayın yarısı) salınıyor, küçük bir ortalama değeri de var. Gelgit kuvveti çift taraflı olduğu için ayda iki kez tepe yapıyor. Denizlerdeki gelgitin günde iki kez olmasının sebebi de bu.

**Basitleştirmeler** (rapora yaz): Ay'ın yörüngesi çember ve ekliptik düzleminde kabul edildi. Gerçekte 5° eğik ve eksantrikliği 0.055.

### Doğrulama

`tests/test_perturbations.py` içindeki testler:
- `test_moon_matches_inertial_integration`: **En önemli test bu.** Aynı uzay aracını, Dünya ve Ay'ı iki ayrı cisim olarak koyup, sabit (inertial) bir çerçevede `solve_ivp` ile kaba kuvvetle entegre ediyoruz. Bizim dönen/titreşen çerçevedeki entegratörümüzle 30 gün sonra **1 metreden az** fark var. Hem çember hem eliptik Dünya yörüngesi için geçiyor.
- `test_moon_tidal_accel_matches_point_masses`: Formülü numpy vektörleriyle bağımsız olarak yeniden hesaplıyoruz, aynı sonuç çıkıyor.
- `test_moon_keeps_total_mass_so_far_field_vanishes`: Uzakta kuvvet 1/R⁴ gibi sönüyor. Toplam kütle korunuyor demek bu (gelgit kuvveti).
- `test_moon_off_is_the_old_code_path...`: Ay kapalıyken eski sonuçlar birebir aynı. Faz 1–8'in hiçbir sayısı değişmedi.

**İlk düzeltme:** İlk versiyonda bir gün içinde zamanı "gerçek anomaliyle doğrusal" kabul etmiştim. Eliptik durumda inertial test 30 günde 2 m fark buldu. Fiziksel olarak önemsiz ama gerek yoktu, Kepler denklemiyle zamanı tam hesaplayacak şekilde düzelttim. Şimdi 1 m'nin altında.

### Klasik tarafa da adil davranmak: Harmonic-LQR

Faz 7'de SRP gelince klasik tarafa "ders kitabı çözümünü" (LQI, Bias-LQR) vermiştik. Ay için de aynısını yapıyoruz, yoksa karşılaştırma adil olmaz.

Bias-LQR'nin Kalman filtresi kuvveti **sabit** kabul ediyor, o yüzden Ay'ın salınımını takip edemiyor. Ders kitabı çözümünün adı **iç model ilkesi** (internal model principle): Bozucunun frekansını biliyorsan, bu frekansı filtrenin içine koyarsın.

Ay'ın periyodu önceden biliniyor, bilinmeyenler sadece genlik ve faz. **Harmonic-LQR** (`src/cr3bp/control/harmonic.py`) şunu yapıyor:
- Kalman filtresi durumun yanında sabit kuvveti ve **sinodik frekansın 1., 2. ve 3. katlarında** birer salınıcıyı kestiriyor (eksen başına sabit + 3 × (cos, sin)).
- Her salınım için en ucuz **periyodik** kararlı durumu hesaplıyor, LQR ile aynı Q, R ölçüsünü kullanarak. Yakıt pahalıysa aracı salınımla birlikte salındırıyor, ucuzsa salınımı itkiyle iptal ediyor.

Testler (`tests/test_control.py`):
- Harmonik sayısı sıfır olunca Bias-LQR ile birebir aynı sonuç veriyor.
- Ay'ın kuvvetini kestirme hatası: Harmonic-LQR'de gerçek kuvvetin ~%22'si kadar, Bias-LQR'de ~%100. Yani Bias-LQR Ay'ı hiç takip edemiyor.

### Deney

Hiçbir kontrolcü Ay ile tasarlanmadı ya da eğitilmedi. Hepsi Ay'la burada **ilk kez** karşılaşıyor. RL ajanları da yeniden eğitilmedi.

- Koşullar: "Moon" (sadece CR3BP + Ay) ve "k=1 + Moon" (gerçek eksantriklik, SRP ve Ay).
- Klasik: LQR, LQI, Bias-LQR, Harmonic-LQR, her biri Faz 7'deki 115 (ρ, r_ref) tasarımıyla.
- RL: DR-PPO, MEM-PPO, RES-PPO, LSTM-PPO, 5'er seed.
- Test seed'leri 100–119, bir yıl.

### Sonuçlar

**Klasik taraf: Harmonic-LQR kazanıyor.** L1'in 10 km yakınında kalmanın yıllık maliyeti:

| | Sadece Ay | k=1 + Ay |
| --- | --- | --- |
| Bias-LQR | 10.11 m/s | 10.52 m/s |
| Harmonic-LQR | 4.97 m/s | 6.32 m/s |

Ay'ın frekansını filtreye koymak maliyeti yarıya indiriyor. İç model ilkesi çalışıyor.

**Ay'ı hiç görmemiş RL ajanları: hepsi domine edildi.** DR-PPO en iyi klasik yakıtın 3.19 katını harcıyor. Sürpriz şu: **Hafıza bu sefer işe yarıyor.** MEM-PPO (1.55×), DR-PPO'nun yarısı kadar yakıt harcıyor. SRP'de tam tersiydi. Açıklaması: Birkaç günde değişen bir kuvvet için son 10 günü görmek bilgi veriyor. Neredeyse sabit SRP için ise sadece gürültü ekliyordu.

**Ay ile eğitilen ajanlar** (adil karşılaştırma bunlarla):

| Tutma, test seed'leri | Ay'ı bilmeyen klasiğe göre | Tüm klasiklere göre | Domine edilmeyen (yoğun arama sonrası) |
| --- | --- | --- | --- |
| DR-PPO-M, sadece Ay | 0.88× | 1.36× | 0/5 |
| DR-PPO-M, k=1 + Ay | 0.82× | 1.03× | **2/5** |
| RES-PPO-M, sadece Ay | 0.79× | 0.98× | 0/5 |
| RES-PPO-M, k=1 + Ay | 0.93× | 0.98× | 3/5 |

- **RES-PPO-M** (Harmonic-LQR'nin üstüne residual): Yoğun aramadan sonra zarfın 1.000–1.006 katında. Faz 8'deki RES-PPO ile aynı desen: Sınıra ulaşıyor ama geçemiyor.
- **DR-PPO-M** (sıfırdan öğrenen, Ay ile eğitilmiş): Ay'ı bilmeyen klasik tasarımlara göre **%9–31 daha az yakıt** harcıyor (0.69–0.91×). Bunu Ay'ın frekansını kimse ona söylemeden, sadece veriden öğrendi.
- **En önemli sonuç:** DR-PPO-M'in 2 seed'i (2 ve 4), k=1 + Ay koşulunda Ay'ın frekansını bilen Harmonic-LQR'nin bile **sınırının altında** kalıyor: test seed'lerinde zarfın 0.983–0.986 katı.
- **Doğrulama** (`scripts/19` ve `scripts/20`):
  - Yoğun arama: Harmonic-LQR ve Bias-LQR, r_ref 10–1000 km × 41 ρ değeri, koşul başına 902 ek tasarım. 2 ajan hâlâ domine edilmiyor. 3. seed bu aramada domine edildi, o yüzden 3 değil 2 diyoruz.
  - **100 yeni seed (200–299)**, çevredeki 538 klasik tasarımla: İki ajan da domine edilmiyor, zarfın 0.984–0.986 katında. Sonuç tutarlı.

**Bu projenin ilk "RL klasik sınırın altında" sonucu.** Ama dürüst olmak lazım, sonuç dar:
- 5 seed'in sadece 2'si,
- sadece bir koşulda (k=1 + Ay), sadece tutmada,
- gevşek takipte (L1'e 29 ve 170 km uzaklıkta),
- kazanç en fazla **yılda 0.08 m/s**.
- Toparlanma (recovery) senaryosunda 50 RL durumunun hepsi domine edildi.

Faz 8'deki bootstrap güven aralığını burada hesaplamadık. Jüri sorarsa: "İki ayrı seed kümesinde (test ve 100 yeni seed) aynı sonuç çıktı, ama bootstrap aralığı hesaplamadık."

### Nasıl yorumlanır? (Raporun ana mesajlarından biri olabilir)

**RL'in avantajı modelleme zahmetinden tasarruf, daha iyi kontrol değil.**

- Bozucu kuvveti bilmiyorsan ya da modellemesi zorsa RL bunu veriden öğreniyor ve Ay'ı bilmeyen bütün klasik tasarımları %10–30 farkla geçiyor.
- Bozucuyu biliyorsan, klasik mühendis frekansı filtreye yazıyor (Harmonic-LQR, birkaç düzine satır kod) ve neredeyse aynı yere geliyor. Fark en fazla birkaç yüzde.

Faz 7'deki tahmin de bununla doğrulanmış oldu: "RL, eksik fizik sabit bir sapma olarak özetlenemediğinde avantaj kazanabilir." Kısmen doğru çıktı: Avantaj var, ama küçük.

### Gerçek görev kıyasına etkisi

Ay eklenince 10 km içinde tutmanın maliyeti yılda **6.32 m/s**'ye çıkıyor. En ucuz klasik tasarım 1.30 m/s harcıyor ama L1'den ortalama ~720 km uzaklaşıyor. Gerçek görevler ise yılda 0.36–2.43 m/s harcıyor. Sebebi: Halo yörüngesi, Ay'ın ve Güneş'in periyodik çekimlerini **iptal etmiyor, onlarla birlikte salınıyor**. Noktanın kendisini tutmak bu yüzden birkaç kat pahalı. Jüriye bunu böyle anlat: "Ay olmadan aynı mertebedeyiz. Ay ile noktayı tutmak pahalılaşıyor, ki bu da gerçek görevlerin neden halo yörüngesi kullandığını açıklıyor."
