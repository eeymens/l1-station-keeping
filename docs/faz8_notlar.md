# Faz 8 notları: hafızalı RL ve hibrit (residual) RL

Faz 7'yi okumadıysan önce `docs/faz7_notlar.md`'ye bak. Bu not onun devamı.

---

## 1. Soru ne?

Faz 7'nin sonucu şuydu: domain randomization ile eğitilmiş PPO (DR-PPO), SRP altında en iyi klasik kontrolcünün yakıtına %1–13 kadar yaklaşıyor ama hep domine ediliyor. Domine eden de çoğunlukla Bias-LQR, yani SRP'yi **Kalman filtresiyle kestiren** LQR.

Bu bir ipucu veriyor. DR-PPO sadece **o anki** gözlemi görüyor. SRP'nin ne kadar güçlü olduğunu bilemiyor, o yüzden "ortalama bir SRP var" diye davranıyor. Kanıtı da var: SRP yokken bile L1'in 9.7 km Güneş tarafında duruyor. Bias-LQR'nin avantajı ise geçmişe bakarak SRP'yi kestirebilmesi.

> RL'e aynı bilgiyi verirsek klasik tarafı geçebilir mi?

Bunu iki yoldan deniyoruz.

## 2. MEM-PPO: hafızalı PPO

**Fikir:** Politika ağına sadece bugünün gözlemini değil, **son 10 günün** gözlemlerini ve **son 10 aksiyonunu** veriyoruz. Ağ hâlâ basit bir MLP (128–128), yapısı değişmiyor. Sadece girdisi büyüyor: 8 × 10 gözlem + 2 × 10 aksiyon = 100 sayı.

**Neden işe yarayabilir?** Bir kuvvetin varlığını tek bir anlık fotoğraftan çıkaramazsın, ama "ne yaptım, araç nasıl tepki verdi?" sorusundan çıkarabilirsin. "Son 10 günde hep aynı itkiyi verdim ama araç yine de Güneş'ten uzaklaştı" diyebiliyorsan bir kuvvet var demektir. Kalman filtresinin yaptığı şey de matematiksel olarak bu.

**Literatürde adı:** "frame stacking" ya da "observation history". Daha karmaşık versiyonu tekrarlayan ağlar (LSTM). Biz basit olanı seçtik çünkü hiperparametreleri değiştirmeden yapılabiliyor. LSTM gelecek çalışma.

**Kod:** `src/cr3bp/envs/l1_env.py` → `ObsBuilder`. `history_len = 10` olunca:
- `self.feats`: son 10 özellik vektörü. İlk adımda bugünkü gözlem 10 kere tekrarlanıyor.
- `self.acts`: son 10 aksiyon. Başta sıfır.
- `observe()` yeni gözlemi sona ekliyor, en eskiyi atıyor. `act()` aksiyonu kaydediyor.

## 3. RES-PPO: hibrit / residual RL

**Fikir:** Sıfırdan bir kontrolcü öğrenmek yerine, iyi bir klasik kontrolcünün (Bias-LQR, ρ = 1) üstüne bir **düzeltme** öğreniyoruz:

    u_toplam = u_BiasLQR + u_PPO        (normu 10⁻⁶ m/s² ile sınırlı)

Ajana ek olarak şunları da veriyoruz:
- filtrenin durum kestirimi x̂ (gürültüsü azaltılmış konum ve hız),
- filtrenin kuvvet kestirimi d̂, yani "SRP şu kadar" bilgisi,
- tabanın o anki itkisi u_BiasLQR.

Gözlem 8 + 8 + 2 + 2 = 20 sayı.

**Neden mantıklı?** PPO sıfırdan başlamıyor. u_PPO = 0 olsa bile kontrolcü zaten Bias-LQR kadar iyi (bunu bir testle doğruluyoruz: `test_residual_zero_action_is_the_bias_lqr`). RL'in tek işi, doğrusal modelin kaçırdığı şeyleri düzeltmek. Literatürde "residual reinforcement learning" deniyor (Johannink vd. 2019, Silver vd. 2018). Faz 4'teki literatür taramasında Molnar'ın tezindeki fikre de yakın.

**Önemli bir hata yakalandı:** İlk versiyonda Kalman filtresi, araca uygulanan **toplam** itkiyi değil, sadece **kendi** itkisini biliyordu. Doyum ya da RL düzeltmesi olunca filtre yanlış tahmin yapıyordu. Test (`test_residual_zero_action_is_the_bias_lqr`) 2. günde farkı yakaladı. Düzeltme: `self.base.u_prev = u` (toplam itki). Jüri "filtre neyi biliyor?" diye sorarsa cevabı bu: araca gerçekten uygulanan itkiyi.

**Adil karşılaştırma:** RES-PPO'nun kendi tabanını (Bias-LQR ρ = 1, 10⁻⁶ ile sınırlı) da ayrıca uçuruyoruz ("RES base"). Residual ajan tabanını iyileştirmiyorsa bir işe yaramıyor demektir.

## 4. Eğitim

DR-PPO ile birebir aynı: 3M adım, kübik eşleme, aynı ödül, aynı hiperparametreler. Domain randomization k ~ U(0, 4). Model seçimi k = 0, 1, 2, 4'te, seed 0–4 ile. 5'er eğitim seed'i:

    python scripts/08_train_ppo.py --steps 3000000 --seed S --dr-k-max 4 --eval-ks 0,1,2,4 --history 10 --name ppo_mem_sS
    python scripts/08_train_ppo.py --steps 3000000 --seed S --dr-k-max 4 --eval-ks 0,1,2,4 --residual-rho 1 --name ppo_res_sS

## 5. Değerlendirme

`scripts/13_memory_residual.py`: aynı test seed'leri (100–119), aynı k değerleri, aynı 345 klasik tasarım. Faz 7'deki sütunlara ek olarak RES-PPO için tabana göre oranlar (`dv_vs_base`, `dist_vs_base`).

## 6. Sonuçlar

### Test seed'leri (100–119): yakıt ÷ en az onun kadar doğru en ucuz klasik tasarımın yakıtı

| k | Tutma: DR-PPO | MEM-PPO | RES-PPO | Domine edilmeyen RES seed'i |
| --- | --- | --- | --- | --- |
| 0 | 5.92 ± 1.87 | 12.20 ± 5.10 | 0.987 ± 0.093 | 3/5 |
| 0.5 | 1.095 ± 0.029 | 1.332 ± 0.200 | 0.990 ± 0.005 | 5/5 |
| 1 | 1.057 ± 0.016 | 1.151 ± 0.104 | 0.994 ± 0.001 | 5/5 |
| 2 | 1.028 ± 0.009 | 1.071 ± 0.056 | 0.996 ± 0.003 | 5/5 |
| 4 | 1.017 ± 0.008 | 1.045 ± 0.017 | 0.999 ± 0.003 | 3/5 |

Toparlanmada (1000 km'den başla) RES-PPO 25 durumun 25'inde domine ediliyor (zarfın 1.03–1.07 katı).

### Ana bulgular, sade dille

1. **Hafıza işe yaramadı.** MEM-PPO, 10 k/senaryo kombinasyonunun 10'unda da DR-PPO'dan kötü, seed'ler arası fark da çok büyük. Bilgi girdide var ama 10 kat daha fazla navigasyon gürültüsüyle birlikte geliyor. Aynı eğitim bütçesiyle ağ bunu kullanmayı öğrenemedi. LSTM ya da daha uzun eğitim farklı sonuç verebilir, ama denemedik. Bunu "denemedik" diye açıkça söyle.
2. **Hibrit RL, klasik Pareto sınırına ulaştı.** Bu çalışmada ilk kez bir öğrenen kontrolcü domine edilmedi. k = 0.5–2 tutmada 15 durumun 15'inde hem daha ucuz hem en az onun kadar doğru bir klasik tasarım yok.
3. **Ne öğrendi?** Tabanını Pareto eğrisi **boyunca** daha sıkı takibe doğru kaydırmış. k = 1'de taban Bias-LQR 18 km'de 1.75 m/s harcıyor, RES-PPO ise 6.4 km'de 1.87 m/s. Yani yeni bir kontrol kanunu bulmadı, klasik kontrolcüyü eğri üzerinde yeniden ayarladı. Ödül fonksiyonu doğruluğu klasik kontrolcünün ρ = 1 ayarından daha çok önemsediği için de o yöne gitti.

### "RL kazandı mı?" sorusunu nasıl test ettik?

%1'in altındaki farklar tehlikeli, o yüzden iki ek sınama yaptık:

- **Yeni 100 seed (200–299), daha sık klasik ızgara** (`scripts/14_confirmation.py`): 1710 tasarım (Bias-LQR ve LQI, r_ref = 1000/300/100/30/10 km × 57 ρ) ve eşleştirilmiş bootstrap. Sonuç: 15 durumun 13'ünde domine edilmiyor, zarfın 0.994–1.001 katı.
- **Sık yerel arama** (`scripts/15_frontier_search.py`): Bias-LQR'nin iki ayar parametresi (r_ref ve c = ρ·(r_ref/1000)²) RES noktalarının etrafında 21 × 21 ızgarayla tarandı. Her k için toplam 1011 tasarım var. Sonuç: **10 ajanın hiçbiri domine edilmiyor, 9'u zarfın altında. Ama en büyük fark %0.6, yani yılda en fazla 0.010 m/s.**

### Tek cümlelik özet (jüri için)

> RL'e Kalman filtresinin bilgisini verip klasik kontrolcünün üstüne düzeltme öğretince, en iyi klasik tasarımla eşit seviyeye geliyor ama onu anlamlı biçimde geçemiyor. Sadece hafıza vermek ise işe yaramadı.

### Jüriye nasıl anlatırsın?

- "RL klasiği yendi" **deme**. "RL klasik sınıra ulaştı, %0.6'dan fazla geçemedi" de. Bunu üç farklı sınamayla gösterdin, güçlü bir iddia bu.
- Residual RL'in pratik avantajı: ajan sıfır çıktı verirse kontrolcü tam olarak güvenilir klasik kontrolcüye dönüyor. Uzay görevlerinde sertifikasyon için önemli bir özellik.
- Bulduğun hata (filtrenin toplam itkiyi bilmesi gerekiyordu) iyi bir anlatı: test yazmanın neden önemli olduğunu gösteriyor.

### Dikkat

- Bootstrap aralıkları çok dar, çünkü ajan ve klasik tasarımlar aynı seed'lerde uçuyor (eşleştirilmiş karşılaştırma). Bu aralıklar seed gürültüsünü kapsıyor ama "başka bir hata modeli olsaydı" sorusunu kapsamıyor.
- Sık arama yalnızca Bias-LQR ailesinde ve k = 0.5, 1'de yapıldı. k = 2'de sadece 14 numaralı script'in sonucu var (15 durumun 2'si domine).
