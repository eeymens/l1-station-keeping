# Faz 7 notları: model uyumsuzluğu

Bu not Faz 7'yi jüriye kendi cümlelerinle anlatabilmen için yazıldı. Önce fikir, sonra denklemler, en sonda da hangi dosyada ne olduğu var. Sonuçlar için en alttaki "Sonuçlar" bölümüne bak. Oradaki her sayı `experiments/model_mismatch*.csv` dosyalarından geliyor.

---

## 1. Soru ne?

Faz 1–6'da her şey **dairesel** kısıtlı üç cisim problemindeydi (CR3BP). Bu modelde Dünya, Güneş'in etrafında tam bir çember çiziyor ve uzay aracına Güneş ile Dünya'nın çekiminden başka bir şey etki etmiyor (küçük rastgele bozucular hariç). Bu modelde LQR Pareto-optimal çıktı: hiçbir PPO seed'i LQR'nin yakıt–doğruluk eğrisinin altına inemedi.

Ama LQR'nin bir varsayımı var. Kazancı **doğrusallaştırılmış CR3BP modelinden** hesaplanıyor. RL'in ise böyle bir varsayımı yok, sadece simülatörden öğreniyor. Doğal soru şu:

> Gerçek dinamik LQR'nin modelinden saptıkça LQR bozuluyor mu? RL bu durumda kazanıyor mu?

Bunu test etmek için simülatöre iki gerçek fizik etkisi ekledik. **Kontrolcülerin hiçbirine bunu söylemedik.** Hepsi hâlâ CR3BP'yi bildiğini sanıyor.

## 2. Eklenen iki etki

### 2a. Dünya'nın eliptik yörüngesi (ER3BP)

Dünya'nın yörüngesi tam çember değil, eksantrikliği e = 0.0167. Güneş–Dünya mesafesi yıl içinde değişiyor:

    r(f) = (1 − e²) / (1 + e·cos f)        (AU cinsinden)

Buradaki **f, gerçek anomali**, yani Dünya'nın yörüngedeki açısı. Ocak başında (günberi) r ≈ 0.983 AU, Temmuz başında (günöte) r ≈ 1.017 AU.

**Problem:** Güneş–Dünya mesafesi değişince L1 noktası da hareket ediyor. Sabit bir noktayı tutmaya çalışmak anlamsız olur.

**Çözüm (pulsating koordinatlar):** Koordinat sistemini hem Güneş–Dünya çizgisiyle döndürüyoruz hem de uzunlukları o anki r(f) ile ölçekliyoruz. Ayrıca zaman yerine f'yi bağımsız değişken yapıyoruz. Böyle olunca Güneş ve Dünya hep aynı noktada duruyor, L1 de CR3BP'deki yerinde sabit kalıyor. Denklemler şöyle oluyor (′ = d/df):

    x″ − 2y′ = w(f) · ∂Ω/∂x + g(f) · uₓ
    y″ + 2x′ = w(f) · ∂Ω/∂y + g(f) · u_y

    w(f) = 1 / (1 + e·cos f)
    g(f) = (1 − e²)² / (1 + e·cos f)³

Ω, CR3BP'deki potansiyelin aynısı. **e = 0 olunca w = g = 1 ve f = t olur, yani tam olarak eski CR3BP denklemlerine dönüyoruz.** Bunu bir testle de kontrol ediyoruz (bkz. bölüm 6).

**w(f) ne demek?** Yerçekimi ve merkezkaç terimleri yıl içinde yaklaşık ±%1.7 oranında "nefes alıyor".

**g(f) nereden geliyor?** Fiziksel bir ivmeyi (itki gibi) pulsating koordinatlara çevirirken iki şey değişiyor: uzunluk r ile ölçekleniyor, zaman da f'ye dönüyor. Kepler'in ikinci yasasından df/dt = √(1−e²) / r² çıkıyor. Bunu yerine koyunca çarpan r³/(1−e²) oluyor, bu da g(f)'nin ta kendisi. Kontrolcü "şu kadar itki ver" diyor ama araca etkisi yıl içinde ±%5 oranında değişiyor. LQR bunu bilmiyor.

**Neden bu yöntem?** (1) e = 0 limiti tam olarak eski simülatörümüz, bu da kesin bir regresyon testi veriyor. (2) L1 sabit kalıyor, dolayısıyla "L1'e uzaklık" o anki gerçek L1'e uzaklık demek. (3) Libration noktası literatüründe standart yöntem bu (Szebehely 1967, *Theory of Orbits*, 10. bölüm).

**Raporlanan mesafe** fiziksel km: pulsating mesafe × r(f) × 1 AU. **Δv** de fiziksel: ∫|itki ivmesi| dt.

**Zamanlama:** Kontrolcü hâlâ her **gün** bir karar veriyor. Her günün başındaki ve sonundaki f değerini Kepler denklemiyle buluyoruz (M = E − e·sin E, Newton yöntemiyle çözülüyor), sonra o aralıkta f'ye göre RK4 ile entegre ediyoruz.

### 2b. Güneş ışınım basıncı (SRP)

Güneş ışığı araca çarpınca onu Güneş'ten dışarı doğru itiyor. 1 AU'daki ivme:

    a₀ = P · C_r · (A/m)
       = 4.56×10⁻⁶ N/m² · 1.3 · 0.01 m²/kg
       ≈ 5.9×10⁻⁸ m/s²

- **P = 4.56×10⁻⁶ N/m²:** 1 AU'daki ışınım basıncı. Güneş akısının (1367 W/m²) ışık hızına bölümü. Kaynak: Montenbruck & Gill, *Satellite Orbits* (2000), 3.4. bölüm.
- **C_r = 1.3:** yansıtma katsayısı (1 = tam soğurucu, 2 = ayna). Bu bir varsayım.
- **A/m = 0.01 m²/kg:** ~1 tonluk, ~10 m² güneş gören yüzeyi olan bir araç için varsayım. Belirli bir görevin değeri değil.

**Karşılaştırma:** İtki limiti 10⁻⁶ m/s². SRP bunun ~%6'sı, yani limit içinde kalıyor. Ama nominal "unmodelled force" (10⁻⁹ m/s²) seviyesinin ~60 katı ve **sabit bir yönde**. Tamamen karşılamaya kalkarsan yılda ~1.9 m/s Δv eder. Bu da Faz 5'teki tutma Δv'sinin (~0.06–0.1 m/s) 20–30 katı.

**Pulsating koordinatlarda SRP:** Hesap çok temiz çıkıyor. SRP, Güneş'in efektif kütlesini (1 − μ)'den (1 − μ − a₀)'a düşürüyor. Kodda tek satırlık bir fark bu.

**Fiziksel sonucu:** Denge noktası Güneş'e doğru ~180 km kayıyor (testte −100 ile −400 km arasında doğruluyoruz). SRP varken "L1'i tam tutmak" ile "kaymış denge noktasında beklemek" arasında büyük bir yakıt farkı var. Faz 7'nin en önemli fiziksel noktası bu.

### 2c. Uyumsuzluk şiddeti k

    e = k · 0.0167,      a₀ = k · 5.9×10⁻⁸ m/s²,      k ∈ {0, 0.5, 1, 2, 4}

- **k = 0:** eski CR3BP (bit bit aynı sonuç).
- **k = 1:** gerçekçi değerler.
- **k = 4:** dört katı. "Model daha da yanlış olsaydı ne olurdu?" sorusu için.

Her test seed'inde yılın hangi noktasından başlandığı (f₀) rastgele seçiliyor. Bunun için ayrı bir rastgele sayı üreteci kullanılıyor (seed + 7919). Böylece navigasyon ve itki hataları **bütün kontrolcüler ve bütün k değerleri için birebir aynı** kalıyor.

## 3. Klasik taraf neden güçlendirildi? (LQI ve Bias-LQR)

Sabit bir kuvvet (SRP) varken düz LQR (u = −Kx) **kalıcı bir hata** bırakıyor. Araç yerçekimi, itki ve SRP'nin dengelendiği bir noktaya oturuyor ve orada sürekli itki harcıyor. Bu LQR'nin değil, saf durum geri beslemesinin bilinen bir sınırlaması. Ders kitabında çözümü var. Sadece düz LQR ile kıyaslasaydık jüri haklı olarak "LQR'yi bilerek zayıf bırakmışsın" derdi. O yüzden iki standart çözüm ekledik:

### LQI (integral terimli LQR) — `src/cr3bp/control/lqi.py`, `LQIController`

Konum hatasının birikimini yeni bir durum olarak ekliyoruz:

    z[k+1] = z[k] + dt · (konum hatası)[k]

Sonra [x; z] genişletilmiş sistemi için tek bir LQR tasarlıyoruz. Ortalama konum hatası sıfır olana kadar integral büyüyor. Sonuçta araç **tam L1'de** duruyor ve SRP'yi sabit bir itkiyle karşılıyor.

- `t_int_days = 60`: integralin ağırlığı. "60 gün boyunca 1000 km hata" bir "1000 km anlık hata" kadar sayılıyor.
- **Anti-windup:** itki doyuma girerse integrali o adımda büyütmüyoruz.

**Zayıf yanı:** L1'i tam tutmak en pahalı seçenek, çünkü SRP'nin tamamını yakıtla karşılıyorsun.

### Bias-LQR (bozucu kestirimli LQR) — aynı dosya, `BiasLQRController`

İki parçası var:

1. **Kalman filtresi:** Durumu (konum, hız) ve **sabit bozucu ivmeyi d'yi** birlikte kestiriyor. Model şu:

       x[k+1] = A_d x[k] + B_d (u[k] + d[k])
       d[k+1] = d[k] + (küçük rastgele yürüyüş)

   Ölçüm gürültülü konum ve hız. Filtre "beklediğimden hep aynı yöne sapıyorum, demek ki bir kuvvet var" diyerek d'yi öğreniyor. Rastgele yürüyüşün büyüklüğü (`bias_walk_mps2 = 2e-10`) filtrenin yavaş değişen bozucuları ne kadar hızlı takip edeceğini belirliyor.

2. **En ucuz denge noktası:** d̂ kestirimi elindeyken şu problemi çözüyoruz:

       min  xᵀQx + uᵀRu     öyle ki    (A_d − I) x + B_d (u + d̂) = 0

   Yani "doğrusal modelde d̂'ye karşı tutulabilecek en ucuz sabit durum hangisi?" sorusu. Q ve R, LQR'ninkiyle aynı. Çözüm d̂'ye göre doğrusal (x_ss = M_x d̂, u_ss = M_u d̂). M_x ve M_u'yu bir kere KKT sistemiyle hesaplıyoruz. Kontrol kanunu:

       u = u_ss − K (x̂ − x_ss)

**Sezgisel anlamı:** ρ büyükse (yakıt pahalıysa) araç, yerçekiminin SRP'yi bedavaya dengelediği kaymış noktaya yakın park ediyor. ρ küçükse SRP'yi itkiyle karşılayıp L1'e yakın duruyor. Gerçek görevler de SRP'yi böyle ele alıyor: referans yörüngenin bir parçası olarak.

İki kontrolcü de bölümler arasında hafıza tutuyor (integral, filtre durumu). `run_episode` her bölümün başında `reset()` çağırıyor. Bunun testi de var.

## 4. RL tarafı

### (a) Nominal PPO, yeniden eğitilmeden

Faz 5'in kübik eşlemeli PPO modelleri (`ppo_l1`, `ppo_l1_s1`, `ppo_l1_s2`), hiç değiştirilmeden bozuk dinamikte uçuruluyor. Soru şu: CR3BP'de öğrenilen politika gerçeğe ne kadar dayanıklı?

### (b) Domain randomization (DR-PPO)

**Domain randomization nedir?** Ajanı tek bir dünyada değil, her bölümde biraz farklı bir dünyada eğitiyorsun. Böylece "tek bir modele göre en iyi" yerine "birçok olası modelde iyi" bir politika öğrenmesi bekleniyor. Robotikte simülasyondan gerçeğe geçiş (sim-to-real) için standart bir yöntem.

Bizde her bölümün başında rastgele çekilenler:

- k ~ U(0, 4)
- yılın başlangıç noktası f₀ ~ U(0, 2π)
- SRP alan/kütle çarpanı ~ U(0.5, 1.5)

Bunlar dışında **PPO hiperparametrelerine dokunulmadı**: 3M adım, kübik eşleme, aynı ödül, aynı ağ. Model seçimi yine sadece eğitim seed'leriyle (0–4) yapıldı, ama artık k = 0, 1, 2, 4'teki skorların ortalamasıyla. Test seed'leri (100–119) eğitimde hiç kullanılmadı. 5 eğitim seed'i var: `ppo_dr_s0 … ppo_dr_s4`.

**Bilinen bir sınırlama:** Gözlem hâlâ 8 sayı (anlık konum ve hız). Ajanın hafızası yok, yani "SRP'nin ne kadar olduğunu" geçmişe bakarak kestiremiyor. Bias-LQR'nin Kalman filtresi ise bunu yapıyor. Adil olmak için bunu makalede açıkça yazıyoruz. Gelecek çalışma: gözleme geçmiş eklemek (frame stacking) ya da tekrarlayan (LSTM) bir politika kullanmak.

## 5. Nasıl karşılaştırıyoruz?

Her k ve her senaryo için (tutma: L1'den başla, toparlanma: 1000 km'den başla):

- **Δv, ortalama/RMS mesafe, hayatta kalma oranı:** 20 test seed'i üzerinden.
- **`lqr_ratio`:** `10_seed_study.py`'deki yöntemin aynısı. LQR'nin ρ taraması (57 değer), sol kol, log-log interpolasyon. "Aynı doğrulukta LQR'nin kaç katı yakıt?"
- **`best_classical_ratio`:** Model Δv'si bölü, en az onun kadar doğru olan **en ucuz klasik kontrolcünün** Δv'si. Burada LQR, LQI ve Bias-LQR'nin bütün ρ'ları kullanılıyor. Basamak fonksiyonu, interpolasyon yok. **< 1 çıkarsa model, kendi doğruluğunda bütün klasik tasarımları yenmiş demektir.**
- **`dominated`:** Hem daha ucuz hem en az onun kadar doğru bir klasik nokta var mı?
- **Ayrık etkiler:** k = 1 ve 4'te "sadece eksantriklik" ve "sadece SRP" ayrı ayrı uçuruluyor (`model_mismatch_parts.csv`). Hangi etkinin neyi bozduğu buradan görünüyor.

## 6. Testler (`tests/test_perturbations.py`)

| Test | Ne kontrol ediyor |
| --- | --- |
| `test_null_perturbation_reproduces_cr3bp_bit_for_bit` | k = 0 iken sonuçlar eski simülatörle **bit bit** aynı mı |
| `test_er3bp_integrator_reduces_to_cr3bp_when_e_is_zero` | Yeni entegratör e = 0'da eskisiyle 10⁻¹⁵ içinde aynı mı |
| `test_l1_is_still_an_equilibrium_in_pulsating_coordinates` | L1, eliptik durumda da pulsating koordinatlarda denge noktası mı (Szebehely) |
| `test_physical_l1_distance_pulsates_with_earths_orbit` | Fiziksel L1–Dünya mesafesi günberi/günöte oranı (1−e)/(1+e) mi |
| `test_matches_inertial_integration` | **En güçlü test.** Aynı aracı (itki + SRP dahil) Güneş ve Dünya gerçek Kepler elipslerinde dönerken, **eylemsiz (inertial) çerçevede kaba kuvvetle** DOP853 ile entegre ediyor ve pulsating sonuçla karşılaştırıyor. 30 günde fark < 1 m. w(f), g(f), SRP ve Kepler çözümünün hepsi bununla doğrulanıyor |
| `test_srp_moves_the_equilibrium_sunward` | SRP denge noktasını Güneş'e doğru 100–400 km kaydırıyor mu, araç orada itkisiz duruyor mu |
| `test_lqi_removes_steady_offset_under_constant_srp` | LQI, SRP altında L1'e geri dönüyor mu |
| `test_bias_lqr_estimates_the_srp_acceleration` | Kalman filtresi SRP'yi %15 içinde doğru kestiriyor mu |
| `test_stateful_controllers_reset_between_episodes` | Hafızalı kontrolcüler her bölümde sıfırlanıyor mu |
| `test_domain_randomised_env_draws_new_physics_each_episode` | DR ortamı her bölümde yeni fizik çekiyor mu |

## 7. Hangi dosyada ne var?

| Dosya | İçerik |
| --- | --- |
| `src/cr3bp/perturbations.py` | `Perturbation` sınıfı (e, SRP, f₀), `from_k`, Kepler fonksiyonları (`mean_from_true`, `true_from_mean`, `radius`), `er3bp_accel` (yukarıdaki denklemler, satır satır), `rk4_er3bp` |
| `src/cr3bp/control/station_keeping.py` | `StationKeepingSim`'e `perturbation` parametresi. `step()` içinde: perturbation yoksa **eski kod yolu**, varsa Kepler ile f_end'i bulup `rk4_er3bp`. `distance_km` artık × r(f) |
| `src/cr3bp/control/lqi.py` | `LQIController`, `BiasLQRController` |
| `src/cr3bp/evaluation.py` | `perturbation_for(k, seed)`: test seed'inden f₀ üretir. `evaluate(..., perturbation_k=k)` |
| `src/cr3bp/envs/l1_env.py` | `dr_k_max` (domain randomization), `perturbation_k` |
| `src/cr3bp/training.py` | `EvalCallback(ks=...)`: model seçimi birden fazla k'da |
| `scripts/08_train_ppo.py` | `--dr-k-max 4 --eval-ks 0,1,2,4` bayrakları |
| `scripts/12_model_mismatch.py` | Bütün deney: klasik taramalar, RL değerlendirmesi, oranlar, ayrık etkiler, şekiller |

`er3bp_accel` fonksiyonunu satır satır okursan:

- `c = 1 + e cos f`, `w = 1/c`, `g = (1−e²)² w³`: yukarıdaki w(f) ve g(f).
- `k1 = (m1 − a0) / r1³`: Güneş çekimi eksi SRP. SRP, Güneş'in kütlesini "hafifletiyor".
- `ax = 2·vy + w·(x − k1·dx1 − k2·dx2) + g·ux`: Coriolis + w × (merkezkaç + Güneş + Dünya) + g × itki.

## 8. Jürinin sorabileceği sorular

- **"Neden Ay'ı eklemedin?"** Modelimizde ikinci cisim yalnızca Dünya. Ay ve gezegenlerin etkisi şimdilik sadece gürültü modelindeki küçük rastgele "unmodelled force" ile temsil ediliyor. Faz 7 bilinçli olarak en büyük iki sistematik model hatasıyla sınırlı tutuldu. Ay'ı ve tam efemeris modelini (JPL DE) eklemek gelecek çalışma. Bunu sınırlama olarak açıkça söyle.
- **"Gerçek görevler SRP'yi nasıl ele alıyor?"** Referans yörüngeyi SRP'li modelde tasarlıyorlar. Bizim Bias-LQR'nin yaptığı şey de bunun basit bir versiyonu: kaymış denge noktasını referans almak.
- **"LQI neden Bias-LQR'den pahalı?"** LQI tanım gereği L1'i tam tutuyor, bu yüzden SRP'nin tamamını yakıtla karşılamak zorunda. Bias-LQR ise yerçekiminin yardım ettiği kaymış noktaya kayabiliyor.
- **"DR-PPO neden SRP'yi kestiremiyor?"** Hafızası yok. Bkz. bölüm 4. Bu dürüst bir sınırlama.

## 9. Sonuçlar

Test seed'leri 100–119. Kaynaklar: `experiments/model_mismatch.csv`, `model_mismatch_classical.csv` (345 klasik tasarım), `model_mismatch_parts.csv`. Hiçbir kontrolcü hiçbir koşuda aracı kaybetmedi.

### Tutma (L1'den başla): Δv [m/s] (ortalama mesafe [km])

| k | LQR ρ=1 | Bias-LQR ρ=1 | PPO (3 seed) | DR-PPO (5 seed) | PPO × en iyi klasik | DR-PPO × en iyi klasik |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.09 (0.5) | 0.04 (0.6) | 0.11 (0.6) | 0.18 (7.9) | 1.99 ± 0.96 | 5.92 ± 1.87 |
| 0.5 | 1.28 (29) | 0.88 (9.2) | 1.12 (11) | 1.04 (2.1) | 1.31 ± 0.02 | 1.10 ± 0.03 |
| 1 | 2.55 (58) | 1.75 (18) | 2.20 (22) | 2.02 (1.4) | 1.26 ± 0.02 | 1.06 ± 0.02 |
| 2 | 5.10 (117) | 3.50 (37) | 4.40 (48) | 3.95 (0.9) | 1.28 ± 0.05 | 1.03 ± 0.01 |
| 4 | 10.26 (234) | 7.00 (74) | 8.99 (118) | 7.80 (1.9) | 1.34 ± 0.07 | 1.02 ± 0.01 |

### Ana bulgular, sade dille

1. **Eksantriklik neredeyse hiçbir şey yapmıyor.** k = 4'te bile (e = 0.067) LQR'nin tutma maliyeti 0.085 m/s, nominaldekiyle aynı. Sebebi: pulsating koordinatlarda L1 hâlâ denge noktası. Eksantrikliğin etkisi L1'den uzaklıkla orantılı, bizde bu uzaklık birkaç km. 10⁵ km'lik halo yörüngelerde ise önemli olurdu. **Bütün fark SRP'den geliyor.**
2. **SRP her şeyi pahalılaştırıyor.** k = 1'de düz LQR 58 km'ye kayıyor ve yılda 2.55 m/s harcıyor, bu nominalin ~30 katı. L1'e yakın durmak isteyen her kontrolcü SRP'yi karşılamak için yılda ~1.9 m/s ödemek zorunda.
3. **LQR'nin ödünleşimi tersine dönüyor.** Gevşek LQR, SRP'nin ittiği yöne, yani Güneş'ten dışarı kayıyor. L1 o eksende kararsız olduğu için yerçekimi de yükü artırıyor. Sonuçta gevşek LQR hem daha uzakta hem daha pahalı oluyor. Bias-LQR ise Güneş tarafına park ediyor: k = 1'de 222 km'de, yılda sadece 0.40 m/s.
4. **Nominal PPO dayanıklı ama pahalı.** SRP'yi hiç görmemiş olmasına rağmen hiçbir koşuyu kaybetmedi. Tutmada en iyi klasiğin 1.23–1.42 katı, toparlanmada 1.30–3.92 katı yakıt harcıyor.
5. **DR-PPO farkı neredeyse kapatıyor ama geçemiyor.** k > 0'da L1'i 0.8–3.0 km içinde tutuyor, bu ρ = 1 klasik tasarımların hepsinden sıkı. En iyi klasiğin 1.01–1.13 katı yakıtla (toparlanmada 1.06–1.48). **Ama:** (a) oranın 1'e yakın çıkmasının büyük kısmı ortak SRP maliyetinden geliyor, mutlak olarak hâlâ yılda 0.06–0.24 m/s fazla harcıyor. (b) **80 RL test durumunun 80'i de** hem daha ucuz hem en az onun kadar doğru bir klasik tasarım tarafından domine ediliyor. Bunların 73'ünde domine eden tasarım Bias-LQR.
6. **Az kalsın yanlış bir sonuç yazıyorduk.** Klasik taramayı sadece r_ref = 1000 km ile yapınca DR-PPO k = 4'te bütün klasiklerden daha doğru görünüyordu. Ama bu, klasik tarafın dar ayarlanmasından kaynaklanıyordu: r_ref = 10–100 km ile Bias-LQR 1.4 km'ye iniyor ve DR-PPO'yu geçiyor. Jüriye bunu anlat. "RL kazandı" demeden önce karşı tarafı da adil ayarlamak gerektiğini gösteriyor.
7. **Dayanıklılığın bedeli.** k = 0'da (SRP yokken) DR-PPO en iyi klasiğin 5.9 katı yakıt harcıyor ve L1'in 9.7 km Güneş tarafında duruyor. Hafızası olmadığı için SRP'nin var olup olmadığını bilemiyor, o yüzden "ortalama bir SRP var" diye sürekli itki veriyor. Bias-LQR'nin Kalman filtresi tam olarak bu problemi çözüyor. Doğal sonraki adım: hafızalı RL (frame stacking veya LSTM).

### Tek cümlelik özet (jüri için)

> Gerçek fizik modelden saptığında düz LQR bozuluyor ve domain randomization ile eğitilmiş PPO aradaki farkı neredeyse kapatıyor. Ama klasik kontrolün onlarca yıllık standart çözümleri (integral etki, Kalman ile bozucu kestirimi) eklendiğinde klasik taraf yine Pareto-optimal kalıyor.

### Neler tutmadı, nelere dikkat?

- DR-PPO'nun model seçimi k = 0, 1, 2, 4'te yapıldı. Testteki k = 0.5 eğitimde görülen aralığın içinde ama seçimde kullanılmadı.
- SRP'nin alan/kütle oranı ve C_r bir varsayım. Gerçek bir görev için değerler farklı olur. Sonuçlar k taramasıyla bu belirsizliği kapsıyor.
- "× en iyi klasik" oranı basamak fonksiyonuyla hesaplanıyor (interpolasyon yok), bu yüzden muhafazakâr. `lqr_ratio` (Faz 5 yöntemi) k > 0'da çoğunlukla NaN, çünkü LQR eğrisinin "sol kolu" tek noktaya iniyor (bkz. bulgu 3).
