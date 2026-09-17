# Beşinci Deney (Experiment 5): SIGReg’in Eyleme Dönüştürülebilir Yerel Ağırlık Yapısı (Actionable Local Weight Structure) Oluşturup Oluşturmadığının Testi

## Teknik terimler için okuma notu (Technical terminology note)

Bu belgede Türkçe açıklama korunurken teknik bir kavramın yerleşik İngilizce karşılığı ilk anlamlı kullanımında parantez içinde verilir. Kodda, makalelerde ve deney çıktılarında doğrudan İngilizce kullanılan `backbone`, `projector`, `checkpoint`, `batch`, `epoch`, `probe`, `commit`, `hash` ve `manifest` gibi sözcükler de ilk kullanımlarında açıklanır.

Hızlı başvuru için temel eşleştirmeler:


| Türkçe terim              | İngilizce karşılığı                |
| ------------------------- | ---------------------------------- |
| Temsil öğrenmesi          | Representation learning            |
| Temsil kaybı              | Representation loss                |
| Ağırlık imzası            | Weight signature                   |
| Yerel ağırlık yapısı      | Local weight structure             |
| Düzenleyici               | Regularizer                        |
| Hizalama kaybı            | Alignment loss                     |
| Parametre-temelli ceza    | Parameter-based penalty            |
| Veri artırma              | Data augmentation                  |
| Eşleştirilmiş deney       | Paired experiment                  |
| Özellik çıkarıcı          | Backbone / feature extractor       |
| Yansıtıcı başlık          | Projector / projection head        |
| Toplu normalleştirme      | Batch Normalization (BatchNorm/BN) |
| BN katlama                | BatchNorm folding / BN folding     |
| Doğrusal probe            | Linear probe                       |
| Çökme                     | Collapse                           |
| Boyutsal çökme            | Dimensional collapse               |
| Ağırlık matrisi spektrumu | Weight-matrix spectrum             |
| Etkin rank                | Effective rank                     |
| Kararlı rank              | Stable rank                        |
| Birincil uç nokta         | Primary endpoint                   |
| Karşıtlık                 | Contrast                           |
| Yanlışlama kontrolü       | Falsification control              |
| Kaldırma deneyi           | Ablation experiment                |
| İkame                     | Replacement / substitution         |
| Tohum                     | Random seed                        |
| Rastgele sayı akışı       | Random-number stream               |
| Gradyan normu             | Gradient norm                      |
| Güven aralığı             | Confidence interval                |
| Aşağı kalmama             | Non-inferiority                    |
| İş dizisi                 | Job array                          |
| Bağımlılık zinciri        | Dependency chain                   |


## 1. Deneyin amacı (Experiment objective)

Bu deneyin temel amacı, SIGReg’in yalnızca ağın çıktı dağılımını düzenleyen bir temsil kaybı (representation loss) olup olmadığını değil, aynı zamanda CNN ağırlıklarında tekrar üretilebilir (reproducible), ölçülebilir ve doğrudan kullanılabilir bir yapı bırakıp bırakmadığını sınamaktır.

Araştırmanın merkezindeki soru şudur:

> Aynı veri, mimari, başlangıç ağırlıkları, optimizasyon programı (optimization schedule) ve veri artırma akışı (data-augmentation pipeline) altında SIGReg; VICReg-benzeri moment düzenlemesinden (moment regularization) veya yalnızca hizalama kaybından (alignment loss) ayrılan bir ağırlık imzası (weight signature) üretir mi ve bu imza daha sonra veri örneği kullanmadan hesaplanan parametre-temelli bir ceza (parameter-based penalty) ile kısmen yeniden üretilebilir mi?

Buradaki “yerel” (local) sözcüğü tek bir 3×3 çekirdekle (kernel) sınırlı değildir. Deney üç farklı yerellik ölçeğini (locality scale) birbirinden ayırır:

1. Tek bir giriş–çıkış kanal çiftine ait 3×3 çekirdek.
2. Bir katmandaki bütün filtreleri içeren `C_out × (C_in k²)` ağırlık matrisi.
3. Konvolüsyon (convolution), toplu normalleştirme (Batch Normalization/BatchNorm), artık bağlantı (residual connection) ve doğrusal olmayan katmanların (nonlinear layers) oluşturduğu blok fonksiyonu (block function).

Amaç, mümkün olan en küçük ve hâlâ işe yarayan kapsamı bulmaktır. Tek çekirdek düzeyinde bir açıklama yetersizse katman veya blok düzeyine geçilir; sonuç zorla “tek filtreye ait” diye yorumlanmaz.

## 2. Neden önceki dört deney yeterli değil? (Why are the previous four experiments insufficient?)

İlk dört deney önemli ölçüm altyapısı ve güçlü gözlemler sağlamıştır; ancak beşinci deneyin nedensel sorusunu doğrudan yanıtlamaz:


| Önceki deney | Korunan bulgu                                                                                        | Nedensel sınırlama                                                                                                                            |
| ------------ | ---------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| Deney 1      | ImageNet ve LeJEPA filtre dağılımları geniş ölçekte birbirine yakındır                               | Eğitim verisi, başlangıç, hedef ve eğitim rotası eşit değildir                                                                                |
| Deney 2      | Artık farkların (residual differences) derinlik ve PCA yönü yapısı vardır                            | Deney 1’in yeniden analizidir; bağımsız tekrar (independent replication) değildir                                                             |
| Deney 3      | Son bloklarda güçlü farklar ve ham/BN-katlanmış (raw/BN-folded) ölçek sıralaması terslenmesi görülür | Rastgele başlangıçlı öz-denetimli öğrenme (self-supervised learning, SSL) ile denetimli ImageNet aktarımı birçok faktörü aynı anda değiştirir |
| Deney 4      | Derinlik örüntüsü harici PCA altında sürer ve kaba filtre şekli erken düzenlenir                     | Aynı seçilmiş model çifti ve tek SSL yörüngesi yeniden kullanılır                                                                             |


Özellikle üç sonuç yeni tasarımı zorunlu kılar:

- Ham ağırlık ölçeğinin BN katlamasından (BatchNorm folding) sonra tersine dönmesi, ham standart sapmanın (raw standard deviation) tek başına işlevsel güç (functional strength) göstergesi olmadığını gösterir.
- Filtre histogramının erken kararlı görünmesine rağmen doğrusal probe (linear probe) başarımının daha sonra artması, histogram yakınlığının öğrenmenin tamamlandığı anlamına gelmediğini gösterir.
- ResNet18’de filtrelerin büyük bölümü `layer4` içinde bulunduğu için havuzlanmış ölçüler (pooled metrics) erken katman değişimlerini gizleyebilir.

Bu nedenle Deney 5, farklı eğitim rotalarını (training trajectories) karşılaştırmak yerine aynı başlangıçtan çıkan eşleştirilmiş amaç fonksiyonlarını (matched objectives) karşılaştırır.

## 3. Sınanan hipotezler (Tested hypotheses)

### H1 — Tekrar üretilebilir ağırlık imzası (Reproducible weight signature)

Yalnızca düzenleyici (regularizer) değiştirilirken SIGReg kolunun önceden belirlenmiş (pre-specified) ağırlık istatistiğinde tutarlı bir değişim üretmesi beklenir. Değişim bağımsız eğitim tohumlarında (training seeds) görülmeli ve ham ölçek, BN katlama ve filtre yönü (filter direction) kontrollerinden sonra yorumlanmalıdır.

H1’i zayıflatan sonuçlar:

- Kollar arası farkın tohukolmlar arası değişkenlikten küçük olması.
- Farkın yalnızca keyfi ham ağırlık ölçeğinde görülmesi.
- Farkın özellik çıkarıcı (backbone) yerine yalnızca yansıtıcı başlık (projector/projection head) içinde bulunması.
- Keşif tohumlarında (discovery seeds) görülen etkinin yeni tohumlarda kaybolması.

### H2 — İmzanın müdahale edilebilir olması (Intervenable signature)

Bulunan yapı yalnızca sınıflandırıcı bir belirteç değil, mekanik olarak işe yarar olmalıdır. Yapıyı teşvik eden bir parametre cezasının temsil kalitesi veya kararlılık üzerinde seçici bir etkisi bulunmalı; yapı kaldırıldığında ya da kontrollü şekilde bozulduğunda davranış değişmelidir.

H2’yi zayıflatan sonuçlar:

- İstatistiği eşleştirmek temsil başarımını değiştirmiyorsa.
- Eş normlu rastgele pertürbasyon (equal-norm random perturbation) aynı etkiyi üretiyorsa.
- Marjinal filtre dağılımı (marginal filter distribution) korunurken bağlantı yapısı (connectivity structure) bozulduğunda ağ davranışı ciddi biçimde değişiyor, fakat aday ölçü bunu ayırt edemiyorsa.

### H3 — Kullanışlı parametre-temelli ikame (Useful parameter-based replacement)

Rastgele başlangıçtan (random initialization) itibaren hizalama ve yalnızca parametrelerden hesaplanan yerel ceza (local penalty) ile eğitim yapılır. Bu kol, SIGReg’e göre önceden belirlenen doğruluk marjı (accuracy margin) içinde kalmalı, çökme (collapse)/nümerik hata (numerical error) oranını artırmamalı ve ölçülmüş zaman veya bellek maliyetini düşürmelidir.

“Veriden bağımsız ceza” (data-independent penalty) şu anlama gelir: cezanın değeri ve gradyanı (gradient) hesaplanırken görüntü, özellik batch’i (feature batch) veya aktivasyon koşu istatistiği (activation running statistics) gerekmez. Ana hizalama kaybı yine veri kullanır. Ceza SIGReg modelleri incelenerek tasarlanırsa buna “veriyle bilgilendirilmiş tasarım, veriden bağımsız değerlendirme” (data-informed design, data-independent evaluation) denir.

## 4. Deneyin ana modeli ve ortak hizalama kaybı (Main model and shared alignment loss)

Model iki parçadan oluşur:

- Özellik çıkarıcı (backbone): CIFAR uyarlamalı ResNet18; 3×3, adım-1 giriş gövdesi (stride-1 stem) kullanır ve maksimum havuzlama (max-pooling) yoktur.
- Yansıtıcı başlık (projector/projection head): `512 → 2048 → 2048 → 64`; iki gizli blokta BatchNorm ve GELU, son katmanda doğrusal çıktı (linear output) vardır.

Bir görüntünün dört bağımsız global görünümü için:


y=f_\theta(x), \qquad z=g_\phi(y)



\bar z_i=\frac{1}{V}\sum_{v=1}^{V}z_i^{(v)}



L_{align}=\frac{1}{BVd}\sum_{i=1}^{B}\sum_{v=1}^{V}
\leftz_i^{(v)}-\bar z_i\right_2^2


Merkez üzerinde gradyan durdurma (`stop-gradient`) yoktur. Dört görünümün (views) tamamı aynı amaçta simetrik biçimde yer alır.

## 5. Deney 5A — Kayıp ve ölçüm mekanizmasının doğrulanması (Loss and measurement validation)

### 5.1 SIGReg kaynak sabitleme (Source pinning)

Yerel LeJEPA deposu (local repository) şu kaynak sürümüne (`commit`) sabitlenmiştir:

```text
c293d291ca87cd4fddee9d3fffe4e914c7272052
```

Ana çalıştırıcı (main runner), kaynak `commit` farklıysa veya LeJEPA kaynak dosyalarında kayda geçirilmemiş değişiklik (uncommitted change) varsa bilimsel koşuyu (scientific run) durdurur. `__pycache__` gibi üretilmiş dosyalar (generated files) kaynak değişikliği sayılmaz.

### 5.2 SIGReg sayısal tanımı (Numerical definition)

Ana yapılandırma:

- 1.024 rastgele normalize projeksiyon yönü (random normalized projection directions).
- Sınır: 5.
- 17 yamuk-integrasyon düğümü (trapezoidal integration nodes).
- Gauss ağırlığı (Gaussian weighting).
- Dilimler (slices) üzerinde ortalama.
- Görünümler (views) üzerinde ortalama.
- İstatistik hesaplarında tek duyarlıklı kayan nokta (`float32`).

Önemli denetim sonucu: sabitlenmiş yazar uygulaması (pinned author implementation) 17 düğümü `[0,5]` aralığında tutar ve negatif frekansları simetrik ağırlıklarla temsil eder. Bu nedenle etkin integral aralığı (effective integration interval) `[-5,5]` olmakla birlikte bellekte tam aralığa yayılmış toplam 17 düğüm yoktur. Bu ayrıntı `validate_experiment5_losses.py` çıktısında açıkça kaydedilir.

### 5.3 Sentetik kayıp kontrolleri (Synthetic loss checks)

SIGReg aşağıdaki sentetik gömmelerde (synthetic embeddings) bağımsız açık form uygulama (independent closed-form implementation) ile karşılaştırılır:

- Standart Gauss.
- Ortalaması kaydırılmış Gauss.
- Ölçeği değiştirilmiş Gauss.
- Rank eksiği bulunan Gauss (rank-deficient Gaussian).
- Sabit sıfır tensörü (constant-zero tensor).

Her durumda hem kayıp değeri (loss value) hem giriş gradyanı (input gradient) karşılaştırılır. Sonlu örnekli Gauss istatistiğinin (finite-sample Gaussian statistic) sıfır olması beklenmez. Sabit sıfır tensörü yüksek kayba sahip olduğu hâlde sıfır gradyanlı olabilir; bu beklenen bir kontrol sonucudur.

Tarihsel deneyle karşılaştırma için sınır 3 ve sınır 5 aynı örnek ve yönlerde ayrıca ölçülür.

### 5.4 Analitik doğrusal kontrol (Analytical linear control)

İki katmanlı, 16 boyutlu doğrusal bir ağ kullanılır:

- 10.000 eğitim çifti.
- 2.000 ayrılmış değerlendirme çifti.
- Beş tohum.
- 5.000 eniyileyici güncellemesi (optimizer updates).
- Koşul sayıları (condition numbers) 1, 10 ve 100 olan tersinir karıştırma matrisleri (invertible mixing matrices).
- Yöntemler: hizalama+SIGReg, hizalama+VC ve hizalama+Yumuşak Ortogonallik (Soft Orthogonality).

Ölçülen çıktılar:

- Çıktı kovaryansının (output covariance) birim matrise uzaklığı.
- Ortalama koordinat varyansı (mean coordinate variance).
- Bileşik doğrusal haritanın (composite linear map) tekil değerleri (singular values).
- Bileşik harita ile karıştırma matrisinin birlikte tekil değerleri.

Bu kontrol yalnızca yerel ortogonalliğin (local orthogonality) hangi giriş kovaryansı koşullarında çıktı beyazlatmaya (output whitening) yaklaşabildiğini sınar; görüntü temsil kalitesi (image-representation quality) kanıtı değildir.

## 6. Deney 5B — Eşleştirilmiş A/B/C keşif deneyi (Paired A/B/C discovery experiment)

### 6.1 Deney kolları (Experimental arms)


| Kol | Amaç                        | Rol                                                                                                 |
| --- | --------------------------- | --------------------------------------------------------------------------------------------------- |
| A   | `0.98 L_align`              | Anti-çökme düzenleyicisi (anti-collapse regularizer) olmayan negatif kontrol (negative control)     |
| B   | `0.98 L_align + 0.02 R_SIG` | Ana SIGReg referansı (reference arm)                                                                |
| C   | `0.98 L_align + α R_VC`     | Aynı mimari ve hizalama altında varyans/kovaryans karşılaştırıcısı (variance/covariance comparator) |


C kolu tam VICReg tekrarı (replication) değildir. Doğru adı “LeJEPA hizalaması + VICReg varyans/kovaryans terimleri”dir; ikinci bir değişmezlik terimi (invariance term) eklenmez.

### 6.2 VC cezası (VC penalty)

Her görünüm için örnek kovaryansı (sample covariance) `B-1` paydasıyla hesaplanır.

Varyans terimi:


L_{var}=\frac{1}{d}\sum_j \max\left(0,1-\sqrt{Var(z_j)+10^{-4}}\right)


Kovaryans terimi:


L_{cov}=\frac{1}{d}\sum_{i\neq j}Cov(z)_{ij}^{2}


Toplam karşılaştırıcı:


R_{VC}=25L_{var}+L_{cov}


Görünüm başına değerler ortalanır. `α`, ortak hizalama kaybına göre bu düzenleyicinin genel katsayı ölçeğini (coefficient scale) belirler.

### 6.3 Katsayı kalibrasyonu (Coefficient calibration)

Pilot tohumları (pilot seeds) 901 ve 902 yalnızca sayısal doğrulama ve ölçek kalibrasyonu (scale calibration) için kullanılır. İlk sabit mini-yığınlarda (`batch`) ağırlık güncellemesi yapılmadan şu normlar ölçülür:

- Backbone SIGReg gradyan normu (gradient norm).
- Projector SIGReg gradyan normu.
- Birleşik SIGReg gradyan normu.
- Aynı üç normun ölçeklenmemiş VC karşılığı.

`α₀`, `0.02 × median(||g_SIG||) / median(||g_VC||)` olarak belirlenir. Sonra şu komşuluklar eşit bütçeyle çalıştırılır:

- SIGReg: `0.02/3`, `0.02`, `0.06`.
- VC: `α₀/3`, `α₀`, `3α₀`.

Gradyan normu eşleştirme (gradient-norm matching) yalnızca ölçek kuralıdır; iki kaybın aynı yönde veya aynı mekanizmayla çalıştığının kanıtı değildir.

### 6.4 Veri bölünmesi (Data split)

CIFAR10 resmi eğitim kümesi şu şekilde sabitlenmiştir:

- 45.000 etiketsiz eğitim görüntüsü (unlabeled training images).
- 5.000 doğrulama görüntüsü (validation images).
- Her sınıfta tam 4.500 eğitim ve 500 doğrulama örneği.
- Bölme tohumu: `20260916`.

Doğrulama kümesinde ayrıca birbirinden ayrık iki sabit grup vardır:

- 2.048 temsil tanı görüntüsü.
- 2.048 kalibrasyon görüntüsü.

Ana bölme (`split`) dosyası:

```text
outputs/(5thEXP)rn18_cifar10_sigreg_structure/protocol/cifar10_split_indices.npz
```

Bölme özeti (`split` SHA-256 hash):

```text
c9863466901b807a4c31362a30c5971348e265edf41f0465dcab64d3e814ce45
```

CIFAR10 test kümesi (test set) önceki deneylerde kararları etkilediği için tarihsel olarak geliştirmeye maruz kalmış kabul edilir. Katsayı, kontrol noktası (`checkpoint`) veya aday ceza seçimi test kümesine göre yapılmaz.

### 6.5 Veri artırma (Data augmentation)

Her kaynak görüntü dört bağımsız 32×32 genel görünüm (global view) üretir. İşlem sırası kodda sabittir:

1. `RandomResizedCrop`, ölçek `[0.08, 1.0]`.
2. Yatay çevirme, olasılık `0.5`.
3. ColorJitter `(0.8, 0.8, 0.8, 0.2)`, uygulanma olasılığı `0.8`.
4. Gri tonlama, olasılık `0.2`.
5. Gaussian blur, olasılık `0.5`, kernel 3, sigma `[0.1, 2.0]`.
6. Solarize, olasılık `0.2`, eşik 128.
7. Tensör dönüşümü (tensor conversion) ve CIFAR10 normalizasyonu.

Her görünüm; deney tohumu, optimizer güncellemesi, kaynak görüntü kimliği ve görünüm numarasından türetilen sabit bir anahtarla (deterministic key) üretilir. Dolayısıyla SIGReg’in rastgele yön çekmesi daha sonraki görüntü artırmalarını değiştiremez.

### 6.6 Eğitim ayarları (Training configuration)


| Ayar                                                 | Değer                                            |
| ---------------------------------------------------- | ------------------------------------------------ |
| Mini-yığın (`batch`)                                 | 256 farklı kaynak görüntüsü                      |
| Görünüm sayısı                                       | 4; güncelleme başına 1.024 görüntü görünümü      |
| Toplam güncelleme                                    | 21.000                                           |
| Eşdeğer dönem (`epoch`)                              | 120                                              |
| Dönem başına tam mini-yığın (full batches per epoch) | 175; kalan 200 örnek o dönem için düşürülür      |
| Eniyileyici (`optimizer`)                            | AdamW                                            |
| Öğrenme oranı                                        | `0.002`                                          |
| Isınma (`warmup`)                                    | 175 güncelleme                                   |
| Son öğrenme oranı                                    | Kosinüs programıyla (cosine schedule) `2×10⁻⁶`   |
| Beta                                                 | `(0.9, 0.999)`                                   |
| Epsilon                                              | `10⁻⁸`                                           |
| Ağırlık çürümesi                                     | Conv/Linear matrislerinde `5×10⁻⁴`               |
| Sapma (`bias`) ve norm afin parametre çürümesi       | 0                                                |
| Backbone hassasiyeti                                 | CUDA bfloat16 otomatik tür dönüşümü (`autocast`) |
| İstatistik hassasiyeti                               | `float32`; spektrumlarda gerektiğinde `float64`  |


### 6.7 Tohumlar ve eşleştirme (Seeds and pairing)

Keşif tohumları:

```text
13, 17, 23, 31, 43
```

Taze doğrulama tohumları:

```text
101, 103, 107, 109, 113
```

Her ana rastgelelik tohumundan (random seed) beş ayrı rastgele sayı akışı (random-number stream) türetilir:

- Model başlangıcı.
- Veri sırası.
- Veri artırma.
- SIGReg projeksiyon yönleri.
- Probe eğitimi (probe training).

Aynı tohumun A/B/C kolları backbone ve projector başlangıç durumlarının birebir aynı SHA-256 özetine (`hash`) sahip olmalıdır. Toplama betiği (aggregation script) bu eşleşmeyi sonuç üretmeden önce zorunlu olarak doğrular.

### 6.8 Kontrol noktası ve devam politikası (Checkpoint and resume policy)

Checkpoint güncellemeleri:

```text
0, 1, 10, 50,
175, 350, 875, 1750,
3500, 5250, 10500, 15750, 21000
```

Tam temsil tanıları:

```text
0, 175, 875, 1750, 5250, 10500, 21000
```

Her kontrol noktası (`checkpoint`) şunları içerir:

- Backbone ve projector ağırlıkları.
- BatchNorm tamponları (`buffers`).
- Eniyileyici durumu (`optimizer state`).
- SIGReg yön sayacı.
- Python, NumPy, Torch CPU ve CUDA rastgele sayı üreteci durumları (RNG states).
- Akış tohumları.
- Yapılandırma ve bölme özetleri (`config` and `split` hashes).

Devam (`resume`) yalnızca açıkça verilen tam kontrol noktasıyla yapılır. “En iyi” doğrulama kontrol noktası seçilmez (`best-checkpoint selection`); ana sonuç sabit son güncellemedir.

## 7. Ağırlık ölçümleri (Weight measurements)

### 7.1 Üç parametre görünümü (Three parameter views)

Her ilgili matris üç biçimde incelenir:

1. Ham ağırlık matrisi (raw weight matrix).
2. Çıkarım modu BN-katlanmış matris (inference-mode BN-folded matrix): `γ / sqrt(running_var + 10⁻⁵)`.
3. Her satırı birim norma getirilmiş filtre yönü (unit-normalized filter direction); normlar ayrıca raporlanır.

Giriş gövdesi (`stem`), 1×1 kısa yollar (`shortcuts`), BatchNorm parametreleri ve projector matrisleri işlevsel açıklamada tutulur. `Stem`, tarihsel şekillerle uyum için ayrı etiketlenir.

### 7.2 Katman matris spektrumu (Layer-matrix spectrum)

Bir ağırlık matrisi `A` için tekil değerler (singular values) `σ_j` ve enerji payları (energy shares):


q_j=\frac{\sigma_j^2}{\sum_k\sigma_k^2}


Ağırlık-enerjisi etkin rank (weight-energy effective rank):


r_{eff}=\exp\left(-\sum_jq_j\log q_j\right)


Normalize etkin rank (normalized effective rank):


\frac{r_{eff}}{\min(m,n)}


Kararlı rank (stable rank):


r_{stable}=\frac{A_F^2}{A_2^2}


Matrisler merkezlenmez (uncentered matrices). Merkezleme (centering) farklı bir sorudur ve ana uç noktaya sessizce eklenmez.

### 7.3 Birincil ağırlık uç noktası (Primary weight endpoint)

Önceden sabitlenen birincil ölçü:

> Güncelleme 5.250’de, yani eşdeğer dönem (`epoch`) 30’da, ResNet18’in artık blokları (residual blocks) içindeki 16 adet 3×3 `conv1/conv2` matrisinin ham ve merkezlenmemiş normalize ağırlık-enerjisi etkin rank değerlerinin eşit-katman ortalaması (equal-layer mean).

Birincil karşıtlık (primary contrast) `B-C`’dir. `B-A` ikincil karşıtlıktır (secondary contrast); A kolunun tamamen çökmesi farkı tek başına açıklayabilir.

## 8. Temsil ölçümleri (Representation measurements)

Sabit 2.048 doğrulama görüntüsünden şu noktalar çıkarılır:

- `stage1`, `stage2`, `stage3`, `stage4` genel ortalama havuzlama çıktıları (global-average-pooling/GAP outputs).
- Son backbone özelliği `y`.
- Projector birinci ve ikinci Linear çıkışı.
- Projector birinci ve ikinci BatchNorm çıkışı.
- Projector birinci ve ikinci GELU çıkışı.
- Son `z`.

Ölçüler:

- Özellik kovaryans spektrumu (feature-covariance spectrum).
- Toplam varyans ve koordinat standart sapmaları.
- Kovaryans-enerjisi etkin rank (covariance-energy effective rank).
- RankMe; tekil değerleri karelemeden normalize eden ayrı rank tanımı.
- Aktif ve pozitif birim oranları (active-unit and positive-unit fractions).
- Dört sabit görünüm arasında hizalama hatası (alignment error).
- Backbone ve projector için ayrı çökme görünümü (collapse view).

### 8.1 LiDAR temsil rank’i (LiDAR representation rank)

Her temiz görüntü bir vekil sınıf (surrogate class), görüntünün dört artırılmış görünümü bu sınıfın örnekleridir. Sınıflar-arası kovaryans (between-class covariance) `Σ_b`, sınıf-içi kovaryans (within-class covariance) `Σ_w` ile:


\Sigma_{LiDAR}=\Sigma_w^{-1/2}\Sigma_b\Sigma_w^{-1/2}


Bu matrisin özdeğerlerine (eigenvalues) etkin rank uygulanır. Sınıf-içi kovaryansa `10⁻⁶ I` düzenleme terimi (regularization term) eklenir. Deneyde görüntü kimlikleri ve görünüm artırmaları tüm yöntemlerde aynıdır.

### 8.2 Son-z Gauss uygunluğu (Final-z Gaussian fit)

Son `z`, eğitimde kullanılmayan 4.096 yön ve 33 frekans noktasıyla standart çok değişkenli Gauss (`N(0,I)`) karşısında ölçülür. Aynı örnek sayısı ve boyutta üretilen gerçek Gauss örneği de aynı yönlerde ölçülerek sonlu örnek hatası (finite-sample error) için bağlam sağlar.

Düşük sonlu-yön istatistiği (finite-direction statistic) çok değişkenli Gauss olduğunun ispatı değildir. Ortalama, ölçek, kovaryans anizotropisi (covariance anisotropy) ve şekil ayrı ayrı okunmalıdır.

## 9. Doğrusal probe (Linear probe)

Backbone değerlendirme modunda (`eval` mode) tamamen dondurulur (frozen). Üzerine yalnızca `Linear(512,10)` sınıflandırma başlığı (classification head) eğitilir.


| Ayar                      | Değer                                 |
| ------------------------- | ------------------------------------- |
| Dönem (`epoch`)           | 50                                    |
| Eniyileyici (`optimizer`) | AdamW                                 |
| Öğrenme oranı             | `0.001`                               |
| Ağırlık çürümesi          | `10⁻⁶`                                |
| Mini-yığın (`batch`)      | 256                                   |
| Program                   | Kosinüs zamanlaması (cosine schedule) |


45.000 eğitim havuzunda sınıf-dengeli (class-balanced) ve iç içe (nested) üç etiket oranı kullanılır:

- %1: 450 görüntü.
- %10: 4.500 görüntü; birincil performans uç noktası.
- %100: 45.000 görüntü.

Üç sabit etiket çekilişi (fixed label draws) vardır. Test başarımı (test accuracy) yalnızca son sabit dönem sonrasında raporlanır; model veya katsayı seçmez.

## 10. Ucuz yanlışlama kontrolleri (Cheap falsification controls)

### 10.1 3×3 kanal-çifti çekirdeklerini karıştırma (Channel-pair kernel shuffling)

Seçilen katmandaki bütün 3×3 kanal-çifti çekirdekleri yer değiştirir (shuffle). Katmanın ham marjinal çekirdek popülasyonu (raw marginal kernel population) bit düzeyinde aynıdır; fakat hangi giriş kanalının hangi çıkış kanalına bağlandığı değişir.

Eğer marjinal tanımlayıcılar (marginal descriptors) aynen korunurken özellikler ciddi biçimde değişirse, marjinal filtre dağılımı fonksiyonu tek başına belirlemiyor demektir.

### 10.2 Geçerli koordineli kanal permütasyonu (Valid coordinated channel permutation)

Bir artık blokta (residual block):

- `conv1` çıkış satırları permüte edilir (permuted).
- Eşleşen `bn1` gamma, beta, hareketli ortalama (`running mean`) ve hareketli varyans (`running variance`) değerleri aynı şekilde permüte edilir.
- `conv2` giriş sütunları aynı permütasyonla dönüştürülür.

Bu dönüşüm blok fonksiyonunu korur (function-preserving transformation). Sayısal çıktı değişiminin `10⁻⁴` altında olması zorunludur. Permütasyona değişmez (permutation-invariant) olması beklenen spektrum ve marjinal ölçüler de değişmemelidir.

## 11. Deney 5C — Yerel cezanın SIGReg yönüyle ilişkisi (Relation between the local penalty and the SIGReg direction)

Bu aşama A/B/C keşif sonuçları geldikten sonra açılır. Sabit B kontrol noktalarında (`checkpoints`) toplam eğitim gradyanı (total training gradient) yerine yalnızca SIGReg gradyanı ayrılır:


g_{\ell,b}^{SIG}=\nabla_{W_\ell}R_{SIG}(Z_b)


Dönem (`epoch`) 1, 5, 30 ve 120’de sekiz bağımsız mini-yığın (`batch`) kullanılır. Aday parametre cezası için katman bazında (per-layer) şunlar ölçülür:

- SIGReg gradyanı ile kosinüs benzerliği (cosine similarity).
- Göreli norm (relative norm).
- Tutulan mini-yığında SIGReg kaybının birinci dereceden tahmini değişimi (first-order predicted change).
- Mini-yığın kaynaklı değişkenlik (batch-induced variability).
- Yeni SIGReg projeksiyon yönlerinden kaynaklanan değişkenlik.
- Hizalama ve VC gradyanlarıyla benzerlik (gradient similarity).

Yaklaşık sıfır normlu katmanda kosinüs yorumlanmaz. AdamW momentum ve önkoşullama (preconditioning) kullandığı için ham gradyan benzerliği tek başına yörünge eşitliği (trajectory equivalence) anlamına gelmez.

### İlk yerel temel yöntem: Yumuşak Ortogonallik (First local baseline: Soft Orthogonality)

`A ∈ R^{m×n}` için küçük Gram tarafı (smaller Gram side) seçilir:


G=AA^T \quad (m\leq n), \qquad G=A^TA \quad (m>n)



R_{SO}=\frac{G-I_r_F^2}{r}, \qquad r=\min(m,n)


Dikdörtgen matrisin büyük tarafında olanaksız birim matris hedefi kurulmaz. Bu ceza tam konvolüsyon operatörü ortogonalliği (full convolution-operator orthogonality) olarak yorumlanmaz.

Yeni bir bant cezası (band penalty) ancak keşif aşamasında tekrar üretilebilir bir istatistik bulunduğunda tanımlanacaktır. Sınırlar, katsayılar ve katman kapsamı (layer scope) doğrulama tohumları görülmeden dondurulacaktır.

## 12. Deney 5D — Kaldırma, ikame ve zaman müdahaleleri (Ablation, replacement, and timing interventions)

Dönem 30’daki B kontrol noktasından tam eğitim durumu (full training state) klonlanır:


| Dal     | Kalan amaç                                            | Soru                                                                      |
| ------- | ----------------------------------------------------- | ------------------------------------------------------------------------- |
| B→B     | SIGReg ile devam                                      | Eşleştirilmiş referans (matched reference)                                |
| B→A     | SIGReg kaldırılır (ablation)                          | Öğrenilmiş durumda düzenleyici hâlâ gerekli mi?                           |
| B→Local | Sabit yerel ceza                                      | Yerel ceza öğrenilmiş durumu koruyabilir mi?                              |
| B→VC    | İsteğe bağlı moment düzenleme (moment regularization) | Geç dönemde yüksek mertebe eşleştirme (higher-order matching) gerekli mi? |


Eniyileyici durumu (`optimizer state`), öğrenme oranı programı (learning-rate schedule), veri sırası ve artırma akışları korunur. Yalnızca bir dalda eniyileyici sıfırlanmaz.

Sıcak başlangıç (warm start) başarısı yalnızca “durumu sürdürme” (state maintenance) kanıtıdır. Tam ikame (full replacement) için rastgele başlangıçlı kollar gerekir:


| Kol | Başlangıç ve amaç                                                                  |
| --- | ---------------------------------------------------------------------------------- |
| D   | Rastgele başlangıç + hizalama + yerel ceza                                         |
| E   | Rastgele başlangıç + hizalama + SIGReg + yerel ceza                                |
| F   | Rastgele başlangıç + hizalama + 256 dilimli ekonomik SIGReg (reduced-slice SIGReg) |
| G   | D farklı bir aday ise yayınlanmış SO/SRIP temel yöntemi (`baseline`)               |


## 13. İstatistiksel kararlar (Statistical decisions)

Bağımsız deney birimi (independent experimental unit) eğitim koşusudur. Filtreler, görüntüler, katmanlar, kontrol noktaları veya aynı ebeveynden dallanan devam koşuları yeni bağımsız tekrar (independent replicate) sayılmaz.

Ana raporlama:

- Her tohum ayrı gösterilir.
- Eşleştirilmiş `B-C` etkisi (paired effect) hesaplanır.
- %95 eşleştirilmiş güven aralığı (paired confidence interval) verilir.
- Tam iki yönlü işaret-çevirme testi (exact two-sided sign-flip test) sonucu raporlanır.
- Katman bazlı çoklu testler (multiple tests) ya düzeltilir ya da betimsel olarak işaretlenir.

Beş eşleştirilmiş koşuda tam iki yönlü işaret-çevirme testinin en küçük mümkün p-değeri (`p-value`) `2/32 = 0.0625`’tir. Bu nedenle beş tohum kesin istatistiksel anlamlılık (statistical significance) iddiası için değil, etki ve varyans keşfi için kullanılır.

Yerel ikame aşamasında birincil performans ölçüsü %10 etiketli doğrusal probe’dur. Örnek pratik aşağı kalmama marjı (non-inferiority margin):


D-B > -1 \text{ yüzde puanı}


Aşağı kalmama (non-inferiority) için güven aralığının alt sınırı `-1` puanın üzerinde olmalıdır. “Anlamlı fark yok” ifadesi eşdeğerlik (equivalence) kanıtı değildir.

## 14. Çökme ve hata politikası (Collapse and failure policy)

Tam ama sonlu çökme koşusu (finite collapsed run) durdurulmaz ve sonuç setinden çıkarılmaz.

Tanı bayrağı (diagnostic flag):

```text
son-z ortalama koordinat varyansı < 1e-4
ve bu durum art arda üç log kontrolünde sürüyor
```

Bu bir inceleme bayrağıdır (review flag); otomatik dışlama (automatic exclusion) değildir.

Şunlar ayrı sınıflandırılır:

- Tam temsil çökmesi (complete representation collapse).
- Boyutsal çökme (dimensional collapse).
- Çökme olmadan düşük doğrusal probe başarımı.
- Sonlu olmayan (`non-finite`) kayıp veya gradyan.
- CUDA/cuDNN gibi altyapı arızası (infrastructure failure).

Sonlu olmayan aritmetik (non-finite arithmetic) anında durdurulur ve `failure.json` yazılır. Yalnızca bir kola gradyan kırpma (gradient clipping) veya kurtarma mantığı (recovery logic) eklenmez.

## 15. Hesaplama maliyeti (Compute cost)

Her yöntem için aynı donanım ve batch altında ölçülecekler:

- Güncelleme başına milisaniye.
- Saniye başına kaynak görüntüsü.
- Tepe cihaz belleği (peak device memory).
- Düzenleyici ileri/geri geçiş süresi (regularizer forward/backward time).
- En az 200 güncellemelik ısınma (warm-up) sonrası zaman blokları.
- Önceden belirlenen doğrulama hedefi için geçen süre (time to target).

Bir cezanın “yerel” veya “ağırlık-temelli” (weight-based) olması onu otomatik olarak ucuz yapmaz. Özellikle 64 boyutlu çıktı kovaryansı küçükken büyük katman Gram matrisleri SIGReg’den pahalı olabilir. Hızlanma iddiası yalnızca ölçülmüş uçtan uca zamanla (measured end-to-end time) yapılır.

## 16. Dizin yapısı ve denetlenebilirlik

Ana çıktı kökü:

```text
outputs/(5thEXP)rn18_cifar10_sigreg_structure/
```

Önemli alt dizinler:

```text
protocol/                 split dosyası ve hash manifesti
validation/               SIGReg kayıp eşlik kontrolleri
linear_toy/               analitik doğrusal deney
pilot/                    katsayı ledger'ı ve hassasiyet koşuları
discovery/arm_A|B|C/      ana keşif koşuları
aggregate_discovery/      tohum düzeyi eşleştirilmiş sonuçlar
existing_checkpoint_falsifications/
```

Her bilimsel koşu manifestinde şu bilgiler bulunur:

- Tam yapılandırma ve SHA-256.
- Kaynak commit ve dirty durumu.
- Paket/Python/CUDA/cuDNN sürümleri.
- Split dosyası hash’i.
- RNG akış politikası ve tohumları.
- Başlangıç model durum hash’leri.
- Optimizer parametre grupları.
- Komut satırı.
- Katsayı ledger yolu ve hash’i.

## 17. Çalıştırma sırası

Önerilen bağımlı sıra:

1. SIGReg eşlik doğrulaması ve birim testler.
2. Var olan checkpoint yanlışlama kontrolleri.
3. 45 hücrelik doğrusal toy deneyi.
4. Pilot katsayı kalibrasyonu.
5. İki tohumlu B/C katsayı hassasiyet ızgarası.
6. İlk üç keşif tohumunda dokuz A/B/C koşusu.
7. Epoch 0/1/5/10/30/60/120 ağırlık ve temsil tanıları.
8. Son checkpoint doğrusal probe’ları.
9. Tohum düzeyi birincil `B-C` toplaması.
10. Keşif kanıtına göre Deney 5C/5D adayının dondurulması.
11. Kalan keşif ve daha sonra taze doğrulama tohumları.

## 18. 16 Eylül 2026 Slurm gönderim kaydı

Gönderilen zincir:


| İş                             | Job ID    | Bağımlılık        | Dizi sınırı |
| ------------------------------ | --------- | ----------------- | ----------- |
| SIGReg doğrulama               | `1565141` | Yok               | Tek iş      |
| Var olan checkpoint yanlışlama | `1565142` | `afterok:1565141` | Tek iş      |
| Doğrusal toy                   | `1565143` | `afterok:1565141` | `0-44%4`    |
| Katsayı kalibrasyonu           | `1565144` | `afterok:1565141` | Tek iş      |
| Pilot hassasiyet ızgarası      | `1565145` | `afterok:1565144` | `0-11%3`    |
| A/B/C ilk üç tohum taraması    | `1565146` | `afterok:1565144` | `0-8%3`     |
| Tam tanılar                    | `1565147` | `afterok:1565146` | `0-62%4`    |
| Son checkpoint probe’ları      | `1565148` | `afterok:1565146` | `0-8%3`     |
| Tohum düzeyi toplama           | `1565149` | `afterok:1565147` | Tek iş      |


Bu zincir, dizi elemanları dahil toplam 142 Slurm görevi tanımlar. Toplama adımı ayrıca yeniden kullanılabilir `jobs/aggregate_experiment5.sbatch` betiğine kaydedilmiştir.

Dizi sonundaki `%N`, aynı anda en fazla `N` alt işin GPU istemesine izin verir. Bu sınır, kümeyi bir anda onlarca GPU işiyle doldurmadan deney zincirinin otomatik ilerlemesini sağlar.

İlk gönderimde kümenin `kolyoz-cuda` için uyguladığı iki yerel politika görüldü ve iş dosyalarına yansıtıldı:

- CPU sayısı 16 veya 16’nın katı olmalıdır.
- Her 16 CPU için bir GPU istenmelidir.

İlk kontrol anında doğrulama işi ilişki-grubu CPU kotası nedeniyle `PENDING (AssocGrpCpuLimit)` durumundaydı; diğer işler doğru biçimde bağımlılık bekliyordu. Bu durum kod hatası değil, geçici küme kaynak/kota beklemesidir. Gerçek ilerleme, `squeue`, `sacct` ve `logs/exp5-*` dosyaları üzerinden izlenmelidir.

## 19. Başarı durumlarının doğru yorumu


| Sonuç                                                       | Savunulabilir iddia                                                        |
| ----------------------------------------------------------- | -------------------------------------------------------------------------- |
| B-C ağırlık farkı tekrar eder, müdahale işe yaramaz         | Düzenleyiciyle ilişkili tekrar üretilebilir parametre yapısı               |
| B→Local çalışır, D başarısız olur                           | Yerel ceza SIGReg ile öğrenilmiş durumu koruyabilir                        |
| E çalışır, D başarısız olur                                 | Yerel ceza ikame değil tamamlayıcıdır                                      |
| D geliştirme ortamında noninferior ve kararlıdır            | Bu ortam içinde ampirik ikame                                              |
| D yeni veri kümesine aktarılır ve ölçülmüş maliyeti düşürür | Aktarılabilir parametre-temelli ikame için kanıt                           |
| D başarımı tutturur ama SIGReg temsil ölçülerini tutturmaz  | Başarılı alternatif mekanizma; mekanik eşdeğerlik değil                    |
| Marjinal imza yok, operatör/aktivasyon farkı var            | Hipotez daha büyük yapısal ölçeğe taşınmalıdır                             |
| Hiçbir yararlı imza kontrollerden geçmez                    | Sınanan imza ailesi başarısızdır; bütün olası parametre açıklamaları değil |


Bu deneyin değerli sonucu yalnızca “yeni bir düzenleyici bulmak” değildir. SIGReg’in ağırlıklar, ara temsiller ve son görev davranışı arasındaki etkisinin hangi koşullarda yerel bir parametre kısıtıyla yaklaşıklandığını ve hangi koşullarda bunun mümkün olmadığını güvenilir biçimde göstermektir.