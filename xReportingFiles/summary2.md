# Dördüncü Deneyler ve Küresel PCA ile Üçüncü Deneyin Yeniden Analizi

Bu belge, CNN Filter DB projesinde yürütülen dördüncü deneylerin teknik denetimini, yeniden çalıştırılan analizleri, elde edilen sonuçları ve sonuçların bilimsel yorumunu ayrıntılı biçimde açıklar. Belgenin kapsamı yalnızca son rakamları vermek değildir. Hangi checkpoint'in neden kullanıldığı, küresel PCA tabanının nasıl üretildiği, eşiklerin hesapta tam olarak nerede devreye girdiği, hangi sonuçların PCA tabanından bağımsız olduğu ve hangi yorumların tek-seed/single-run sınırlaması nedeniyle ihtiyatlı tutulması gerektiği de açıklanmaktadır.

Bu aşamada yeni model eğitimi yapılmamıştır. Üçüncü deneyin şekilleri ve dördüncü deneyin analizleri mevcut checkpoint'lerden yeniden üretilmiştir. Tek istisna, daha önce kaybolmuş olan seçili ImageNet epoch-1 checkpoint'ini geri kazanmak için daha önce tamamlanmış deterministik temel-çizgi yeniden üretimidir; bu belgenin kapsadığı son analiz işleri tekrar eğitim yapmamıştır.

## 1. Kısa sonuç

Teknik denetimin sonucu genel olarak olumludur:

- Eşik duyarlılığı analizi doğru iki checkpoint'i, doğru 16 residual-block `3×3` katmanını ve her metrik ailesinde yedi ayrı eşik kullanmaktadır.
- Entropi ve sparsity hesapları küresel PCA tabanına bağlı değildir. Bu iki ölçüm ham katman filtreleri üzerinde yerel olarak hesaplanmaktadır.
- Eski varsayılan eşikleri yeniden üreten `160/160` regresyon kontrolünün tamamı geçmiştir.
- Küresel PCA tabanı yayımlanmış CNN Filter DB dosyasındaki `1.464.797.156` filtrenin tamamından hesaplanmıştır; kısmi örnekleme veya yalnızca iki ResNet18 modeline fit söz konusu değildir.
- PCA kovaryans hesabı yaklaşık/mini-batch PCA değildir. Dokuz boyutlu tam örnek kovaryansı, parça istatistiklerinin matematiksel olarak tam birleştirilmesiyle elde edilmiştir.
- PCA bileşenleri ortonormaldir; açıklanan varyans oranları `1.0` toplamına sahiptir; kovaryans ile özdeğer/özvektör ayrışımı tutarlıdır.
- Makalenin ekindeki merkezleme vektörüyle hesaplanan vektör arasında en fazla `0.00105831` mutlak fark vardır. Bu küçük fark gizlenmemiş, artifact ve tüketici provenance kayıtlarına uyarı olarak yazılmıştır. Kullanıcının bu büyüklükteki farkı kabul etmesi üzerine fark engelleyici değil, belgelenmiş bir sınırlama olarak değerlendirilmiştir.
- ImageNet epoch 1 ile LeJEPA epoch 110 arasındaki küresel drift, iki-model yerel PCA tabanında `0.128806`, yeni tam-veri küresel tabanda `0.133022` bulunmuştur. Mutlak değişim küçüktür (`+0.004216`, yaklaşık `%3,27`). Ana sonuç, yani farkın özellikle son residual blok ve bazı erken layer1 katmanlarında yoğunlaşması, taban değişiminden sonra korunmuştur.
- PCA'dan bağımsız ham ağırlık, BN-folded ağırlık, normalize filtre şekli ve kernel descriptor tabloları eski üçüncü deney sonuçlarıyla sayısal olarak birebir aynıdır. Bu, yeniden üretilen ImageNet epoch-1 checkpoint'inin analiz açısından doğru checkpoint olduğunu ayrıca doğrular.
- Üçüncü deneyin tam şekil paketi yeni küresel PCA tabanıyla ayrı bir klasöre yeniden üretilmiş; heatmap'ler korunurken karşılık gelen chart sürümleri eklenmiştir.
- Dördüncü deneyin dört eşik heatmap'i korunmuş ve her biri için yedi ayrı ImageNet–LeJEPA line-chart içeren alt klasör oluşturulmuştur.
- Yalnızca LeJEPA kullanan eğitim-dinamiği bölümü, epoch `1, 10, 20, ..., 120` olmak üzere tam 13 checkpoint üzerinde tamamlanmıştır. Global filtre-dağılımı değişiminin büyük kısmı epoch 20–30'a kadar oluşurken probe doğruluğu daha sonraki epoch'larda da artmaya devam etmektedir.
- LeJEPA dinamiği için 13 checkpoint'in SHA256 doğrulamaları, sabit histogram sınırlarının bağımsız yeniden kurulması, `13x13` matris simetrisi, bileşen-katkı toplamları ve 25 görselin okunabilirliği dahil bütün zorunlu kontroller geçmiştir. Tek `False` kontrol, kabul edilmiş makale-ortalaması farkını görünür tutan `warning` satırıdır.

Buradan çıkarılabilecek en güçlü genel yorum şudur: sonuçlar yalnızca iki modele fit edilen bir PCA koordinat sisteminin ürünü değildir. Küresel ve modelden bağımsız CNN Filter DB tabanına geçildiğinde ana derinlik deseni ve baskın fark bölgeleri korunmaktadır. Buna karşılık, bileşen numarası ve işaretine bağlı çok ayrıntılı yorumlar taban seçimine daha duyarlıdır ve evrensel semantik yönler gibi sunulmamalıdır.

## 2. Denetlenen deney kapsamı

Dördüncü deney üç bölüme ayrılmıştır:

1. ImageNet epoch 1 ve LeJEPA epoch 110 için sparsity/entropy eşik duyarlılığı.
2. Aynı eşleşmiş çiftin, yalnızca bu iki modele fit edilen eski PCA yerine tam CNN Filter DB küresel PCA tabanında yeniden analizi.
3. Yalnızca LeJEPA SSL omurgasının epoch `1, 10, 20, ..., 120` checkpoint'leri üzerinden eğitim boyunca filtre değişimi.

Buna ek olarak kullanıcı talebiyle üçüncü deneyin şekil üretimi, hiçbir model yeniden eğitilmeden, yeni küresel PCA tabanıyla ayrıca yeniden çalıştırılmıştır. Üçüncü deneyin seçili karşılaştırması değişmemiştir:

| Model | Checkpoint | Değerlendirme doğruluğu |
|---|---:|---:|
| ImageNet1K ön-eğitimli, CIFAR10 fine-tune RN18 | epoch 1 | `0.9125` |
| LeJEPA CIFAR10 SSL RN18 + dondurulmuş linear probe | SSL epoch 110 | `0.8913` |
| Mutlak fark |  | `0.0212` |

Bu doğruluklar filtre analizinin hedef değişkeni değildir. Yalnızca çok büyük performans farkına sahip iki modeli karşılaştırmamak için checkpoint eşleştirmesinde kullanılmıştır.

## 3. İş geçmişi ve iptal edilen işlerin anlamı

İptal veya `FAILED` görünen her Slurm işi bilimsel başarısızlık anlamına gelmemektedir. İş geçmişi şu şekilde yorumlanmalıdır:

| İş | Durum | Bilimsel anlamı |
|---|---|---|
| `1460251` | iptal | Yanlış `batch_size=256` ile başlatılmış ImageNet yeniden üretimiydi; epoch 16'dan sonra durduruldu ve `*_batch256_invalid` altında açıkça geçersiz olarak ayrıldı. |
| `1460256` | `FAILED` | Bütün `1.464.797.156` filtreyi işledi. Yalnızca makaledeki ortalamaya karşı aşırı sıkı `5e-5` son-kontrol toleransı geçmediği için artifact yazmadan çıktı. Tam sufficient-statistics checkpoint'i sağlam kaldı. |
| `1460257` | tamamlandı | Doğru `batch_size=128` ile 20/20 ImageNet checkpoint'ini yeniden üretti. Her epoch'un train/test loss, train/test accuracy ve learning-rate değerleri eski üçüncü deney CSV'siyle birebir eşleşti; yalnızca dosya yolu ve duvar-süresi alanlarının değişmesi beklenir. |
| `1460263` | tamamlandı | İlk eşik analizi; `160/160` regresyon kontrolü geçti. |
| `1460264` | iptal | `afterok:1460256` bağımlılığı karşılanmadığı için hiç başlamadı; dolayısıyla bozuk sonuç üretmedi. |
| `1470891` | tamamlandı | Dördüncü deney, küresel PCA tabanlı ImageNet–LeJEPA çift analizi ve PCA-bağımsız kontroller. |
| `1470892` | tamamlandı | Üçüncü deneyin yeni küresel PCA tabanıyla, yalnızca şekil/istatistik üretimi için yeniden çalıştırılması. |
| `1470893` | tamamlandı | Eşik heatmap alt klasörleri ve line-chart sürümleri dahil güncellenmiş eşik analizi. |
| `1470905` | iptal | Kuyruk düzenlemesi sırasında hiç başlamadan iptal edilen yinelenmiş eşik-şekil işidir; bilimsel çıktı üretmedi. |
| `1470908` | iptal | Son line-chart ölçek ayarını taşıyan yinelenmiş iş 12 saniye sonra durduruldu; aynı kesin iş gövdesi yerel olarak tamamlandı ve bütün çıktıları yeniden doğrulandı. |
| `1470909` | tamamlandı | Yalnızca mevcut LeJEPA checkpoint'lerini kullanan 13-checkpoint eğitim-dinamiği analizi; yeni eğitim yapılmadı. |

Yanlış batch-size koşusunun dosyaları korunmuştur; fakat hiçbir nihai analiz bunları kullanmamaktadır. Kabul edilen ImageNet kaynak yolu yalnızca şudur:

```text
outputs/(4thEXP)rn18_cifar10_filter_dynamics/
  regenerated_imagenet_baseline/checkpoints/imagenet_ft_epoch001.pth
```

## 4. Küresel CNN Filter DB PCA tabanının denetimi

### 4.1. Neden küresel taban?

Üçüncü deneydeki eski PCA tabanı, karşılaştırılan ImageNet ve LeJEPA RN18 filtrelerinin birleşimine fit edilmişti. İki modeli aynı koordinat sistemine koyduğu için bu yaklaşım kendi içinde geçerlidir; fakat koordinat sistemi karşılaştırılan modellerin kendilerine bağlıdır. Yeni taban, çok sayıda mimari, görev ve veri kümesinden gelen yayımlanmış CNN Filter DB popülasyonuna fit edilmiştir. Böylece:

- karşılaştırılan iki RN18 modeli PCA eksenlerini belirlemez;
- LeJEPA'nın her checkpoint'i aynı sabit eksenlere izdüşürülür;
- checkpoint'ler arasında yeniden fit edilen/dönen bir taban olmadığı için eğitim-zamanı drift'i daha temiz yorumlanır;
- eski iki-model tabanındaki sonucun dışsal bir tabanda korunup korunmadığı sınanabilir.

Makale 647 model, 21.436 konvolüsyon katmanı ve tam olarak `1.464.797.156` adet `3×3` filtre raporlamaktadır. Yerel HDF5 dosyasındaki `/filters` dizisi de:

```text
(1464797156, 3, 3), dtype=float64
```

biçimindedir. Artifact'in örnek aralığı `[0, 1464797156)` olarak kayıtlıdır. Dolayısıyla hiçbir filtre aralığı atlanmamış ve örnek altkümesi kullanılmamıştır.

### 4.2. Kaynak veri provenance'ı

Artifact şu kaynak bilgilerini taşımaktadır:

- DOI: `10.5281/zenodo.6371680`
- İndirilen sıkıştırılmış kaynağın MD5'i: `11d0ca3f9c3f7b6e0f73d389db35105a`
- HDF5 dosya büyüklüğü: `105.488.156.386` byte
- Tam filtre sayısı: `1.464.797.156`
- PCA artifact SHA256: `1706ba1b78db4115763bff5e34e02e21c393c6fb739ad20bedf4f6c30d06b135`

HDF5 metadata içindeki model/katman filtre aralıkları başlangıçtan sona kesintisizdir; boşluk veya üst üste binme saptanmamıştır.

### 4.3. Makale ve özgün repo ile uyumlu ön işleme

Makalenin Denklem 9'u, filtre yapısı analizi için her `3×3` filtrenin kendi en büyük mutlak katsayısına bölünmesini tanımlar:

\[
\widetilde f =
\begin{cases}
f/\max_j |f_j|, & \max_j |f_j|>0 \\
f, & \max_j |f_j|=0.
\end{cases}
\]

Özgün reponun güncel `main.ipynb` kod yolu ise ham HDF5 verisini önce dokuz sütuna dönüştürüp `float16` türüne çevirmekte, daha sonra filtre-başına max-abs normalizasyonu uygulamaktadır. Bu nedenle yeniden hesaplama sırası şöyledir:

1. `[N,3,3] -> [N,9]` row-major dönüşüm;
2. `float16` dönüşümü;
3. her satırı kendi `float16` maksimum mutlak katsayısına bölme;
4. parça ortalaması ve merkezlenmiş çapraz çarpımları `float64` biriktirme;
5. örnek kovaryansını `n-1` ile hesaplama;
6. simetrik `9×9` kovaryansı `numpy.linalg.eigh` ile ayrıştırma.

Makale ana metni `float16` zorunluluğu belirtmez; bu, yayımlanmış güncel repo notebook'unun uygulama ayrıntısıdır. Dolayısıyla kullandığımız artifact'i “yazarların yayımlamadığı bit düzeyi orijinal PCA dosyası” diye adlandırmak doğru değildir. Doğru ifade şudur:

> Yayımlanmış CNN Filter DB veri dosyasındaki bütün filtrelerden, makalenin max-abs normalizasyonu ve güncel özgün reponun dtype sırası izlenerek yeniden hesaplanmış küresel PCA tabanı.

RN18 hedef filtreleri bu tabana yansıtılırken de aynı `float16 -> float16 max-abs -> merkezleme -> projeksiyon` sırası kullanılmaktadır. İlk taslakta bulunan float32 hedef-normalizasyon uyumsuzluğu denetimde saptanmış ve nihai koşulardan önce düzeltilmiştir.

### 4.4. Streaming hesabın neden yaklaşık olmadığı

1,46 milyar satırı tek bir RAM matrisinde tutmak yerine her chunk için örnek sayısı, ortalama ve merkezlenmiş toplam çarpım matrisi hesaplanmıştır. İki parçanın sufficient statistics'i şu eşitliklerle birleştirilmiştir:

\[
\delta=\mu_B-\mu_A,
\qquad
\mu=\mu_A+\delta\frac{n_B}{n_A+n_B},
\]

\[
M_2=M_{2,A}+M_{2,B}+\delta\delta^\top\frac{n_A n_B}{n_A+n_B}.
\]

Son örnek kovaryansı:

\[
C=\frac{M_2}{n-1}
\]

olarak hesaplanmıştır. Bu yöntem bütün satırları aynı anda merkezleyip `X^T X/(n-1)` hesaplamakla cebirsel olarak aynıdır. `IncrementalPCA`, randomized SVD veya düşük-rank yaklaşım kullanılmamıştır. Özellik sayısı yalnızca dokuz olduğu için son `9×9` kovaryansın tam özdeğer ayrışımı doğrudan yapılmıştır.

Kontrollü küçük veri testinde kesintisiz hesap ile checkpoint/resume hesabı bit düzeyinde aynı sonucu vermiştir. Bağımsız sklearn PCA karşılaştırmalarında açıklanan varyans oranı ve bileşen eksenleri sayısal hassasiyet düzeyinde eşleşmiştir. Bu nedenle chunk kullanımı sonuç doğruluğu açısından bir yaklaşım değil, yalnızca bellek yönetimidir.

### 4.5. Sayısal doğrulamalar

Yeni tabanın açıklanan varyans oranları şöyledir:

```text
[0.39815034, 0.18326029, 0.11703953,
 0.09852558, 0.06576389, 0.05521558,
 0.03007510, 0.02823360, 0.02373609]
```

Denetim sonuçları:

- EVR toplamı: `0.9999999999999999`
- Bileşen ortonormallik maksimum hatası: `4.718×10^-16`
- Kovaryans simetrik
- Kaydedilmiş kovaryans ile `V^T diag(lambda) V` yeniden yapımı tutarlı
- Bütün ortalama, kovaryans, özdeğer, EVR ve bileşen değerleri sonlu
- `n_samples == full_dataset_filter_count == 1.464.797.156`

### 4.6. Makalenin merkezleme vektörüyle fark

Makalenin ek materyali tam veri kümesi için şu SVD/PCA merkezleme vektörünü verir:

```text
[-0.04262863, -0.04113670, -0.04461834,
 -0.04071190, -0.03574134, -0.04268694,
 -0.04350573, -0.04138637, -0.04486743]
```

Yayımlanmış veri ve güncel notebook ön işlemesiyle hesaplanan vektör:

```text
[-0.04223531, -0.04043981, -0.04412311,
 -0.04007022, -0.03468303, -0.04198876,
 -0.04310292, -0.04076126, -0.04435161]
```

| Konum | Makale | Hesaplanan | Mutlak fark |
|---:|---:|---:|---:|
| 0 | -0.04262863 | -0.04223531 | 0.00039332 |
| 1 | -0.04113670 | -0.04043981 | 0.00069689 |
| 2 | -0.04461834 | -0.04412311 | 0.00049523 |
| 3 | -0.04071190 | -0.04007022 | 0.00064168 |
| 4 | -0.03574134 | -0.03468303 | 0.00105831 |
| 5 | -0.04268694 | -0.04198876 | 0.00069818 |
| 6 | -0.04350573 | -0.04310292 | 0.00040281 |
| 7 | -0.04138637 | -0.04076126 | 0.00062511 |
| 8 | -0.04486743 | -0.04435161 | 0.00051582 |

En büyük mutlak fark `0.00105831`'dir. İlk işteki `5×10^-5` toleransı bu farktan çok daha sıkı olduğu için Slurm işi yalnızca son koruyucu kontrolde `FAILED` olmuştur. Bu durum veri taramasının yarıda kalması anlamına gelmez; tam tarama checkpoint'i `next_index=1.464.797.156` ile tamamlanmıştır.

Bu farkın kesin nedeni eldeki kanıtlarla tayin edilemez. Makale hesabında farklı bir iç snapshot, dtype veya belgelenmemiş küçük bir uygulama ayrıntısı bulunmuş olabilir. Fark normalize katsayıların birim ölçeğinde küçüktür ve kovaryansın iç tutarlılığını bozmamaktadır. Kullanıcının bu büyüklükteki farkı kabul etmesiyle nihai politika şöyledir: fark gizlenmez, `published_mean_validation_passed=False` olarak saklanır, tüketici analizlerde belirgin uyarı üretilir; fakat tam-veri ve sayısal doğrulamalar geçtiği için artifact kullanılır.

### 4.7. PCA işaret belirsizliği

Bir PCA özvektörü `v` ise `-v` de aynı geçerli eksendir. Makale eki de bileşen ters çevirmelerinin karakteristik fark olmadığını söyler. Yeniden üretilebilir dosyalar için her bileşendeki mutlak değeri en büyük yük pozitif olacak şekilde deterministik işaret seçilmiştir.

Bu seçim:

- açıklanan varyansı değiştirmez;
- iki modele aynı anda uygulandığı için symmetric-KL/TV drift büyüklüğünü değiştirmez;
- yoğunluk grafiğini ilgili eksende aynalayabilir;
- signed `LeJEPA - ImageNet` ortalama farkının işaretini konvansiyona bağlar.

Bu nedenle bileşen işaretleri fiziksel “pozitif/negatif özellik” yönleri gibi yorumlanmamalıdır.

### 4.8. Yönlerin ek sanity check'i ve karşılaştırmanın sınırı

Makale/repo, yazarların fit ettiği dokuz yönü element-element karşılaştırmaya izin veren ayrı ve makine-okunur bir `9x9` PCA artifact'i yayımlamamaktadır. Bu nedenle “yazar dosyasıyla bit düzeyinde aynıdır” iddiası yapılamaz. Doğrudan sayısal makale çapası, yukarıda karşılaştırılan merkezleme vektörüdür; farkın maksimumu `0.00105831`'dir. Buna karşılık yön hesabının doğru olduğuna dair birbirini tamamlayan kanıtlar vardır: tam kaynak aralığı ve ön işleme provenance'ı, tam kovaryansın bağımsız yeniden kurulması, sklearn/SVD kontrollü-altküme eşleşmeleri, ortonormallik/EVR kontrolleri ve yayımlanmış eigenfilter ailesiyle uyumlu yön görselleri.

Ek bir sanity check olarak yeni küresel yönler, üçüncü deneydeki iki-model PCA yönleriyle mutlak kosinüs üzerinden eşleştirilmiştir:

| Küresel bileşen | En yakın eski bileşen | Mutlak kosinüs |
|---:|---:|---:|
| 0 | 0 | 0.9989 |
| 1 | 1 | 0.9992 |
| 2 | 3 | 0.9731 |
| 3 | 2 | 0.9712 |
| 4 | 4 | 0.9979 |
| 5 | 5 | 0.9991 |
| 6 | 6 | 0.9991 |
| 7 | 7 | 0.9990 |
| 8 | 8 | 0.9992 |

Yedi yön neredeyse çakışıktır; esas fark eski indeks 2/3 yönlerinin küresel tabanda yer değiştirip aynı iki-boyutlu altuzay içinde hafif dönmesidir. İlk iki bileşenin kümülatif EVR'si eski tabanda `0.581737`, küresel tabanda `0.581411`; ilk dört için sırasıyla `0.794582` ve `0.796976`'dır. Bu benzerlik tek başına küresel hesabı ispatlamaz, çünkü eski taban da RN18 çiftine özeldir; ancak yanlış reshape, yanlış merkezleme ekseni veya tamamen farklı normalizasyon gibi büyük uygulama hatalarına karşı güçlü bir çapraz kontroldür.

Kullanıcının küçük farkın kabul edilebilir olduğu yönlendirmesiyle sonuç şudur: hesap yeniden açılıp başka bir belirsiz dtype kombinasyonu peşinde koşturulmamıştır. Mevcut artifact, küçük makale-ortalaması farkı açıkça işaretlenerek kabul edilmiş; bütün nihai tüketiciler tam checksum'lu bu tek artifact'e bağlanmıştır.

## 5. ImageNet temel çizgisi neden 20 epoch yeniden üretildi ve nerede kullanılıyor?

### 5.1. Eksik olan metrik değil, ağırlık dosyasıydı

Üçüncü deneyin seçilmiş ImageNet epoch-1 doğruluğu, eski metrik CSV'leri ve eski analiz tabloları korunmuştu. Ancak şekilleri yeniden üretmek için gerekli özgün `imagenet_ft_epoch001.pth` ağırlığı klasör yeniden adlandırma/temizleme sürecinde artık mevcut değildi. Dolayısıyla amaç yeni bir baseline tasarlamak değil, önceden seçilmiş aynı model durumunu geri kazanmaktı.

### 5.2. Neden yalnızca bir epoch değil?

Özgün denetimli fine-tuning protokolü 20 epoch ve `CosineAnnealingLR(T_max=20)` kullanıyordu. Yalnızca epoch 1'e kadar çalışmak bir ağırlık dosyası üretebilirdi; ancak aynı toplam eğitim ufku ve scheduler yapılandırmasıyla özgün koşuyu eksiksiz yeniden oynatmış olmazdı. Tam 20 epoch'un yeniden çalıştırılması şu güçlü doğrulamayı sağladı:

- seed `13` korundu;
- doğru batch size `128` kullanıldı;
- aynı ImageNet1K_V1 RN18 başlangıcı ve CIFAR stem'i kullanıldı;
- 20 epoch boyunca train/test loss, train/test accuracy ve learning-rate değerlerinin tamamı korunmuş üçüncü deney CSV'siyle birebir eşleşti;
- en iyi epoch yine 19'da `0.9595`, epoch 20 doğruluğu yine `0.9591`, epoch 1 doğruluğu yine `0.9125` oldu.

Bu tam yörünge eşleşmesi, yalnızca tek doğruluk sayısının eşleşmesinden daha güçlü bir yeniden üretilebilirlik kanıtıdır.

### 5.3. Hangi checkpoint gerçekten kullanılıyor?

Mevcut nihai analizlerde yalnızca yeniden üretilmiş ImageNet epoch 1 ağırlığı kullanılmıştır. Kullanıldığı yerler:

1. Dördüncü deney Part 1: eşik duyarlılığında LeJEPA epoch 110 ile karşılaştırma.
2. Dördüncü deney Part 2: tam CNN Filter DB PCA tabanında ImageNet–LeJEPA çift analizi.
3. Üçüncü deneyin küresel PCA tabanıyla şekil/istatistik yeniden üretimi.
4. PCA-bağımsız raw, BN-folded, normalized-shape ve descriptor kontrol analizleri.

Epoch 2–20 checkpoint'leri mevcut ana karşılaştırmada kullanılmamıştır. Bunlar yeniden üretim yörüngesini doğrulamak, arşiv bütünlüğü sağlamak ve gelecekte başka bir epoch gerektiğinde tekrar eğitim ihtiyacını azaltmak için tutulmuştur. LeJEPA-only eğitim dinamiği bölümünde ImageNet checkpoint'i hiç kullanılmaz.

Kısa cevap:

> Asıl ihtiyaç epoch-1 ağırlığını geri kazanmaktı. Yirmi epoch'un tamamı, aynı toplam eğitim ufku ve scheduler ile özgün koşunun gerçekten yeniden üretildiğini ispatlayan bir regresyon kontrolü sağladı. Güncel karşılaştırmalar yalnızca epoch 1'i kullanmaktadır.

## 6. Sparsity ve entropy eşikleri hesapta nasıl kullanılıyor?

### 6.1. PCA bağımlılığı yoktur

Her residual `3×3` konvolüsyon tensörü `[c_out,c_in,3,3]` biçiminden:

\[
X_L\in\mathbb{R}^{n_L\times 9},\qquad n_L=c_{out}c_{in}
\]

matrisine dönüştürülür. Entropi yerel katman kovaryansından, sparsity ham ağırlık büyüklüklerinden hesaplanır. Küresel CNN Filter DB PCA bileşenleri bu iki metriğin girdisi değildir.

### 6.2. Entropi H

Katman filtresi kendi dokuz-boyutlu ortalaması etrafında merkezlenir. SVD tekil değerlerinden veya eşdeğer kovaryans özdeğerlerinden varyans oranları hesaplanır:

\[
\lambda_i=\frac{\sigma_i^2}{n_L-1},\qquad
p_i=\frac{\lambda_i}{\sum_j\lambda_j}.
\]

Makaledeki taban-10 Shannon entropisi:

\[
H(L)=-\sum_{i=0}^{8}p_i\log_{10}p_i
\]

olarak bulunur. Dokuz yön eşit varyansa sahip olduğunda teorik maksimum `log10(9)≈0.95424`'tür. Düşük `H`, katman filtresi varyansının az sayıda yönde yoğunlaştığını; yüksek `H`, daha eşit dağıldığını gösterir.

Temel `H` her katman için yalnızca bir kez hesaplanır. `0.30,...,0.90` cutoff sweep'i SVD'yi veya `H` değerini değiştirmez; yalnızca:

\[
\text{low-entropy}(L;c)=\mathbf{1}[H(L)<c]
\]

kararını değiştirir. Cutoff yükseldikçe işaretlenen katman sayısı monoton olarak artar.

### 6.3. Rastgelelik referansı TH(n)

Makalenin rastgele normal filtre deneylerine fit ettiği referans:

\[
TH(n)=\frac{1.26}{1+\exp[-0.89(\log_2 n-2.30)]}-0.31
\]

şeklindedir. Makalenin doğrudan random kuralı `H>TH(n)`, yani `H/TH>1`'dir. Analizde önce sabit:

\[
R(L)=H(L)/TH(n_L)
\]

oranı hesaplanır; sonra `0.75,0.80,0.85,0.90,0.95,1.00,1.05` cutoff'ları için yalnızca `R>cutoff` kararı değiştirilir. `1.00` makalenin gerçek sınırı, `0.95` önceki deneydeki daha toleranslı “random-like” heuristiğidir.

Makalenin ekindeki birleşik degeneration ablation ölçütü:

\[
(H\ge TH-0.02)\ \lor\ [(H<0.5)\land(S\ge0.14)]
\]

şeklindedir. Bizim eski `H/TH>0.95` bayrağımız bunun birebir aynısı değildir; ayrı bir duyarlılık heuristiğidir. Yeni sweep'in `1.00` içermesi makalenin doğrudan random sınırını da görünür yapmaktadır.

### 6.4. Sparsity epsilon

Önce ilgili katmanın bütün ham katsayıları içindeki tepe büyüklüğü bulunur:

\[
M_L=\max_{f\in L,j}|f_j|.
\]

Her epsilon için gerçek mutlak near-zero eşiği:

\[
\tau_L(\epsilon)=\epsilon M_L
\]

olur. Bir `3×3` filtre, ancak dokuz katsayısının tamamı bu eşikten küçükse sparse sayılır:

\[
\text{sparse}(f;\epsilon)=
\mathbf{1}[\forall j, |f_j|<\tau_L(\epsilon)].
\]

Katman sparsity oranı:

\[
S(L;\epsilon)=\frac{\#\text{sparse filters}}{n_L}
\]

olarak hesaplanır. Kullanılan epsilons:

```text
0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.10
```

`epsilon=0.01`, mutlak ağırlık sınırının `0.01` olması demek değildir; katmanın tepe mutlak ağırlığının yüzde biri demektir. Örneğin `M_L=0.8` ise gerçek sınır `0.008` olur.

İki farklı yüzde-bir ifadesi karıştırılmamalıdır:

- `epsilon=0.01`: katsayı near-zero sınırının katman tepesine oranı;
- `S>0.01`: katmandaki filtrelerin yüzde birinden fazlasının sparse olması.

Epsilon arttıkça near-zero aralığı genişler; sparse filtre kümesi küçülemez. Sonuçlarda da her model/katman için `S(epsilon)` monoton artan veya sabit bulunmuştur.

### 6.5. Clean entropy

Temel `H` sparse filtreler çıkarılmadan hesaplanır ve cutoff sweep'inde sabittir. Ayrı adlandırılan clean-entropy tanısı ise her epsilon için:

1. epsilon-sparse filtreleri belirler;
2. bu filtreleri katman örneklerinden çıkarır;
3. kalan filtrelerin kovaryans entropisini yeniden hesaplar.

\[
H_{clean}(L;\epsilon)=H(\{f\in L:\neg sparse(f;\epsilon)\}).
\]

Dolayısıyla epsilon'a bağlı olan temel entropi değil, filtre çıkarımı sonrasındaki ayrı `H_clean` ölçümüdür. `H_clean`'in monoton olması gerekmez.

### 6.6. Eşiklerin ne olmadığı

Bu eşikler model eğitim hiperparametresi değildir; ağırlıkları değiştirmez, pruning yapmaz, PCA tabanını değiştirmez ve tek başına doğruluk/temsil kalitesi kanıtı değildir. Amaç, “sparse/low-entropy/random-like” kararının seçilen sayısal sınıra ne kadar duyarlı olduğunu göstermektir.

## 7. Part 1: eşik duyarlılığı sonuçları

Analiz stem'i dışarıda bırakır ve model başına 16 residual-block `3×3` katmanı inceler. Temel özet:

| Model | Ortalama H | Min H | Max H | Ortalama H/TH | epsilon=0.01 ortalama S | epsilon=0.01 max S |
|---|---:|---:|---:|---:|---:|---:|
| ImageNet FT epoch 1 | 0.731123 | 0.209724 | 0.872600 | 0.769668 | 0.007925 | 0.125244 |
| LeJEPA SSL epoch 110 | 0.804338 | 0.688594 | 0.888862 | 0.846737 | 0.000000 | 0.000000 |

Her iki modelde minimum entropi `layer4.1.conv2.weight` katmanındadır. Fakat ImageNet için `H=0.209724` çok belirgin bir düşük-çeşitlilik durumu iken LeJEPA için aynı katman `H=0.688594` düzeyindedir. ImageNet'in maksimum entropisi `layer2.0.conv1`, LeJEPA'nın maksimumu `layer2.0.conv2` katmanındadır.

### 7.1. Sparsity epsilon sweep'i

| Epsilon | ImageNet ort. S | ImageNet max S | ImageNet S>0 katman | LeJEPA ort. S | LeJEPA max S | LeJEPA S>0 katman |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0010 | 0.007812 | 0.125000 | 1 | 0.000000 | 0.000000 | 0 |
| 0.0025 | 0.007813 | 0.125000 | 2 | 0.000000 | 0.000000 | 0 |
| 0.0050 | 0.007814 | 0.125000 | 2 | 0.000000 | 0.000000 | 0 |
| 0.0100 | 0.007925 | 0.125244 | 4 | 0.000000 | 0.000000 | 0 |
| 0.0200 | 0.011669 | 0.133057 | 15 | 0.000002 | 0.000015 | 3 |
| 0.0500 | 0.106536 | 0.477062 | 16 | 0.001589 | 0.007080 | 15 |
| 0.1000 | 0.504424 | 0.901447 | 16 | 0.080413 | 0.229980 | 16 |

En önemli desen, ImageNet'in küçük epsilon'larda bile belirli bir katmanda yaklaşık `%12,5` sparse filtre göstermesi; LeJEPA'nın ise `epsilon<=0.01` için bütün residual katmanlarda sıfır ölçülmesidir. Epsilon çok büyütüldüğünde iki modelde de sparsity yükselir; çünkü tanım giderek daha fazla küçük fakat sıfır olmayan filtreyi “sparse” kabul etmeye başlar. Buna rağmen `epsilon=0.10` gibi oldukça geniş sınırda bile ImageNet ortalaması `0.5044`, LeJEPA ortalaması `0.0804`'tür.

Bu, seçili checkpoint'lerdeki sparsity farkının yalnızca tam `1%` eşiğine özgü olmadığını gösterir. Fakat `epsilon=0.05` ve özellikle `0.10` artık çok geniş near-zero tanımlarıdır; bu satırlar gerçek “prunable zero filter” oranı gibi değil, eşik hassasiyetinin üst sınırı gibi okunmalıdır.

### 7.2. Düşük-entropi cutoff sweep'i

| H cutoff | ImageNet işaretli /16 | LeJEPA işaretli /16 |
|---:|---:|---:|
| 0.30 | 1 | 0 |
| 0.40 | 1 | 0 |
| 0.50 | 1 | 0 |
| 0.60 | 2 | 0 |
| 0.70 | 4 | 1 |
| 0.80 | 9 | 6 |
| 0.90 | 16 | 16 |

ImageNet'in `layer4.1.conv2` düşük-entropi sonucu `0.30–0.50` aralığının tamamında kararlıdır. LeJEPA'da `H<0.50` katman yoktur. Cutoff `0.8–0.9` düzeyine çıkarıldığında teorik maksimum `0.95424`'e yaklaşılır ve “low entropy” etiketi seçici niteliğini kaybeder; hemen her katman işaretlenir. Bu yüzden sweep'in amacı tek doğru cutoff ilan etmek değil, kararın nerede kararlı ve nerede keyfî hâle geldiğini göstermektir.

### 7.3. H/TH random-like cutoff sweep'i

| H/TH cutoff | ImageNet işaretli /16 | LeJEPA işaretli /16 |
|---:|---:|---:|
| 0.75 | 12 | 15 |
| 0.80 | 10 | 12 |
| 0.85 | 6 | 10 |
| 0.90 | 1 | 2 |
| 0.95 | 0 | 0 |
| 1.00 | 0 | 0 |
| 1.05 | 0 | 0 |

Makalenin doğrudan `H>TH` sınırında (`H/TH>1`) iki modelde de random olarak işaretlenen katman yoktur. Eski daha toleranslı `0.95` sınırında da sonuç sıfırdır. Ancak cutoff `0.85` gibi daha aşağıya çekildiğinde LeJEPA daha fazla katmanda random referansa yaklaşır. Bu bulgu “LeJEPA katmanları randomdır” anlamına gelmez; hiçbiri paper boundary'yi aşmamaktadır. Daha doğru yorum, LeJEPA'nın yerel varyans spektrumunun ImageNet'e göre genellikle daha eşit ve random referansına daha yakın, fakat hâlâ referansın altında olduğudur.

### 7.4. Clean entropy

Küçük epsilon'larda sparse filtre çıkarımı temel entropiyi neredeyse hiç değiştirmez. En geniş `epsilon=0.10` koşulunda ortalama clean-H farkı ImageNet için yaklaşık `-0.0322`, LeJEPA için `-0.0030`'dır. ImageNet'te en büyük tek-katman mutlak değişim yaklaşık `0.1046`, LeJEPA'da `0.0114`'tür. Bu da ImageNet'in entropy profilinin geniş sparsity tanımında çıkarılan filtrelerden daha fazla etkilendiğini gösterir. Fakat bu sonuç temel `H` sweep'inin parçası değil, ayrı bir post-exclusion tanısıdır.

### 7.5. Part 1 görselleri

Dört heatmap korunmuştur. Her heatmap için yedi ayrı chart alt klasörü oluşturulmuştur:

```text
figures/sparsity_threshold_by_depth_heatmap/                 # 7 epsilon chart'ı
figures/low_entropy_decision_heatmap/                        # 7 H cutoff chart'ı
figures/random_like_decision_heatmap/                        # 7 H/TH cutoff chart'ı
figures/clean_entropy_after_sparse_exclusion_heatmap/        # 7 epsilon chart'ı
```

Karar heatmap'lerinin chart karşılıkları yalnızca `0/1` çizgisi çizmek yerine daha bilgilendirici olarak ImageNet ve LeJEPA'nın sabit sürekli `H` veya `H/TH` eğrisini ve ilgili yatay cutoff çizgisini birlikte göstermektedir.

## 8. Part 2: küresel PCA tabanında ImageNet–LeJEPA çift sonucu

PCA analizi stem'i dışarıda bırakır ve model başına `1.220.608` residual-block `3×3` filtresi kullanır. Yeni küresel tabandaki genel sonuç:

```text
Global drift D = 0.1330218026
```

Stage sonuçları:

| Stage | Küresel taban drift D |
|---|---:|
| layer1 | 0.070725 |
| layer2 | 0.011124 |
| layer3 | 0.017624 |
| layer4 | 0.193849 |

En yüksek layer drift'leri:

| Katman | Stage | Drift D |
|---|---|---:|
| `layer4.1.conv2.weight` | layer4 | 0.779966 |
| `layer1.0.conv1.weight` | layer1 | 0.285191 |
| `layer1.0.conv2.weight` | layer1 | 0.235985 |
| `layer1.1.conv1.weight` | layer1 | 0.181766 |
| `layer4.1.conv1.weight` | layer4 | 0.125888 |
| `layer1.1.conv2.weight` | layer1 | 0.106079 |
| `layer2.0.conv1.weight` | layer2 | 0.096252 |
| `layer2.0.conv2.weight` | layer2 | 0.071522 |

Global weighted symmetric-KL katkısında ilk beş bileşen `c0`, `c2`, `c5`, `c7`, `c6`'dır. `c0` katkısı `0.102808` ile toplam farkın büyük bölümünü taşır. Bununla birlikte eski iki-model PCA ile yeni küresel PCA'nın bileşen sıraları aynı semantiğe sahip değildir; örneğin eski `c3` yönü yeni tabanda başka bir numaraya daha yakın olabilir. Bu yüzden “c2 biyolojik/uzamsal olarak şudur” gibi yalnızca numaraya dayalı yorum yapılmamalıdır.

### 8.1. Eski iki-model PCA ile karşılaştırma

Eski dedicated no-stem analizde global drift `0.128806` idi. Yeni değer `0.133022`'dir:

| Taban | Global D |
|---|---:|
| Yalnızca eşleşmiş iki RN18 modeline fit PCA | 0.128806 |
| Tam CNN Filter DB küresel PCA | 0.133022 |
| Mutlak fark | +0.004216 |
| Göreli fark | yaklaşık +%3,27 |

Stage karşılaştırması:

| Stage | Eski yerel PCA | Yeni küresel PCA | Değişim |
|---|---:|---:|---:|
| layer1 | 0.079488 | 0.070725 | -0.008763 |
| layer2 | 0.015189 | 0.011124 | -0.004065 |
| layer3 | 0.017935 | 0.017624 | -0.000311 |
| layer4 | 0.185562 | 0.193849 | +0.008287 |

Mutlak olarak ana sıralama korunur: layer4 en farklı, layer1 ikinci, orta layer2/layer3 daha yakındır. Özellikle `layer4.1.conv2` drift'i eski tabanda `0.779819`, yeni tabanda `0.779966` ile neredeyse değişmemiştir. Bu katmanın güçlü farkı PCA tabanına bağlı bir artefakt görünmemektedir.

Erken layer1 içindeki bireysel büyüklükler daha fazla değişmektedir. Bu beklenebilir: histogram drift'i bileşen yönleri ve bileşen EVR ağırlıkları değişince nicel olarak yeniden dağılır. Bu nedenle en sağlam sonuç, tek bir layer1 alt katmanının tam değerinden ziyade “erken bloklarda ikincil bir fark kümesi ve son blokta çok güçlü bir fark” desenidir.

### 8.2. PCA-bağımsız kontroller

Raw ağırlık, BN-folded ağırlık, per-kernel normalized shape, kernel descriptors, BN scale ve spatial-energy tablolarının tamamı eski üçüncü deneyle birebir aynıdır. Altı CSV ailesindeki maksimum sayısal fark `0.0` bulunmuştur. Bu kontrollerin ana sonuçları değişmemiştir:

- LeJEPA raw ağırlık ölçeği ImageNet'ten çok büyüktür; fakat BatchNorm folding sonrasında sıralama tersine döner.
- Bu nedenle ham ağırlık büyüklüğü tek başına fonksiyonel fark gibi yorumlanamaz.
- Magnitude kaldırıldığında global normalized-shape TV farkı yaklaşık `0.0531`'dir.
- LeJEPA seçili checkpoint'te daha yüksek high-frequency ratio ve roughness, daha düşük center-energy ratio gösterir.
- Normalize şekil/descriptor farkları stem ve son residual blokta yoğunlaşır; orta bloklar çoğunlukla daha benzerdir.

Bu sayıların yeni PCA tabanıyla değişmemesi normaldir; bu analizler PCA katsayılarını kullanmaz. Birebir eşleşme aynı zamanda yeniden üretilen ImageNet epoch-1 checkpoint'inin eski analizdeki ağırlık davranışını eksiksiz geri getirdiğine dair güçlü regresyon kanıtıdır.

### 8.3. Heatmap ve chart sürümleri

PCA drift heatmap'leri korunmuş; `drift_global_chart.png`, her stage için `drift_stage_<stage>_chart.png` ve bütün stage'leri birlikte gösteren `drift_stage_chart.png` eklenmiştir. Kernel distribution bölümünde:

- `layerwise_weight_distance_heatmap.png` yanında `layerwise_weight_distance_chart.png`;
- `layerwise_descriptor_distance_heatmap.png` yanında `layerwise_descriptor_distance_chart.png`;
- `spatial_energy_maps.png` yanında her global/stage scope için ayrı line-chart alt klasörü

üretilmiştir.

## 9. Üçüncü deneyin küresel PCA ile şekil-only yeniden çalıştırılması

Yeni çıktı:

```text
outputs/(3rdEXP)(theONE)rn18_cifar10_true_lejepa_cfg_l02_v4_p64_strong/
  analysis_global_cnn_filter_db_basis_epoch001_vs_epoch110/
```

altındadır. Bu iş eğitim betiğini çağırmamış, yalnızca iki mevcut checkpoint'i yükleyip analiz/şekil üretmiştir. Çıktı 63 dosya ve 39 PNG içermektedir. Dördüncü deney Part 2 ile karşılaştırılabilen 18 CSV'nin tamamı aynı shape, kolonlar ve sayısal değerleri taşır; maksimum sayısal fark `0.0`'dır. Böylece dördüncü deneyin pair sonucu ile üçüncü deney altındaki yeniden üretim arasında herhangi bir hesap ayrışması yoktur.

Eski çıktıların üzerine yazılmamıştır. Bu önemlidir: iki-model yerel PCA sonucu ve yeni küresel PCA sonucu yan yana denetlenebilir.

## 10. Part 3: LeJEPA eğitim dinamikleri

### 10.1. Kapsam ve hesap protokolü

Bu bölümde ImageNet modeli kullanılmamıştır. Amaç iki modeli karşılaştırmak değil, tek bir LeJEPA SSL omurgasının filtre dağılımının eğitim boyunca nasıl değiştiğini aynı dışsal koordinat sisteminde izlemektir. Analiz tam olarak şu checkpoint'leri kullanır:

```text
1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120
```

Her checkpoint'te stem hariç aynı 16 residual `3x3` konvolüsyon katmanı ve toplam `1.220.608` filtre bulunur. Katman adları, tensor shape'leri ve filtre aralıkları 13 checkpoint'in tamamında birebir aynıdır. Analiz hiçbir checkpoint'i güncellememiş, ara checkpoint üretmemiş, eksik epoch interpolate etmemiş ve model eğitmemiştir. İş yalnızca üçüncü deney klasöründeki mevcut LeJEPA ağırlıklarını okur.

Her hedef filtre, küresel artifact ile tutarlı olarak önce `float16`'ya çevrilir, sonra kendi `float16` maksimum mutlak katsayısına bölünür, küresel ortalamayla merkezlenir ve sabit dokuz PCA yönüne yansıtılır. PCA her epoch'ta yeniden fit edilmez. Bu kritik bir tasarım tercihidir: epoch'a göre dönen bir PCA ekseni, gerçek dağılım hareketiyle koordinat-sistemi hareketini birbirine karıştırırdı.

Drift için her bileşende 70 histogram bin'i kullanılır. Bin sınırları her checkpoint çifti için yeniden seçilmemiştir. Önce 13 checkpoint'in ilgili PCA bileşenindeki birleşik minimum ve maksimumu bulunmuş, küçük bir dış padding eklenmiş ve oluşan tek sınır kümesi global, stage ve layer kapsamlarının tamamında yeniden kullanılmıştır. Böylece örneğin `D(e1,e20)` ile `D(e80,e100)` aynı sayısal bölmeler üzerinde hesaplanır. Kaydedilen `fixed_pca_histogram_edges.csv` tablosunun `9 x 71 = 639` satır içermesinin nedeni budur.

İki epoch arasındaki toplam değer:

\[
D(a,b)=\sum_{k=0}^{8} w_k
\left[KL(P_{a,k}\|P_{b,k})+KL(P_{b,k}\|P_{a,k})\right],
\]

biçimindedir. Burada `w_k`, tam CNN Filter DB tabanının açıklanan varyans oranı; `P_{a,k}` ise epoch `a` için bileşen `k` histogramıdır. Bu sürüm iki yönlü KL toplamını kullanır; yüzde, doğruluk farkı veya sınırlı bir uzaklık değildir. Aynı protokol korunduğunda karşılaştırmalı bir dağılım-drift ölçüsüdür.

### 10.2. İş ve çıktı doğrulaması

LeJEPA dinamiği Slurm işi `1470909` ile tamamlanmıştır: durum `COMPLETED`, exit code `0:0`, süre `00:01:17`. Nihai klasör:

```text
outputs/(4thEXP)rn18_cifar10_filter_dynamics/lejepa_training_dynamics/
```

altında 15 CSV, 25 PNG, iki provenance JSON'u ve bir özet dosyası vardır. Temel tablo boyutları şöyledir:

| Çıktı | Satır | Beklenen yapı |
|---|---:|---|
| `global_pairwise_epoch_drift.csv` | 169 | `13 x 13` epoch çifti |
| `global_pairwise_component_contributions.csv` | 1.521 | `13 x 13 x 9` bileşen |
| `global_drift_trajectories.csv` | 13 | checkpoint başına bir satır |
| `stage_drift_trajectories.csv` | 52 | `13 x 4` stage |
| `layer_drift_trajectories.csv` | 208 | `13 x 16` katman |
| `global_component_drift_contributions.csv` | 351 | `13 x 3 x 9` referans türü/bileşen |
| `layer_quality_by_epoch.csv` | 208 | `13 x 16` yerel kalite tanısı |
| `distribution_summary_by_epoch.csv` | 3.549 | raw, BN-folded, shape ve descriptor özetleri |
| `spatial_energy_by_epoch.csv` | 1.755 | global/stage `3x3` enerji haritaları |
| `pca_component_statistics.csv` | 117 | `13 x 9` bileşen özeti |
| `quality_aggregate_by_epoch.csv` | 39 | `13 x 3` entropy/H-TH/sparsity özeti |
| `milestone_scalar_histograms.csv` | 4.550 | beş milestone'un sabit-bin histogramları |
| `dynamics_training_context.csv` | 13 | SSL loss ve probe bağlamı |
| `fixed_pca_histogram_edges.csv` | 639 | `9 x 71` sabit sınır |
| `validation_checks.csv` | 9 | sekiz zorunlu geçiş ve bir uyarı |

Denetimde yalnızca betiğin kendi `validation_checks.csv` dosyasına güvenilmemiştir. Aşağıdaki kontroller ayrıca bağımsız olarak tekrar edilmiştir:

- 13 kaynak checkpoint'in SHA256 değeri yeniden hesaplanmış ve provenance ile eşleşmiştir.
- Her checkpoint'te stem'in gerçekten dışarıda, residual katman sayısının 16 ve filtre sayısının `1.220.608` olduğu doğrulanmıştır.
- `13x13` drift matrisinin simetri hatası tam `0`, diagonal maksimumu tam `0` bulunmuştur.
- Dokuz bileşen katkısının toplamı her epoch çifti için global `D` değerini en fazla `4.996x10^-16` hatayla geri vermektedir.
- Sabit 71 sınır her bileşende kesin artandır; birleşik projeksiyon aralığını kapsar ve hiçbir katsayı histogram dışında kalmaz.
- Bütün 13 checkpoint yeniden projekte edilip histogramlar bağımsız kurulduğunda kayıtlı pairwise matris en fazla `9.78x10^-17` farkla yeniden elde edilmiştir.
- Histogram olasılık toplamlarının maksimum hatası `2.22x10^-16`, milestone histogramlarının maksimum toplam hatası `3.77x10^-15`'tir.
- Bütün 25 PNG dosyası açılmış, ayrıca temel heatmap/chart/trajectory örnekleri görsel olarak incelenmiştir.
- Çekirdek sayısal tablolarda `NaN` veya sonsuz değer yoktur. Descriptor tablosundaki bazı `stage/layer_order` boşlukları yalnızca global-scope metadata alanlarının yapısal olarak uygulanamaz olmasındandır.

Sekiz zorunlu validation kontrolünün tamamı geçmiştir. Tek `False` satır `published_mean_matches_supplement` kontrolüdür ve severity değeri açıkça `warning`'dir. Kabul edilen maksimum `0.00105830942` makale-ortalaması farkı validation CSV'sinde, iki provenance JSON'unda ve bölüm özetinde görünür kalmaktadır. Dolayısıyla bu satır gizli bir dinamik-hesap hatası değildir.

### 10.3. Global drift yörüngesi

Global sonuçların tamamı:

| Epoch | Epoch 1'den D | Önceki kayıtlı checkpoint'ten D | Epoch 120'ye D |
|---:|---:|---:|---:|
| 1 | 0.000000 | 0.000000 | 0.419884 |
| 10 | 0.172998 | 0.172998 | 0.053262 |
| 20 | 0.341305 | 0.027179 | 0.005730 |
| 30 | 0.417520 | 0.003989 | 0.000948 |
| 40 | 0.441594 | 0.000520 | 0.000678 |
| 50 | 0.442967 | 0.000182 | 0.000560 |
| 60 | 0.437054 | 0.000203 | 0.000340 |
| 70 | 0.431374 | 0.000180 | 0.000227 |
| 80 | 0.426087 | 0.000124 | 0.000123 |
| 90 | 0.423185 | 0.000102 | 0.000096 |
| 100 | 0.420848 | 0.000112 | 0.000042 |
| 110 | 0.420222 | 0.000042 | 0.000007 |
| 120 | 0.419884 | 0.000007 | 0.000000 |

Epoch 1 ile epoch 120 arasındaki toplam global drift `0.419884`'tür. En büyük ardışık-kayıt değişimi epoch 1->10 aralığında `0.172998`'dir; 10->20'de `0.027179`'a, 20->30'da `0.003989`'a düşer. Epoch 30'dan final checkpoint'e kalan dağılım uzaklığı yalnızca `0.000948`'dir. Bu nedenle sabit küresel PCA koordinatlarında görülen global histogram organizasyonunun büyük kısmı ilk 20–30 epoch'ta gerçekleşmiştir.

İlk aralık 1'den 10'a dokuz epoch, sonraki kayıt aralıkları çoğunlukla on epoch'tur. Dolayısıyla ardışık değerleri hassas bir “epoch başına hız” olarak okumamak gerekir. Buna rağmen ilk iki aralığın daha sonraki değerlerden bir–iki mertebe büyük olması yalnızca bu bir-epoch uzunluk farkıyla açıklanamaz.

`D(e1,e)` monoton değildir. Epoch 1'den uzaklık epoch 50'de `0.442967` ile maksimuma çıkar, sonra finalde `0.419884`'e hafifçe geri döner. Bu, dağılımın ilk durumdan uzaklaşıp daha sonra aynı ilk dağılıma kısmen yaklaşan bir “overshoot/relaxation” deseni gösterdiğini söyler. Symmetric KL üçgen eşitsizliği sağlayan bir metrik olmadığı için bu geri dönüşü doğrusal yol uzunluğu veya tek tek filtrelerin geriye hareketi gibi yorumlamak doğru olmaz.

Part 2'deki ImageNet–LeJEPA `D=0.133022` değeriyle bu `0.419884` değerini doğrudan büyüklük karşılaştırmasına sokmak da doğru değildir. PCA tabanı aynı olsa bile Part 2 bin sınırlarını iki model çifti üzerinden, dinamik analizi ise 13 checkpoint'in birleşik aralığı üzerinden sabitler. Buradaki güçlü sonuç mutlak iki sayının oranı değil, tek dinamik protokol içindeki zaman desenidir.

### 10.4. Stage ve katman düzeyindeki hareket

Epoch 1->120 stage sonuçları:

| Stage | Filtre sayısı | Global havuzdaki pay | D(e1,e120) |
|---|---:|---:|---:|
| layer1 | 16.384 | %1,34 | 0.529162 |
| layer2 | 57.344 | %4,70 | 0.236437 |
| layer3 | 229.376 | %18,79 | 0.359590 |
| layer4 | 917.504 | %75,17 | 0.466528 |

En büyük stage drift'i `layer1`, ikincisi `layer4`'tür. Ancak global havuzun yaklaşık dörtte üçünü layer4 filtreleri oluşturduğu için global `D` doğal olarak derin katman dağılımlarına daha duyarlıdır. Global değer stage değerlerinin basit filtre-ağırlıklı ortalaması değildir; histogramlar havuzlandıktan sonra KL hesaplandığı için işlem doğrusal değildir. Yine de filtre-sayısı dengesizliği, neden global grafiğin tek başına yeterli olmadığını açıklar.

Zaman desenleri de stage'e göre farklıdır. Epoch 10'da layer4 `0.190914` ve layer3 `0.163763` ile erken güçlü hareket gösterirken layer1 yalnızca `0.063439`'dur. Layer4 epoch 40 civarında yaklaşık `0.497` düzeyine ulaşır ve sonra `0.466528`'e gevşer. Buna karşılık layer1 daha uzun süre değişmeye devam eder: epoch 10'da `0.063439`, epoch 30'da `0.313164`, epoch 50'de `0.472242` ve finalde `0.529162`'dir. Dolayısıyla “global dağılım epoch 30 civarında stabilize oluyor” ifadesi bütün katmanların aynı anda donduğu anlamına gelmez; özellikle az sayıda filtre taşıyan erken katmanlardaki devam eden hareket global havuzda bastırılabilir.

Final endpoint'te en yüksek katman drift'leri:

| Katman | Stage | D(e1,e120) |
|---|---|---:|
| `layer1.0.conv2.weight` | layer1 | 1.128378 |
| `layer1.1.conv2.weight` | layer1 | 0.941002 |
| `layer1.0.conv1.weight` | layer1 | 0.847634 |
| `layer4.1.conv2.weight` | layer4 | 0.776601 |
| `layer1.1.conv1.weight` | layer1 | 0.762926 |
| `layer3.0.conv1.weight` | layer3 | 0.606682 |
| `layer3.1.conv2.weight` | layer3 | 0.600993 |
| `layer2.0.conv1.weight` | layer2 | 0.587541 |

`layer1.0.conv2` finalde en büyük değişime sahiptir ve epoch 100 civarında yaklaşık `1.167` ile daha yüksek bir noktadan final `1.128` değerine gevşer. `layer4.1.conv2` de epoch 50'de yaklaşık `0.837` düzeyine çıktıktan sonra `0.777`'ye iner. Bu katman-temelli overshoot örnekleri global eğrinin hafif geri dönüşüyle uyumludur.

Part 2'de ImageNet–LeJEPA farkının en güçlü katmanı `layer4.1.conv2` iken LeJEPA'nın kendi eğitiminde en çok hareket eden katmanın `layer1.0.conv2` olması çelişki değildir. Birincisi iki farklı eğitim protokolünün endpoint farkını, ikincisi tek protokolün epoch 1'e göre zaman değişimini ölçer.

### 10.5. Hangi küresel PCA bileşenleri değişimi taşıyor?

Epoch 1->120 global drift'inin bileşen katkıları:

| Bileşen | Ağırlıklı symmetric-KL | Toplam D payı |
|---:|---:|---:|
| c0 | 0.309096 | %73,615 |
| c1 | 0.050059 | %11,922 |
| c8 | 0.020019 | %4,768 |
| c7 | 0.013642 | %3,249 |
| c6 | 0.013483 | %3,211 |
| c5 | 0.005855 | %1,395 |
| c4 | 0.003629 | %0,864 |
| c2 | 0.003389 | %0,807 |
| c3 | 0.000711 | %0,169 |

İlk iki bileşen toplam drift'in `%85,54`'ünü taşır. İşaret-kanonikleştirilmiş `c0` görsel olarak bütün `3x3` katsayıları benzer işaretli, DC/ortalama-benzeri bir yöndür; `c1` ise başlıca bir uzamsal karşıtlık yönüdür. Ancak PCA yönleri istatistiksel eksenlerdir ve işaretleri konvansiyoneldir; bunlara öğrenilmiş semantik özellik etiketi vermek için bu analiz tek başına yeterli değildir.

Bileşen katsayı ölçekleri de varyansın önde gelen yönlerde yoğunlaştığını gösterir. `c0` standart sapması epoch 1'de `0.5811` iken finalde `0.9448`, `c1` için `0.5613`'ten `0.7728`'e çıkar. Buna karşılık `c6`, `c7` ve `c8` standart sapmaları sırasıyla yaklaşık `0.5367->0.3367`, `0.5377->0.3302` ve `0.5377->0.2768` değişir. Yani eğitim yalnızca toplam ölçeği büyütmez; normalize filtre şekli varyansını belirli küresel yönlerde yeniden dağıtır.

### 10.6. Entropi ve sparsity'nin eğitim boyunca değişimi

Yerel katman entropy'si PCA'dan bağımsızdır. Katmanların eşit ağırlıklı ortalaması:

| Epoch | Ortalama H | Filtre-ağırlıklı H | Ortalama H/TH | Ortalama S, epsilon=0.01 |
|---:|---:|---:|---:|---:|
| 1 | 0.953554 | 0.953275 | 1.003817 | 0.000000 |
| 10 | 0.898432 | 0.862468 | 0.945792 | 0.000000 |
| 20 | 0.848626 | 0.793140 | 0.893362 | 0.000000 |
| 30 | 0.824790 | 0.771432 | 0.868270 | 0.000000 |
| 50 | 0.807711 | 0.767450 | 0.850289 | 0.000000 |
| 80 | 0.803970 | 0.772406 | 0.846349 | 0.000000 |
| 110 | 0.804338 | 0.774045 | 0.846737 | 0.000000 |
| 120 | 0.804359 | 0.774083 | 0.846759 | 0.000000 |

Dokuz yön için teorik maksimum `log10(9)=0.95424` olduğundan epoch-1 ortalaması maksimuma çok yakındır. Epoch 1'de bütün 16 katmanın `H/TH` oranı `1.0031–1.0043` arasındadır ve makalenin doğrudan random referansını çok az aşar. Epoch 10'da aralık `0.8750–0.9920`'ye düşer; artık hiçbir katman `H/TH>1` değildir. Bu, birinci epoch checkpoint'inin yerel ikinci-derece filtre istatistiğinin random referansa çok yakın, eğitim ilerledikçe varyansın daha az sayıda yapısal yöne yoğunlaşmış olduğunu gösterir. Epoch 1 bir random initialization dosyası değil, bir epoch eğitilmiş checkpoint olduğundan ifade “tamamen rastgele filtreler” biçiminde aşırı genellenmemelidir.

Entropy düşüşünün büyük kısmı yine ilk 30 epoch'ta oluşur: `0.953554->0.824790`. Daha sonra layer-ortalaması yaklaşık `0.804` çevresinde plato yapar. Finalde en düşük `H=0.688687` ile `layer4.1.conv2`, en yüksek `H=0.888859` ile `layer2.0.conv2`'dir. Epoch-120 sayılarının epoch-110 Part 1 değerlerine çok yakın olması beklenen geç plato davranışıyla tutarlıdır.

Legacy `epsilon=0.01` tanımında sparsity, bütün 13 checkpoint ve bütün 16 residual katmanda tam sıfırdır. Bu sonuç “ağırlıklarda küçük katsayı yoktur” demek değildir. Bir filtrenin sparse sayılması için dokuz katsayının tamamının kendi katman tepesinin yüzde birinden küçük olması gerekir. Sonuç yalnızca bu güçlü, filtre-düzeyi near-zero koşulunun LeJEPA eğitim yörüngesinde hiç gerçekleşmediğini söyler.

### 10.7. Ham ağırlık, BatchNorm ve şekil descriptor'ları

PCA drift'i tek başına yorumlanmamıştır. PCA'dan bağımsız global endpoint özetleri:

| Metrik | Epoch 1 | Epoch 120 | Yorum için önemli nokta |
|---|---:|---:|---|
| Ortalama raw kernel L2 | 0.07149 | 0.39805 | Ham konvolüsyon normu güçlü biçimde büyür. |
| Ortalama BN-folded kernel L2 | 0.04543 | 0.07111 | Fonksiyonel ölçeğe daha yakın artış çok daha küçüktür. |
| Ortalama mutlak BN scale | 0.65330 | 0.17622 | BN ölçeği raw ağırlık büyümesini kısmen telafi eder. |
| High-frequency ratio | 0.87512 | 0.70721 | Başlangıçtaki çok yüksek frekans içeriği azalır. |
| Spatial roughness | 2.59997 | 1.56371 | Normalize filtre şekilleri daha düzgün hâle gelir. |
| Orientation anisotropy | 0.23420 | 0.35878 | Yönsel eşitsizlik artar. |
| Center-energy ratio | 0.11078 | 0.11104 | Global ortalamada neredeyse sabittir. |

Raw kernel L2 yaklaşık `5,57` kat büyürken BN-folded L2 yalnızca yaklaşık `1,57` kat büyür. Bu ayrışma Conv–BatchNorm ölçek serbestliğinin doğrudan bir örneğidir. Dolayısıyla “ağırlık normu büyüdü, model daha güçlü filtre öğrendi” gibi bir yorum eksiktir; BN parametreleriyle birlikte okunmalıdır.

High-frequency ve roughness azalırken orientation anisotropy'nin yükselmesi, başlangıçta daha gürültü-benzeri ve yönsel olarak dengeli filtre popülasyonunun eğitimle daha düzgün fakat daha yönlenmiş şekillere organize olduğunu düşündürür. Center-energy ortalamasının değişmemesi ise bütün yapısal değişimin merkez katsayıya enerji taşınmasıyla açıklanamayacağını gösterir. Bunlar yine popülasyon özetleridir; tek tek filtre semantiği veya nedensel performans katkısı değildir.

### 10.8. SSL loss ve probe doğruluğuyla birlikte okuma

Checkpoint-aligned bağlam:

| Epoch | SSL loss | Quick probe | Fixed-probe final accuracy |
|---:|---:|---:|---:|
| 1 | 0.537856 | 0.4662 | 0.5151 |
| 10 | 0.254591 | 0.7522 | 0.7764 |
| 20 | 0.223640 | 0.8042 | 0.8208 |
| 30 | 0.206163 | 0.8271 | 0.8408 |
| 60 | 0.173338 | 0.8639 | 0.8746 |
| 90 | 0.155432 | 0.8803 | 0.8905 |
| 110 | 0.150305 | 0.8849 | 0.8913 |
| 120 | 0.148598 | 0.8847 | 0.8905 |

Epoch 30 ile 120 arasında global histogramın finale uzaklığı yalnızca `0.000948` iken quick-probe doğruluğu `0.8271->0.8847`, fixed-probe final doğruluğu `0.8408->0.8905` yükselir. Başka bir deyişle, kaba global normalize-filter dağılımı erken stabilize olduktan sonra temsil kalitesi yaklaşık beş puan daha iyileşmektedir.

Bu sonuç filtre değişiminin performans için önemsiz olduğunu göstermez. Histogram analizi kanal eşleşmesini, belirli filtrelerin hangi girdilerde aktive olduğunu, kanal bileşimlerini, daha sonraki doğrusal olmayan işlevi veya küçük fakat görev açısından önemli parametre hareketlerini ölçmez. Daha doğru çıkarım şudur: geç dönem doğruluk kazanımlarını açıklamak için yalnızca global `3x3` filtre-katsayısı histogramındaki büyük ölçekli yeniden organizasyona ihtiyaç yoktur.

Epoch 110'da quick probe `0.8849` ve fixed-probe final `0.8913` ile kaydedilmiş checkpoint'ler içindeki tepeye ulaşır; epoch 120 değerleri sırasıyla `0.8847` ve `0.8905` ile çok az daha düşüktür. Bu, üçüncü deneyde epoch 110'un seçilmesiyle uyumludur. Fakat fark küçüktür ve tek run'a dayanır; keskin bir optimum kanıtı değildir.

Loss sürekli azalırken probe'un plato yapması ve filter drift'in çok daha erken küçülmesi, bu üç eğrinin farklı nesneleri ölçtüğünü gösterir. Birlikte hareket ettikleri erken dönem korelasyon, belirli bir PCA bileşeninin doğruluğa neden olduğu anlamına gelmez.

### 10.9. Part 3'ün ana yorumu

LeJEPA'nın residual `3x3` filtre popülasyonu eğitim başlangıcında teorik-maksimuma yakın entropy ve random referansına yakın yerel kovaryans yapısı gösterir. İlk 20–30 epoch'ta sabit küresel PCA koordinatlarında büyük bir yeniden organizasyon gerçekleşir: entropy düşer, varyans önde gelen küresel bileşenlerde yoğunlaşır, yüksek-frekans/roughness azalır ve yönsel anizotropi yükselir. Global dağılım bundan sonra çok az değişirken probe doğruluğu ve SSL loss gelişmeye devam eder.

Bu hareket ağ boyunca eşzamanlı ve homojen değildir. Çok sayıda filtre taşıyan layer4 global eğriyi güçlü biçimde etkiler ve erken organize olur; az filtreli layer1 daha uzun süre değişir ve final endpoint'te en büyük stage drift'ini gösterir. Bu nedenle global, stage ve layer grafiklerinin birlikte raporlanması zorunludur.

En savunulabilir sonuç şudur:

> LeJEPA eğitiminde normalize edilmiş `3x3` filtre-şekli dağılımının kaba global organizasyonu erken oluşur; daha sonraki temsil kazanımları büyük global histogram hareketi olmadan devam eder. Bu, geç eğitimin önemsiz olduğu anlamına değil, geç kazanımların global filtre-popülasyonu istatistiğinden daha ince kanal, katman, aktivasyon veya optimizasyon yapılarında gerçekleşebileceği anlamına gelir.

## 11. Bilimsel yorum sınırları

### 11.1. Tek seed

ImageNet ve LeJEPA karşılaştırması bir seed ve bir seçili checkpoint çifti üzerindedir. Derinlik deseni açık ve iki PCA tabanında kararlı olsa da “LeJEPA yöntemi her zaman bu filtre imzasını üretir” sonucu için birden çok bağımsız seed gerekir.

### 11.2. Kanal permütasyonu

Bağımsız eğitilmiş ağlarda kanallar permüte edilebilir. Bu yüzden analiz aynı indeksli iki kernel'i semantik olarak eşleştirmez; dağılım seviyesinde karşılaştırma yapar. Bu tercih doğrudur, fakat dağılım farkı fonksiyonel nedensellik kanıtı değildir.

### 11.3. Histogram ve bin sayısı

Drift, 70-bin coefficient histogramlarından ve symmetric KL'den türetilir. Üçüncü deneyle karşılaştırılabilirlik için aynı protokol korunmuştur. Histogram tabanlı değerler bin sayısına ve edge protokolüne duyarlı olabilir. Eğitim dinamiğinde bütün epoch'lar için ortak sabit edge kullanılması bu duyarlılığın zaman karşılaştırmasını bozmasını önler.

### 11.4. PCA tabanı ve bileşen semantiği

Küresel taban karşılaştırmayı modelden bağımsızlaştırır; fakat her bileşenin evrensel bir semantik etiketi olduğu anlamına gelmez. En güçlü yorumlar total drift, stage/layer localization ve farklı temsil kontrollerinde tekrarlanan desenlerdir. Bileşen işareti ve numarası daha ihtiyatlı yorumlanmalıdır.

### 11.5. Entropi ve sparsity kalite skoru değildir

Düşük entropy redundansı, yüksek entropy random benzeri varyans spektrumunu, sparsity ise near-zero filtre oranını gösterebilir. Ancak bunların hiçbiri tek başına model doğruluğu veya temsil kalitesi değildir. Özellikle LeJEPA'nın daha yüksek ortalama `H` göstermesi doğrudan “daha iyi” demek değildir; yalnızca seçili checkpoint'te varyansın dokuz yerel yönde daha eşit dağıldığını söyler.

## 12. Genel değerlendirme

Denetimden sonra dördüncü deneylerin ana metodolojik yapısı geçerli bulunmuştur. Eksik kalan iki önemli uygulama sorunu—hedef filtrelerin global artifact ile aynı dtype/normalizasyon sırasını kullanması ve LeJEPA dynamics betiğinin tamamlanması—nihai işlerden önce düzeltilmiştir. Heatmap'lerin chart karşılıkları eklenmiş ve eski heatmap'ler korunmuştur.

Part 1, sparsity farkının yalnızca tek bir epsilon seçimine bağlı olmadığını; düşük-entropili son ImageNet katmanının geniş bir cutoff aralığında kararlı olduğunu; iki modelde de makalenin gerçek random sınırını aşan katman bulunmadığını göstermektedir.

Part 2, ImageNet–LeJEPA global drift ve derinlik deseninin yalnızca iki modele fit edilen PCA tabanından kaynaklanmadığını göstermektedir. Global D yaklaşık `%3,27` değişse de en baskın son-layer farkı ve genel stage sıralaması korunmuştur. PCA-bağımsız kontrol tablolarının bit düzeyinde aynı olması, checkpoint yeniden üretimi ve analiz akışının güvenilirliğini güçlendirmektedir.

Part 3, LeJEPA'nın global normalize-filtre dağılımındaki en büyük yeniden organizasyonun ilk 20–30 epoch'ta gerçekleştiğini göstermektedir. Epoch 1->120 global drift'i `0.419884` olsa da epoch 30'un finale uzaklığı yalnızca `0.000948`'dir. Buna karşılık fixed-probe doğruluğu epoch 30'dan 120'ye `0.8408`'den `0.8905`'e yükselir. Stage/layer ayrımı, az filtre taşıyan layer1'in global plato sonrasında da değişmeye devam ettiğini; entropy ve descriptor kontrolleri ise başlangıçtaki random-referansa yakın yüksek-entropili şekillerin daha yönlenmiş ve daha düşük-frekanslı bir popülasyona organize olduğunu göstermektedir. Bu ilişki betimseldir; nedensellik iddiası değildir.

En doğru sonuç dili şudur:

> Seçili ve yakın performanslı RN18 checkpoint çiftinde LeJEPA, ImageNet'ten tamamen ayrı bir filtre evreni üretmemektedir. Farklar global olarak sınırlı fakat derinliğe göre yapılandırılmıştır; en güçlü ayrışma son residual blokta, ikincil ayrışma erken residual bloklarda görülür. LeJEPA ayrıca seçili checkpoint'te daha az near-zero filtre ve daha yüksek yerel varyans-spektrum entropisi göstermektedir. Bu desenler küresel, modelden bağımsız CNN Filter DB PCA tabanında da korunur; fakat yöntem-genel bir iddia için çoklu seed doğrulaması gereklidir.

## 13. Temel çıktı yolları

```text
outputs/(4thEXP)rn18_cifar10_filter_dynamics/
  original_cnn_filter_db_pca/
    cnn_filter_db_full_pca.npz
  regenerated_imagenet_baseline/
  part1_threshold_sensitivity/
  part2_original_basis_pair/
  lejepa_training_dynamics/

outputs/(3rdEXP)(theONE)rn18_cifar10_true_lejepa_cfg_l02_v4_p64_strong/
  analysis_global_cnn_filter_db_basis_epoch001_vs_epoch110/
```

## 14. Denetimde kullanılan başlıca yerel kaynaklar

- `papers/CNN_Filter_DB(Paper).pdf`
- özgün Git geçmişindeki ve güncel commit'teki `main.ipynb`
- `(3)EXPERIMENT_PROGRESS.md`
- `scripts/compute_cnn_filter_db_pca.py`
- `scripts/analyze_rn18_threshold_sensitivity.py`
- `scripts/analyze_rn18_cifar10_filters.py`
- `scripts/analyze_rn18_cifar10_kernel_distributions.py`
- `scripts/analyze_rn18_lejepa_training_dynamics.py`
- `scripts/rn18_cifar10_common.py`
- eski ve yeniden üretilmiş ImageNet metrik CSV'leri
- üçüncü deneyin eski iki-model PCA ve kernel-distribution sonuçları
