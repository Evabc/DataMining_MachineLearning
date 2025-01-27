##### 1.5.1什麼是LightGBM
GBDT在訓練時都需要遍歷整訓練集多次，如果把整個訓練data裝進內存則會限制訓練數據的大小，如果不裝進內存，反覆讀寫又會消耗非常多時間。XGBoost的預排序方法，優點是能精確地找到分割點，但是缺點也很明顯：空間消耗大，不只要保存數據的特徵值，還要保存特徵排序的結果，且在遍歷切分點也十分耗間。
LightGBM提出的主因就是為了解決GBDT在大量數據遇到的問題，及XGBoost 的耗時耗空間，比起來LightGBM具有訓練速度快、內存佔用低的特點。基本上LightGBM主要用了3個方式來改善XGBoost：
* Histogram (直方圖算法)
* GOSS (Gradient-based One-Side Sampling 單邊梯度採樣算法)
* EFB (Exclusive Feature Bundling 互斥特徵綁定算法)

以下一個個稍微介紹觀念，這邊就不講太多算法的部份XD 有興趣再上網看

![](https://github.com/Evabc/DataMining_MachineLearning/blob/master/1_Classification/1.5_LightGBM/image/1.jpg "image1")

##### 1.5.2 Histogram (直方圖算法)
直方圖算法是替代XGBoost的預排序(pre-sorted)算法的，XGBoost預排序算法是按照屬性的值排序，然後從全部屬性的值中找到最優的分裂點位，這邊不難想像吧，假設某屬性有5個值，XGBoost要評估4個切分點，如果今天換成500個值，最多就可能會有499個切分點(當然不一定是，因為重覆的值不算)。
而LightGBM的直方圖算法有點類似分組的概念，將連續特徵值離散化到固定數量(如10個)的bins上，這樣我們只需要遍歷bins的切分點(10 bins=>也就是9個切分點)，這樣是不是少很多！

此外，直方圖算法還能夠作直方圖差加速。當節點分裂成兩個時，右邊葉子節點的直方圖等於其父節點的直方圖減去左邊葉子節點的直方圖。這樣做有什麼好處呢? LightGBM就可以挑比較直方圖小的葉子節點去計算，就可以利用直方圖做差來獲得直方圖大的葉子節點，也因為這樣從而大大減少構建直方圖的計算量。

![](https://github.com/Evabc/DataMining_MachineLearning/blob/master/1_Classification/1.5_LightGBM/image/2.jpg "image2")

##### 1.5.3 GOSS (單邊梯度採樣算法)
簡單來說就是，丟掉用不到的樣本(梯度小於設定值)，只留對我們有幫助的樣本(梯度大於設定值)！
對全部樣本進行隨機採樣，可能會對目標函數的gain計算造成影響(例如抽到太多異常值之類的)，所以GOSS 只對梯度絕對值較小的樣本按照一定比例進行抽選，而保留了梯度絕對值較大的樣本，以達到減少計算gain的複雜度。所以這裡的One-Side，就是因為目標函數的gain主要來自於梯度絕對值較大的樣本。

![](https://github.com/Evabc/DataMining_MachineLearning/blob/master/1_Classification/1.5_LightGBM/image/3.jpg "image3")

##### 1.5.4 EFB (互斥特徵綁定算法)
在高維度的數據往往是稀疏的，而且特徵間可能是相互排斥的（即特徵不會同時為非零值，像one-hot），這樣的話，如果把兩個特徵捆綁起來，就可以有效的降低維度，又不會造成缺失。其實簡單來想就是把簡單來說就是將多個特徵壓縮成一個的手法。

當然前面也提到了，LightGBM簡單來說就是快！狠！準！只能說Boosting家族出來的人，各各都是狠角色，將來說不定還有更強的人出來呢....

![](https://github.com/Evabc/DataMining_MachineLearning/blob/master/1_Classification/1.5_LightGBM/image/4.jpg "image4")

參考資料1：https://blog.csdn.net/dQCFKyQDXYm3F8rB0/article/details/103248372


參考資料2：https://medium.com/@jimmywu0621/%E6%B1%BA%E7%AD%96%E6%A8%B9%E7%B3%BB%E5%88%97-lightgbm%E6%A8%A1%E5%9E%8B%E7%90%86%E8%AB%96-96ce38ea8940


