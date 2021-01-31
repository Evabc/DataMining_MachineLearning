#### 1.1.1 什麼是決策樹(Decision Tree)？
用來處理分類問題的樹(廢話)，他的結構如下圖：
內部節點(internal nodes)表示一個評估屬性
e.g. 以會不會看銀魂電影版的例子，第一個選出的屬性是「有沒有看過前作」。
葉子節點(leaf nodes)表示根據此評估項分類後，代表分類的label
而「不去看」這個選項便是葉子節點，但如果還可以繼續分，我們就用「年齡」繼續做切割。
![image1](https://github.com/Evabc/DataMining_MachineLearning/blob/master/1_Classification/1.1_%E6%B1%BA%E7%AD%96%E6%A8%B9_Decision%20Trees/image/1.JPG)

#### 1.1.2 基本的演算法概念
基本上決策樹的概念有三個重點，後續再針對每個重要的地方做說明
(1) 資料設定：將原始資料分成training data及testing data 
(2) 決策樹生成：參考1.1.3
(3) 剪枝：使用testing data來進行決策樹修剪
以上(1)~(3)步驟不斷重複進行，直到所有的新產生節點都是樹葉節點為止。

#### 1.1.3 樹的生成
既然要長出一顆樹，最關鍵的就是怎麼決定內部節點，如果選不好那這顆樹就毫無鑑別力了...囧
所以我們把樹的生成規納為幾個步驟：
(1) 依據屬性選擇指標，從現有未被挑選過的屬性中，選出分類能力最好的屬性做為樹的內部節點。
(2) data依選出的內部節點產生出對應的分支。
(3) 產生出的分枝可能還有屬性可以分，那就是再選內部節點。
    如果沒有屬性可以分，就直接產生對應的樹葉節點(也就是label)。

#### 1.1.4 屬性選擇指標
在上圖可以看到優先使用「有沒有看過前作」這個屬性當第一個分割依據，但為什麼不是先用「年齡」來分呢?，用什麼方式可以判斷屬性好壞? 可以用下面資訊獲利跟吉尼係數來判斷屬性好壞：
所謂的屬性好壞，白話文說法就是我們分的乾不乾淨，這個屬性有沒有辦法把現有的data給清楚劃分。
換個高級講法來就是，檢驗分出來節點的**不純度(Node Impurity)**
直觀的例子來！以下圖來看，隨便辦公室抓10個肥宅，6個人有看電影版，4個沒看。你覺得那個屬性可以清楚的分出「會去電影院看銀魂電影版」的人?
![image2](https://github.com/Evabc/DataMining_MachineLearning/blob/master/1_Classification/1.1_%E6%B1%BA%E7%AD%96%E6%A8%B9_Decision%20Trees/image/2.JPG)
所以，不純度越低越好 (分出來的data越一致越好棒棒!)


* **資訊獲利 (Information Gain)** 代表ID3、C4.5、C5.0
以Entropy為基礎，白話解釋為「亂度」，也就是說Entropy越大，亂度越大，越亂。
Entropy公式為：如果資料集合 S有 c 個不同的類別，pi 為類別 i 在資料集合 S 出現的機率
<img src="http://chart.googleapis.com/chart?cht=tx&chl= $$Entropy(S)=\sum_{i=0}^{c}-p_i \log_2 p_i$$" style="border:none;">

這樣看太沒有帶入感，直接把上面的例子拿來算一次：
<img src="http://chart.googleapis.com/chart?cht=tx&chl= $$Entropy(fat)= -\frac{6}{10} \times \log_2(\frac{6}{10})-\frac{4}{10} \times \log_2(\frac{4}{10}) = 0.97$$" style="border:none;">
    

如果是屬性A 在資料集合S 的Information Gain：
<img src="http://chart.googleapis.com/chart?cht=tx&chl= $$Gain(S,A)=Entropy(S)-\sum_{j=0}^{v}\frac{S_j}{S}Entropy(S_j)$$" style="border:none;">

Entropy(S) : 資料集合S的亂度

Entropy(Sj) : 資料子集合Sj 的亂度
一樣把上面的例子算一次

<img src="http://chart.googleapis.com/chart?cht=tx&chl= $$Entropy(age<40)= -\frac{3}{5} \times \log_2(\frac{3}{5})-\frac{2}{5} \times \log_2(\frac{2}{5}) = 0.97$$" style="border:none;">

<img src="http://chart.googleapis.com/chart?cht=tx&chl= $$Entropy(age\geq40)= -\frac{3}{5} \times \log_2(\frac{3}{5})-\frac{2}{5} \times \log_2(\frac{2}{5}) = 0.97$$" style="border:none;">

<img src="http://chart.googleapis.com/chart?cht=tx&chl= $$Gain(fat,age)=0.97-(\frac{5}{10})0.97-(\frac{5}{10})0.97=0$$" style="border:none;">

    
啊勒！這樣等於0 不就完全沒獲益了！
年齡屬性表示：
![image3](https://github.com/Evabc/DataMining_MachineLearning/blob/master/1_Classification/1.1_%E6%B1%BA%E7%AD%96%E6%A8%B9_Decision%20Trees/image/3.JPG)
真不該寄望這個~~廢物~~屬性，看另一個屬性!

<img src="http://chart.googleapis.com/chart?cht=tx&chl= $$Entropy(Watch Series)= -\frac{4}{5} \times \log_2(\frac{4}{5})-\frac{1}{5} \times \log_2(\frac{1}{5}) = 0.72$$" style="border:none;">

<img src="http://chart.googleapis.com/chart?cht=tx&chl= $$Entropy(No Watch Series)= -\frac{5}{5} \times \log_2(\frac{5}{5})-0 \times \log_2(0) = 0$$" style="border:none;">

<img src="http://chart.googleapis.com/chart?cht=tx&chl= $$Gain(fat,series)=0.97-(\frac{5}{10})0.72-(\frac{5}{10})0=0.61$$" style="border:none;">

所以，我們可以發現：
Gain的值越大，某屬性分出來的凌亂程度越小，用來分資料越好！

Gain的值越小，某屬性分出來的凌亂程度越大，用來分資料越差！


* **吉尼係數 (Gini Index)** – 代表CART
資料集合S包含n個類別， 為在S中那些值屬於類別j的機率，寫成：
<img src="http://chart.googleapis.com/chart?cht=tx&chl= $$Gini(S)=1-\sum_{j=1}^{n}p_j^2$$" style="border:none;">

用我們肥宅的例子寫來就是：<img src="http://chart.googleapis.com/chart?cht=tx&chl= $$Gini(fat)=1-(\frac{6}{10})^2-(\frac{4}{10})^2=0.48$$" style="border:none;">
如果用屬性A去分割資料集S，分成S1跟S2，則根據A為分割條件的Gini係數寫作：
<img src="http://chart.googleapis.com/chart?cht=tx&chl= $$Gini_A(S)=\frac{S_1}{S}Gini(S_1)+\frac{S_2}{S}Gini(S_2)$$" style="border:none;">

再把我們的例子帶進去看一下!
<img src="http://chart.googleapis.com/chart?cht=tx&chl= $$Gini(NoWatchSeries)=1-(\frac{4}{5})^2-(\frac{1}{5})^2=0.32$$" style="border:none;">

<img src="http://chart.googleapis.com/chart?cht=tx&chl= $$Gini(WatchSeries)=1-(\frac{5}{5})^2-(0)^2=0$$" style="border:none;">

<img src="http://chart.googleapis.com/chart?cht=tx&chl= $$Giniseries (fat)=(\frac{5}{10})\times (0.32)+(\frac{5}{10})\times (0)=0.16$$" style="border:none;">

同樣的算法我們算出：
<img src="http://chart.googleapis.com/chart?cht=tx&chl= $$Giniage(fat)=(\frac{5}{10})\times (0.48)+(\frac{5}{10})\times (0.48)=0.48$$" style="border:none;">
所以，我們可以發現：

Gini的值越小，某屬性分出來的凌亂程度越小，用來分資料越好！

Gain的值越大，某屬性分出來的凌亂程度越大，用來分資料越差！

![image4](https://github.com/Evabc/DataMining_MachineLearning/blob/master/1_Classification/1.1_%E6%B1%BA%E7%AD%96%E6%A8%B9_Decision%20Trees/image/4.JPG)

#### 1.1.5 決策樹演算法
(1) ID3演算法：ID3在建構決策樹過程中，以資訊獲利為標準，選擇最大的資訊獲利值作為分類屬性。不過ID3有個缺點！你會發現它所使用的資訊獲利會傾向選擇擁有許多不同數值的屬性，例如：“電影院座位”欄位中，每一個座位的編號皆不同。若依座位編號進行分割，會產生出許多分支，且每一個分支都是很單一的結果，其資訊獲利會最大。但這個屬性對於建立決策樹是沒有意義的。所以....下面一位！

(2) C4.5演算法：C4.5演算法利用屬性的獲利比率(Gain Ratio)克服上面這個問題，簡單來說就是把分支的數量也考慮進去。
我們要求某屬性A的Gain Ratio時，除資訊獲利外，尚需計算該屬性的分割資訊值(Split Information)，公式：<img src="http://chart.googleapis.com/chart?cht=tx&chl= $$SplitInfo_A(S)=-\sum_{j=1}^{v}\frac{S_j}{S} \times \log_2(\frac{S_j}{S})$$" style="border:none;">

所以，某個屬性A的Gain Ratio公式為：**GainRatio(A) = Gain(S, A)/SplitInfoA(S)**

直接看例子，假設現在多了一個「哪裡人」的屬性，好巧不巧10個人都來自不同縣市，這時候：
<img src="http://chart.googleapis.com/chart?cht=tx&chl= $$Entropy(city)=(-1)\times\log_2(1)-(0)\times\log_2(0)...=0$$" style="border:none;">

<img src="http://chart.googleapis.com/chart?cht=tx&chl= $$Gain(fat,city)=0.97-(\frac{1}{10})0-(\frac{1}{10})0...=0.97$$" style="border:none;">

這個0.97已經遠勝於我們剛剛選的前作（0.61）了啊啊！
可是你覺得靠出身地來預測會不會看銀魂電影版合理嗎？
![image5](https://github.com/Evabc/DataMining_MachineLearning/blob/master/1_Classification/1.1_%E6%B1%BA%E7%AD%96%E6%A8%B9_Decision%20Trees/image/5.JPG)

沒錯！所以我們再來看看如果用GainRatio會如何？
<img src="http://chart.googleapis.com/chart?cht=tx&chl= $$SplitInfo_city (fat)=-\frac{1}{10}\times\log_2(\frac{1}{10})\times10=3.32$$" style="border:none;">

<img src="http://chart.googleapis.com/chart?cht=tx&chl= $$SplitInfo_series (fat)=-\frac{5}{10}\times\log_2(\frac{5}{10})-\frac{5}{10}\times\log_2(\frac{5}{10})=1$$" style="border:none;">

<img src="http://chart.googleapis.com/chart?cht=tx&chl= $$GainRatio(city)=\frac{0.97}{3.32}= 0.29$$" style="border:none;">

<img src="http://chart.googleapis.com/chart?cht=tx&chl= $$GainRatio(series)=\frac{0.61}{1}= 0.61$$" style="border:none;">

從上面可以看出，本來用哪裡人屬性可以得到很高的gain，改用ratio後，它就不再是個很好的依據了。

(3) CART演算法：全名為 Classification and Regression Tree ，CART可以產生分類樹及迴歸樹，但它只能產生「二元樹」的技術(ID3 及C4.5 則無此限制)，分類樹以Gini 吉尼係數做為選擇屬性的依據，迴歸樹則以平方差最小為依據。

#### 1.1.6 決策樹剪枝
為什麼要剪枝呢? 因為決策樹很可能會有model overfitting的問題(也就是分太細及自以為啦)
分太細是指：之前例子只以40歲為分界，如果再細分成1-5,6-10...這樣可能就會受到分類太多影響，造成該定律無法適用到其他資料。
自以為是指：假設剛好那些40歲以下會看的人全落在26-30歲，這樣當我們拿這個決策樹來預測「會不會看我的英雄學院劇場版」時，這個樹自然會傾向於把26-30歲人分到會看。

所以，修剪決策樹可移除不可信賴的分支。有兩種修剪方法：
* 事前修剪 (Prepruning)：當選擇某個屬性做為決策樹的一個內部節點，若這個屬
性的指標值低於我們是先決定好的臨界值時，則應該停止這節點及其以下的所有子節點成長。
* 事後修剪 (Postpruning)：先讓它生成一顆完整的決策樹，然後從底部往上對所有內部結點進行檢查，如果這個結點對應的子樹，換為葉子結點能帶來泛化性能提升，則將該子樹替換為葉子結點。
