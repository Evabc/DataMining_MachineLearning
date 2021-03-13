#### 1.8.1 什麼是貝氏定理（Bayes' theorem）
簡單來說就是：根據後來發生的事，再搭配過去經驗，會改變我們一開始的想法。
先上公式,再看例子!

P(A|B)=<img src="http://chart.googleapis.com/chart?cht=tx&chl= \frac{P(A)P(B|A)}{P(B)}" style="border:none;">

什麼是**P(A|B)**? 事件 B 已發生的情況下，再發生另一事件 A 之機率，稱為條件機率。

套個例子來講，當富奸更新獵人連載後，獵人有望在我此生完結的機率=>P(獵人完結|富奸更新連載)

P(A|B)=P(A∩B)/P(B)

![](https://github.com/Evabc/DataMining_MachineLearning/blob/master/1_Classification/1.8_%E8%B2%9D%E6%B0%8F%E5%88%86%EF%A7%90_Bayes/image/1.JPG "1")

#### 1.8.2 什麼是單純貝氏分類 (Naïve Bayes)
單純貝氏分類法假定所有變數(屬性)對分類均是有用的，且這些變數間是相互獨立。常用在文本分類:像垃圾信分類那種。其實也可以把單純貝氏分類當做貝氏信念網路的特殊狀況,這因為網路不邊際,就把各節點想成獨立的吧!

就拿很多課本上都有的天氣圖來當範例算一次吧!

![](https://github.com/Evabc/DataMining_MachineLearning/blob/master/1_Classification/1.8_%E8%B2%9D%E6%B0%8F%E5%88%86%EF%A7%90_Bayes/image/2.JPG "2")

要預測在何種天氣模式下，某人會不會去打網球? 假設現在有一組待分類的天氣模式如下：

Outlook= Sunny, Temp.= Cool, Humidity= High,Wind= Strong, Play Tennis= ?

根據上面的公式,我們把每個選項都看做是獨立的,可以寫成下面的樣子：

不過因為我們預測會不會打球,兩個分母都是一樣的,通常我們就先不算,只比分子即可

P(會打網球|O=S,T=C,H=H,W=S)= P(14天中打球的日子)×P(O=S,T=C,H=H,W=S|會打網球)
= 9/14 × 2/9×3/9×3/9×3/9 = 0.0053

P(不打網球|O=S,T=C,H=H,W=S)= P(14天中不打球的日子)×P(O=S,T=C,H=H,W=S|不打網球)
= 5/14× 3/5×1/5×4/5×3/5 = 0.0206

還透過標準化(Normalizing)的步驟，將上述的計算結果加以轉換以滿足機率總合為 1的要求。

會打球 = 0.0053/0.0053+0.0206 = 20.5%

不打球 = 0.0206/0.0053+0.0206 = 79.5%

根據計算結果，此待分類的天氣屬性， Play Tennis= **NO**。

#### 1.8.3 什麼是貝氏信念網路(Bayesian Belief Networks; BBN)
但其實我們都知道，大部份變數間都有某種程度的關聯性。所以單純貝氏分類可能就會造成錯誤
貝氏信念網路 (簡稱貝氏網路) 則允許指定哪些屬性需符合條件獨立,並提供一個利用圖形模式，表現出網路中變數間的因果關係，其中包含兩個重要部份是：
(1) 用有向的非循環圖表示變數間的相依關係
* 圖(a)中，變數A與B相互獨立，且都會直接影響第三個變數C。變數A與B是變數C的父節點，C為A與B的子節點。而變數W與這三個變數獨立
* 圖(b)中，從變數D到變數A(或B)有一條非直接的有向路徑存在，故節點D是節點A(或B)的祖先，而A(或B)是D的孫節點。
* 前面所提的簡單貝氏分類法中的條件獨立，可繪製成圖(C)。其中 bj 是目標類別，而{H1, …, Hr}是屬性集合。

![](https://github.com/Evabc/DataMining_MachineLearning/blob/master/1_Classification/1.8_%E8%B2%9D%E6%B0%8F%E5%88%86%EF%A7%90_Bayes/image/3.JPG "3")

(2)用機率表記錄每個節點和它的直接父節點間的關聯性
* 如果節點X沒有任何父節點，則表中紅字僅包含事前機率P(X)
* 如果節點X只有一個父節點Y，則表中綠字將包含條件機率P(X|Y)
* 如果節點X有多個父節點{Y1, Y2, …, Yk}，則表中紫字將包含條件機率P(X|Y1, Y2, …, Yk)
也就是說貝氏網路的每個節點間之關聯性，會表現於機率表中，如下圖。

![](https://github.com/Evabc/DataMining_MachineLearning/blob/master/1_Classification/1.8_%E8%B2%9D%E6%B0%8F%E5%88%86%EF%A7%90_Bayes/image/4.JPG "4")

舉個例子，假設有兩個伺服器 (S1, S2)，會傳送封包到使用者端(U)，但是第二個伺服器的封包傳送成功率會與第一個伺服器傳送成功與否有關，因此此貝氏網路的結構可以表示如下。

![](https://github.com/Evabc/DataMining_MachineLearning/blob/master/1_Classification/1.8_%E8%B2%9D%E6%B0%8F%E5%88%86%EF%A7%90_Bayes/image/5.JPG "5")

假設已知使用者端成功接受到封包，求第一伺服器成功發送封包的機率 P(S1=T|U=T)? 

P(S1=T|U=T)=68.96% 我們把他拆成分子分母來看就很清楚了

分母 = 所有U成功的機率 
= S1成功S2成功 + S1成功S2失敗 + S1失敗S2成功 + S1失敗S2失敗

= (0.4×0.7×1) + (0.4×0.3×1) + (0.6×0.3×1) + (0.6×0.7×0) = 0.58

分子 = S1成功S2成功 + S1成功S2失敗 = (0.4×0.7×1) + (0.4×0.3×1)=0.4

參考資料：
1.http://debussy.im.nuu.edu.tw/sjchen/MachineLearning/final/CLS_Bayesian.pdf
