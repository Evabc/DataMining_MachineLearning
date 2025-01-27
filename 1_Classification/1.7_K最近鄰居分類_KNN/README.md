#### 1.7.1 什麼是KNN

KNN全名是**k nearest neighbor**：根據現有已分類好的資料集合，找出K個與待分類資料最鄰近的現存資料，然後根據這K個最鄰近資料之類別，以多數決的方式決定待分類資料的所屬類別。

先舉個例子吧！

先看漫畫分類這個表格中，有漫畫名稱、該漫畫中出現幾次踹人的動作、該漫畫中出現幾次接吻的動作及漫畫連載在那個週刊這些資料。現在有個新銳漫畫家阿毛，完成了第一部原創漫畫「真不想上班阿！」，正在煩惱該將這部漫畫投稿到那個刊物...JUMP?還是Ribon?
我們把這個表格轉換成二維表格來看，好像又更清楚了

![](https://github.com/Evabc/DataMining_MachineLearning/blob/master/1_Classification/1.7_K%E6%9C%80%E8%BF%91%E9%84%B0%E5%B1%85%E5%88%86%E9%A1%9E_KNN/image/1.JPG "1")

那直覺來看，我們會看到「真不想上班阿！」離銀魂跟鋼鍊蠻近的，會傾向推薦投稿JUMP

但KNN的方式是，假設現在K=3，根據「真不想上班阿！」最近的3個漫畫來看，2個在JUMP，1個在Ribon，那KNN就會推薦投稿JUMP

#### 1.7.2 怎麼選擇K? 距離怎麼算?

上面提到K這個值也就是我們要參考的鄰居數量，一般來說都會選擇奇數(還不會發生同票嘛~)
當K太小時候，參考的數量太少，只憑著最近的一點就做判斷，容易overfitting
當K太小時候，參考的數量太多，這樣找最近的鄰居都沒意義了，容易underfitting

當我們決定好K值後，要怎麼計算預測目標離某個鄰居的距離呢?
通常採用歐幾里德距(Euclidean Distance)

∥X-Y∥ = <img src="http://chart.googleapis.com/chart?cht=tx&chl= \sqrt{(x_1-y_1)^2+(x_2-y_2)^2\ldots+(x_n-y_n)^2}" style="border:none;">

#### 1.7.3 KNN的問題是什麼? 該如何解決?

KNN在樣本資料數量龐大時，仍能保有一些錯誤指標的不錯表現。但其主要缺點如下：
* 計算量大 (過多的距離計算)
對樣本資料集合進行組織與整理，藉由分群分層的方式，儘可能將計算量壓縮到小範圍以內，避免盲目地與每個樣本資料進行距離計算。可用**最近鄰居快速搜索演算法**：
(1)透過將樣本資料集合的分解成數個不相交的子集合，每個子集合又可以再繼續往下做分解。進行幾個回合後，即可得到一個樣本集合的樹狀結構。
(2)快速搜索中的:群間過濾及群內過濾

* 記憶空間需求量大 (需儲存所有資料於記憶體)
在原有樣本資料集合中，挑選出對分類計算有效的樣本，使樣本總數合理地減少，以同時達到既減少計算量，又減少記憶量的雙重效果。可用**剪輯近鄰法**及**壓縮近鄰法**：

    **剪輯近鄰法**：對現有樣本資料集合進行剪輯，將不同類別交界處(重疊部分)的樣本資料以適當方式加以篩選，使得剩下的樣本資料形成兩個好的資料群組，而且每個群組內的資料都屬於同一類。可以實現既減少樣本數目、又可提高正確識別率的雙重目的。不過在樣本資料量的壓縮方面並不十分明顯，因為其主要作用在於將原始樣本集合中，把位於邊界處的不同類別交疊樣本刪除掉，但靠近類別中心的大部分樣本仍被保留下來。
    簡言之*就是把不乾淨的地方刪一刪，但刪的不夠多阿！所以需要下面的壓縮近鄰法*
    
    ![](https://github.com/Evabc/DataMining_MachineLearning/blob/master/1_Classification/1.7_K%E6%9C%80%E8%BF%91%E9%84%B0%E5%B1%85%E5%88%86%E9%A1%9E_KNN/image/3.JPG "3")

    **壓縮近鄰法**：以壓縮近鄰法來看，那些靠近類別中心的樣本大多數對分類沒什麼作用(因為他們就已經很明確了，再去計算他們意義不大)。因此，如果能在剪輯的基礎上，再去除掉一部分這類的樣本，將有助於進一步縮短計算時間與壓縮記憶體需求量，此方法稱為壓縮近鄰法。其實我覺得跟SVM的感覺有點類似，其實最重要的只要能找到分開的支持向量就好，那些已經被明確劃分的樣本其實就不再那麼重要了～

![](https://github.com/Evabc/DataMining_MachineLearning/blob/master/1_Classification/1.7_K%E6%9C%80%E8%BF%91%E9%84%B0%E5%B1%85%E5%88%86%E9%A1%9E_KNN/image/2.JPG "2")

參考資料：

1.https://medium.com/@NorthBei/machine-learning-knn%E5%88%86%E9%A1%9E%E6%BC%94%E7%AE%97%E6%B3%95-b3e9b5aea8df

2.http://debussy.im.nuu.edu.tw/sjchen/MachineLearning/final/CLS_Nearest-neighbor.pdf
