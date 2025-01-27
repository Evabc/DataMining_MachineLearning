在開始講隨機森林之前，想先把跟它有關的Ensemble learning(集成學習)先解釋一下。
我們可以很直觀的想，當我們打王的時候，一個人打不過，當然就會想找很多人一起打!
這就跟一顆樹不夠分的不夠好的時候，當然就要靠很多樹一起來做一樣的概念。
所以Ensemble learning可以說是Random Forest的重點。
#### 1.2.1 產生多個分類器 (就是要很多樹)
下面介紹的Bagging、Boosting和Adaboost都是產生多個分類器的手法：

* **Bagging**: Bagging概念是從訓練資料中隨機抽取樣本訓練多個分類器(要多少個分類器自己設定)，每個分類器的權重一致最後用投票方式(Majority vote)得到最終結果。Bagging常用於單一分類器容易overfitting的時候，如果模型不是分類問題而是預測的問題，分類器部份也可以改成regression，最後投票方式改成算平均數。
另外補充什麼是**Out-of-bag(OBB)**：簡單來說就是我們使用沒被抽出的樣本，來當做validation set。

    e.g. 假設我們抽100位已經看過鬼滅之刃電影版的肥宅，發現裡面有5位肥宅他們被銀魂電影宣傳給誤導，以為他們看的是鬼滅，結果根本不是這回事阿！	
照之前的作法我們把這100丟下去訓練成一顆樹，但裡面會參雜了根本沒看鬼滅的人，可能會影響了我們樹的分類，造成預測不好。
如果用了Bagging的方式，可能每次都是抽80位肥宅來訓練成一顆樹，就有機會避免掉抽到那些根本沒看的肥宅，產生多棵樹之後，每棵樹權重一致(可以想成一人一票)，選出分數最高的那棵樹。

![](https://github.com/Evabc/DataMining_MachineLearning/blob/master/1_Classification/1.2_%E9%9A%A8%E6%A9%9F%E6%A3%AE%E6%9E%97_Random%20Forest/image/1.JPG "image1")

* **Boosting**: Boosting是一種將多個弱分離器組合起來形成強分類器的算法框架。它先選一個很爛的分類器，再去找第二個分類器去彌補第一個分類器不足(做不好的資料)的部份，就一直這樣找下去就可以找出很多個分類器(有順序的找)。和Bagging不同的地方在，Bagging產生dataset的方式就是隨機抽(每個data的權重都是1)，而Boosting的話，他對於每筆data的權重就不會都是一致的，根據不同的權重所選出的data不一樣，就會造出不一樣的分類器囉! Boosting他的方法有很多種，其中一種便是AdaBoost。

* **AdaBoost**: AdaBoost是對上一個分類器分類錯的資料提高資料的權重，對分對的資料則降低資料的權重，再藉由這些重新分配權重的資料去訓練出新的分類器。就用上面的方式訓練出一堆分類器後，再根據他們的錯誤率來給予不同分類器權重，好的分類器權重就高一點，不好的分類器權重就低一點，再把每個分類器對這筆資料的判別結果乘上他們本身的權重後加起來，得到最終的結果。但AdaBoost對訓練資料的噪聲(那些沒看鬼滅的肥宅)非常敏感，如果訓練資料裡的噪聲資料很多，那後面分類器都會集中在進行噪聲資料上分類，反而會影響最終的分類性能。

#### 1.2.2Bagging與AdaBoost的區別之處：
* 訓練樣本:
Bagging: 每一次的訓練集是隨機抽取(每個樣本權重一致)，抽出可放回，以獨立同分布選取的訓練樣本子集訓練弱分類器。
AdaBoost: 每一次的訓練集不變，但選擇的訓練集都是依賴上一次學習得結果，通過權重改變來產生不同的訓練集，再根據錯誤率調整權重後，再取樣。
* 分類器的取得與權重:
Bagging: 每個分類器可以並行生成，權重為初始給予，每個分類器的權重相等。
AdaBoost: 每個弱分類器只能依賴上一次的分類器順序生成，也都都有相應的權重，好的分類器權重大，不好的權重小。
* 如何把多個分類器集合起來:
Bagging: 所以分類器加起來選票數高的 (分到A的2票，分到B的1票 => 最終選擇為A)
AdaBoost: 分類器先乘上自身的權重後，再做加總(分到A的2票(權重為0.3跟0.2)，分到B的1票(權重0.8)=>最終選擇為B)。

#### 1.2.3 什麼是Random Forest(隨機森林)
在上面我們解釋過什麼是Bagging後，其實*Random Forest = Bagging + Decision tree* 

![](https://github.com/Evabc/DataMining_MachineLearning/blob/master/1_Classification/1.2_%E9%9A%A8%E6%A9%9F%E6%A3%AE%E6%9E%97_Random%20Forest/image/2.JPG "image2")

補充一點：Random Forest不進行剪枝，Decision tree剪枝是因為防止overfitting，而Random Forest的“隨機”已經防止了過擬合，因此不需要剪枝。

你一定會想，阿上面講了一堆AdaBoost好像跟Random Forest沒什麼關係耶? 
對，他們不一樣XD 
**Boosting Tree : AdaBoost + Decision**
![](https://github.com/Evabc/DataMining_MachineLearning/blob/master/1_Classification/1.2_%E9%9A%A8%E6%A9%9F%E6%A3%AE%E6%9E%97_Random%20Forest/image/3.JPG "image3")
