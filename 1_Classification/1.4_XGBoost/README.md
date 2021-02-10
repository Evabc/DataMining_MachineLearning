##### 1.4.1什麼是XGBoost
延續[上一篇](https://github.com/Evabc/DataMining_MachineLearning/tree/master/1_Classification/1.3_%E6%A2%AF%E5%BA%A6%E6%8F%90%E5%8D%87%E6%B1%BA%E7%AD%96%E6%A8%B9_GBDT) GBDT的精神，XGBoost (eXtreme Gradient Boosting)最主要的改變有：
* 改變目標函數
在原本的loss function基礎上，加了正則化項(regularization)
原GBDT目標函數：

<img src="http://chart.googleapis.com/chart?cht=tx&chl= Obj=\sum_il(\hat{y_i},y_i)" style="border:none;">

XGBoost目標函數：

<img src="http://chart.googleapis.com/chart?cht=tx&chl= Obj=\sum_il(\hat{y_i},y_i)+\sum_k\Omega(f_t)" style="border:none;">

其中的Ω(f_t)就是正則項(也有人叫他懲罰項，但聽起來有點派? 他不過建個樹而已...)

<img src="http://chart.googleapis.com/chart?cht=tx&chl= \Omega(f)= \gamma(J)+\frac{1}{2}\lambda\sum^J_{j=1}b_j^2" style="border:none;">

J為葉子結點的數量, b_j為各個葉子結點上的值, 加號後面的部份即是對b進行L2正則化。整個項限制了樹的複雜度，目的是為了防止overfitting。

剩下的推導公式可以閱讀[參考資料一](https://zhuanlan.zhihu.com/p/46683728)

##### 1.4.2 如何建樹?
如何建樹這個問題，其實也可以說我們如何找一個劃分點來切。(對吧!)
還記得GBDT怎麼切的嗎? GBDT他用的decisiob tree就是CART，CART用在迴歸問題上，是透過**最小平方差**來找到一個好的切分點。而XGBoost呢? 提供兩種算法讓你找到適合的劃分點：
* 貪心算法：遍歷一個葉子結點上的所有屬性和值, 分別計算Gain, 選擇Gain大的屬性再進行分裂。不過這邊的Gain算法跟我們之前decisiob tree的Gain算法不一樣。
* 近似算法：當data量非常大或是在分怖式的環境下等，我們沒辦法使用貪心算法時，近似算法是另一個可行的方法。簡單來說就是對連續值進行離散化，假設一個屬性K，找到L個分位點的候選集合S_k，然後根據集合 S_k中的分位點將相應樣本劃分到桶(bucket)中。當我們需要遍歷這個屬性時，只需遍歷各個分位點就好e.g.我們有10個動漫人物的身高，分別為171~180各一個，先取173及178為分位點，把171-175都分到173這組(bucket), 176-180都分到178這個bucket。這樣當我們要取這十個人的平均身高時，我們不需要把171-180都點一次，就只抓173跟178算平均就好。

##### 1.4.3 系統優化
* Column Block for Parallel Learning
在建樹的過程中，最耗時間的就是找最好的切分點。而這個找的過程中，最耗時的部分是將數據排序。為了減少排序的時間，XGBoost 將數據存在內存單元 block 中，同時在block 採用 CSC 格式存放。(compressed column format)，一次性讀入數據並排好序後，以後在做樹的分裂時就可以重覆使用。
*  Cache-aware Access
在貪心算法中，使用緩存預取（cache-aware prefetching）將下一塊要讀的數據預先放進記憶體裡面。
在近似算法中，對Block的大小進行適當的設置，定義Block的大小為Block中最多的樣本數。
* Blocks for Out-of-core Computation
包含 兩個策略：
(1) block compression ：按列壓縮
(2) block sharding：將數據劃分到不同disk上，每個disk分配一個pre-fetcher，並將數據提取到內存緩衝區中，有助於平行運算時降低資料從 disk 讀取的時間。

以上三個系統優化也是XGBoost加速的主因

##### 1.4.4 XGBoost與GBDT的不同
* 算法優化：GBDT 在優化時只用到一階導數信息，XGBoost 則對目標函數進行了二階泰勒展開，同時用到了一階和二階導數。
* loss function：XGBoost 在損失函數中加入了正則化項，且支持自定義損失函數(只要函數可一階和二階求導)。
* 樣本的取用：GBDT在每輪使用全部的數據，XGBoost則採用了與隨機森林相似的方式，可以對數據進行採樣。
* 缺失值的處理：XGBoost 因為多了對稀疏數據的支持，在計算Gain時不會考慮帶有缺失值的樣本，在分裂點確定了之後，將帶有缺失值的樣本分別放在左子樹和右子樹，比較兩者分裂Gain，選擇Gain較大的那一邊作為默認分裂方向。
* 並行化處理：因為還是有Boosting 本身的詛咒，無法像隨機森林那樣樹與樹之間的並行化。但他在系統上優化了這個部份，提升了訓練速度。


參考資料1：https://zhuanlan.zhihu.com/p/46683728

參考資料2：https://zhuanlan.zhihu.com/p/69381524

