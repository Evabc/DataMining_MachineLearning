#### 1.3.1  什麼是Gradient Boosting?
在講GBDT，一定要先瞭解Gradient Boosting，要先瞭解Gradient Boosting，我們先複習一下(或點我看)：
* Boosting
Boosting是Ensemble learning中非常重要的概念，基本原理就是根據前一個弱分類器，再生成一個新的弱分類器，有序列性的產生一堆堆弱分類器們後，再把弱分類器們組合成一個強の分類器。
Boosting的重點只有兩個：
(1) 怎麼產生新的分類器? 
(2) 怎麼把弱分類器組起來變強?
* AdaBoost
從這兩個問題出發，分別給予不同的作法，就產生了大家世面上看到的一堆算法～
以我們前面講過的*AdaBoost*來看：
**怎麼產生新的分類器** :  通過更新每個樣本的權重產生不同的data set，再以此訓練一個新的弱分類器，並給予每個弱分類器一個權重。
**怎麼把弱分類器組起來變強**：將所有的弱分類器加權組合，再算出最後結果。
* AdaBoost
Gradient Boosting也是一種Boosting的方式，但它AdaBoost不一樣的地方在：
AdaBoost是通過改變訓練資料的權重來找新的分類器，而Gradient Boosting是用Gradient Descent來找一個新的分類器。
**怎麼產生新的分類器**  :  用Gradient Descent來找一個新的分類器，每個弱分類器的目標是擬合先前累加模型的loss function的負梯度，使加上該弱學習器後的累積模型損失往負梯度的方向減少。
**怎麼把弱分類器組起來變強**：把所有弱分類器權重相加，讓最後的模型可以損失最低。

這邊你一定覺得很怪 WHY! 為什麼相加到後來會損失最低!? 
請看李宏毅老師的：[ML Lecture 22: Ensemble](https://www.youtube.com/watch?v=tH9FH1DH5n0 "參考資料1")

這邊舉一個簡單的例子讓大家瞭解Gradient Boosting 的基本思想，順便讓大家知道殘差(residual)的概念。
但這並不代表所有的Gradient Boosting都是用殘差去計算的喔! 
假設銀魂裡神樂為14歲，第一個弱分類器擬合結果為11，則殘差為14-11=3，殘差=3 。
而這個「殘差=3 」它就是下一個弱分類器的學習目標，第二個弱分類器擬合結果為2，則這兩個弱分類器組合而成的Boosting 模型對於神樂的預測為11 + 2 = 13，以此類推可以繼續增加弱學習器以提高性能。

Gradient Boosting還有一個很重要的概念「**函數空間上的梯度下降**」
我們比較熟悉的Gradient Descent通常是值在參數空間上（e.g.訓練神經網絡，每輪迭代中計算當前損失關於參數的梯度，對參數進行更新）。而在Gradient Boosting中，每輪迭代生成一個弱分類器，這個弱分類器擬合loss function關於*之前累積*弱分類器的梯度，然後將新找的弱學習器加入累積function中，逐漸降低累積模型的損失。即參數空間的梯度下降利用梯度信息調整參數降低損失，函數空間的梯度下降利用梯度擬合一個新的函數降低損失。

現在，我們要找能夠彌補前一個弱分類器的新分類器，所以：
$$g_{_t}(x)= g_{_{t-1}}(x) + \alpha_tf_t(x)$$ (式1)

這個$$g_t(x)$$所有分類器加權總合

$$g_{t-1}(x)$$代表所有舊分類器加權總合

$$f_t(x)$$代表新找出來的分類器

$$\alpha_t$$代表新找出來的分類器的權重

那我們怎麼找一個好的新分類器，讓它能得到一個好的所有分類器加權總合(gt(X))?

這時候我們先幫gt(x)設個目標：

$$L(g)=\sum_{i=1}^nl(y_i,g(x_i))$$ (式2)

n= 所有的training data
y= 已知的label
g(x)= 分類器所預測的結果
l= loss function, 算y及g(x)的差異,要用那種方式可自行決定(cross entropy or mse...)

我們再回顧一下Gradient Descent的公式
Gradient Descent用來優化最小化損失函數L(θ), 進而求出對應的參數θ

$$\theta=\theta-\alpha \cdot \frac{\vartheta}{\vartheta\theta}L(\theta)$$ (式3)

再回到我們的gt(x)目標(式2), 我們就把g(x)當成參數，則同樣可以使用Gradient Descent法：

$$g_{_t}(x)= g_{_{t-1}} (x)-\alpha_t\cdot\frac{\vartheta}{\vartheta g_{t-1}(x)}L(y, g_{t-1}(x))$$ (式4)

因此可以得知第t輪弱分類器訓練的目標值是loss function 的負梯度，即

$$f_t(x) = -\frac{\vartheta L(y, g_{t-1}(x))}{\vartheta g_{t-1}(x)}$$

整理一下Gradient Boosting 算法流程：
*Algorithm* Gradient Boosting
1. 初始化
2. For t=1:T Do : 從1開始到T個弱分類器，分別做：
    2.1 Compute the negative gradient 計算負梯度
    2.2 Fit a weak learner which minimize 透過最小化平方誤差(擬合殘差)，產生新的弱分類器
    2.3 Update 更新函數
3. Return 

下面的圖分別寫了用了不同的loss function分別的負梯度為什麼。如果Gradient Boosting中採用平方損失函數(Squared error)，損失函數負梯度計算出來剛好是殘差(yi-f(x))，因此網路上有些寫Gradient Boosting是每一個弱學習器是在擬合之前累積模型的殘差。但這樣的說法感覺不太對，因為如果使用其他loss function或者在損失函數中加入regularization(正則項)，那麼負梯度就不會剛好是殘差了～

#### 1.3.2 GBDT : Gradient Boosting + Decision tree
在Gradient Boosting框架中，最常用的分類器就是是決策樹(一般是CART)了，加起來就是你常聽到的GBDT (Gradient Boosting Decision Tree, 梯度提升樹) 但還記得吧，CART包含了分類樹跟迴歸樹，他們的差異其實就是選擇的loss function不一樣啦～

* 迴歸問題
回歸問題其實比較好處理，就如我們所提到的用平方損失函數(Squared error)，損失函數負梯度計算出來剛好是殘差(yi-f(x))

* 分類問題
如果要把GBDT用在分類也行，如果用指數損失(exponential loss)的話，其實就是AdaBoost，所以才會有一方說法是:AdaBoost用在分類問題, GBDT用在迴歸問題)。如果還想看其他方法，可以看參考資料2跟參考資料3，他們兩個分別用了不同的方式做分類問題。

#### 1.3.2 GBDT 的優缺

正因為Gradient Boosting + Decision tree，因為樹都很簡單，所以有Decision tree的簡單快速，也不用做Normalization，但他因為他生為Boosting家族的一員，終身都要背著「訓練是按順序的」這個罪名，使它在大規模數據上可能導致速度慢。
不過既然存在著這樣罪惡，就一定要有新的英雄出現! XGBoost和LightGBM之後再講他們做了什麼改變。

參考資料1：https://www.youtube.com/watch?v=tH9FH1DH5n0

參考資料2：https://zhuanlan.zhihu.com/p/38329631

參考資料3：https://zhuanlan.zhihu.com/p/64863699
