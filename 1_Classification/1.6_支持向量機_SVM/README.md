SVM算是機械學習中一個非常經典的算法，其實他的概念非常好懂，只是中間有些技巧會讓人數學不好的人一陣矇阿！ 不過，自從我服用[杰哥數位教室](https://www.youtube.com/watch?v=zG9Ux1YS3xs&feature=youtu.be)後，整個打通SVM的任督二脈！以下其實算是看完影片的總結心得，還是推薦大家把整個SVM系列看完，包準你SVM頂瓜瓜！


##### 1.6.1什麼是SVM
SVM是一種監督式學習，大多用來做分類及預測的問題。這邊我們都還是以二元分割來理解，當然也是有多類別分類的作法，但我們先PASS它！
最直覺的想法是，我們要找一條分割線，宛如摩西分海般，把一群資料做出一個好的劃分，但 SVM比摩西更威一點，當海分不開的時候，它還可以把世界投射到超高維來做！

我們用個圖來看，其實下圖每條線都可以把這兩堆資料切開，可是都花時間找了，就找好用一點的阿！
這邊的線，稱為**超平面(Hyperplane)**，線如果到了高維就成了平面，所以統一叫他們超平面。
中間的間隔，稱為**邊界(Margin)**，這個Margin我們希望越大越好，不會因為資料偏過去一點就誤判。
虛線上的點，稱為**支持向量(Support Vector)**
* Hard-Margin Support Vector Machine
是指模型在做資料分類時，不容許任何的誤差(如上面的例子)
不論是線性還是非線性的SVM，都可以找到它的Hard-Margin
* Soft-Margin Support Vector Machine
是指模型在做資料分類時，容許一定程度的誤差
這邊因為包含了容錯的概念，所以找的一定都是線性的SVM

SVM就是要將具有最大Margin的Hyperplane之線性方程式找出來！
出發吧！讓我們尋找~~~ONE PIECE~~~最大Margin的Hyperplane

##### 1.6.2 SVM如何尋找最大Margin的Hyperplane
SVM主要步驟如下：
1. 設計原始最佳化問題的限制條件
2. 設計原始問題的目標函數
3. 找出原始問題的KTT條件
    3.1寫出原始問題的Lagrange Multiplier Function
    3.2藉由KKT條件找出 W(法向量) 和 b(截距) 兩者與 ui(Lagrange乘子) 間的關係式
4. 把3.2的關係式帶回3.1的Lagrange Multiplier Function => 產生主問題的對偶問題
5. 得到最佳ui值 (ui*)
6. 藉由找出最佳的 ui 值，再找到最佳的法向量 W 和截距b 

**1.設計原始最佳化問題的限制條件**
現在我們知道，有最大Margin的Hyperplane是我們想要的夢幻逸品
那這樣的話，我們可以把Hyperplane以及其向兩旁延伸出去的兩個邊界定為：

公式一： <img src="http://chart.googleapis.com/chart?cht=tx&chl= W^TX_i+b\geq+1" style="border:none;"> if yi=+1

公式二： <img src="http://chart.googleapis.com/chart?cht=tx&chl= W^TX_i+b\leq-1" style="border:none;"> if yi=-1

最後把這兩個限制式整合成一個，就設計好我們的限制條件囉!

<img src="http://chart.googleapis.com/chart?cht=tx&chl= y_i(W^TX_i+b)\geq+1" style="border:none;"> ,i=1...N

**2.設計原始問題的目標函數**
最大Margin = 兩條限制式線的距離
經過整理後可以得到，式子左邊是距離最大的寫法，但比較常見的是右邊把它寫成最小的寫法：

<img src="http://chart.googleapis.com/chart?cht=tx&chl= f(W)=\frac{2}{||W||} = \frac{||W||^2}{2}" style="border:none;">

把SVM所要解決的原始的主要問題(Primal Problem)，寫成原始的最佳化問題模式如下：
Min. <img src="http://chart.googleapis.com/chart?cht=tx&chl=\frac{||W||^2}{2}" style="border:none;">

Subject to <img src="http://chart.googleapis.com/chart?cht=tx&chl= y_i(W^TX_i+b)\geq+1" style="border:none;"> ,i=1...N

**3.找出原始問題的KTT條件**
*3.1寫出原始問題的Lagrange Multiplier Function*

<img src="http://chart.googleapis.com/chart?cht=tx&chl= L(W,b,u_i)=\frac{||W||^2}{2}-\sum^Nu_i(y_i(W^TX_i+b)-1)" style="border:none;"> ,i=1...N
ui：拉格蘭吉乘數，為單一數值且ui≧0

根據KTT條件，可得5個條件式：
(a),(b)是根據拉格蘭吉乘式的變數做偏微分 (不包括ui)
(c)為原始問題的限制條件直接寫下來(也就是第一步找出來的條件)
(d)為(c)乘上ui後=0
(e)就是拉格蘭吉乘數ui要≧0

(a)<img src="http://chart.googleapis.com/chart?cht=tx&chl= \frac{\partial L(W,b,u_i)}{\partial W}   " style="border:none;"> => <img src="http://chart.googleapis.com/chart?cht=tx&chl= W=\sum^Nu_iy_iX_i " style="border:none;"> ,i=1...N

(b)<img src="http://chart.googleapis.com/chart?cht=tx&chl= \frac{\partial L(W,b,u_i)}{\partial b}   " style="border:none;"> => <img src="http://chart.googleapis.com/chart?cht=tx&chl=\sum^Nu_iy_i=0 " style="border:none;"> ,i=1...N

(c) <img src="http://chart.googleapis.com/chart?cht=tx&chl= y_i(W^TX_i+b)-1\geq" style="border:none;"> ,i=1...N

(d) <img src="http://chart.googleapis.com/chart?cht=tx&chl= u_i[y_i(W^TX_i+b)-1]=0" style="border:none;"> ,i=1...N

(e) <img src="http://chart.googleapis.com/chart?cht=tx&chl= u_i\geq 0" style="border:none;"> ,i=1...N

*3.2藉由KKT條件找出 W(法向量) 和 b(截距) 兩者與 ui(Lagrange乘子) 間的關係式*
由上述KKT條件中的(c)(d)兩條件可知兩種情況：
(c)>0 則 ui 必須等於 0  或 
(c)=0 則 ui 可大於等於 0 (為了與情況1的資料做區隔，故採用 ui 大於 0)
因此，可讓 ui 大於0的訓練資料即為**支持向量(Support Vector)**

**4. 把3.2的關係式帶回3.1的Lagrange Multiplier Function => 產生主問題的對偶問題**
把上述KKT條件中的(a)(b)兩條件帶回上面3.1的Lagrange Multiplier Function,
就可以得到原始問題的對偶問題,原本求最小值的變成求最大值,寫成如下：

Max.<img src="http://chart.googleapis.com/chart?cht=tx&chl= L(D)=\sum^Nu_i-\frac{1}{2}\sum^N_{i=1}\sum^N_{j=1} u_iu_jy_iy_jX^T_iX_j" style="border:none;">

Subject to <img src="http://chart.googleapis.com/chart?cht=tx&chl= u_i \geq 0" style="border:none;">, i=1...N
<img src="http://chart.googleapis.com/chart?cht=tx&chl= \sum^Nu_iy_i" style="border:none;">, i=1...N

**5.得到最佳ui值 (ui*)**
其中Xi與yi皆為訓練資料，帶進去即可求出最佳 ui值(ui*)

**藉由找出最佳的 ui 值，再找到最佳的法向量 W 和截距b **
求最佳的法向量 W 可以用KTT條件的(a),即可得W* (最佳的法向量 W)
然而，是否所有的點 Xi 均須帶入來找法向量W* ？ => 不用
因為我們要找到margin的邊界,只需要找到support vector,ui 大於0的Xi即可

最後求最佳的截距 b ,可以用KTT條件的(d) , 根據上面我們只看ui 大於0的
故將該式可以改寫成：
<img src="http://chart.googleapis.com/chart?cht=tx&chl= y_i(W^TX_i+b)=1" style="border:none;"> ,i=1...N
因為yi即是我們的分類結果(可能為1或-1), 故()內的部份也會跟著一樣是1或-1
可以再將式子改成下面後，即可得b*值
<img src="http://chart.googleapis.com/chart?cht=tx&chl= W^TX_i+b =y_i" style="border:none;"> ,i=1...N 

##### 1.6.3 Kernel Method (核方法)
上面講完一般的線性劃分法,但 如果今天是像下圖這樣,就沒辦法單純的用線性分開

Kernel Method：在原始空間 X (Original Space)不好區分的資料(無法線性區分)，可以用一個非線性的映射函數Φ，將這些資料轉換到另一個空間 H (特徵空間 Feature Space)，或許比較好區分 (可線性區分)。特徵空間不一定是更高維的空間，但通常愈高維，資料可以分得愈開

但在高維計算量會變的太大,所以我們就需要透過**引進Kernel Function 核函數**來幫忙解決
Kernel Function：特徵空間內兩點的內積，可以由原空間內對應兩點的內積，透過核函數來轉換求得。

透過公式推導,得知特徵空間中點跟點之間的內積、距離和角度,至於映射函數Ф就沒那麼重要。


參考資料1：http://debussy.im.nuu.edu.tw/sjchen/MachineLearning/final/CLS_SVM.pdf


