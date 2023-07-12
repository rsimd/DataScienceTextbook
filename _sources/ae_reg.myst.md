## 正則化項

ニューラルネットワークを含む機械学習において，昔から使われてきた過学習抑制手法として __正則化__ があります．get_lossメソッドの話が出てきたので，ここで触れておきます．

ニューラルネットワークにおいて，学習可能パラメータである結合重みは0周辺の絶対値の小さな数字からスタートして，徐々に更新されていきます．これが更新が進むごとに絶対値の大きな値になっていくことがあります．学習アルゴリズムがパラメータを好き勝手に更新してしまうと，本当は0周辺でタイトにまとまることができた結合重みは大きなノルムを持つようになります．これを，できるだけ小さいノルムで損失関数を小さくするように制限するのが正則化です．

![](https://di-acc2.com/wp-content/uploads/2021/08/DL81-1024x447.jpg)

正則化は，損失関数に追加する形で利用されます．L1正則化は学習可能パラメータのL1ノルム，L2正則化はL2ノルムを損失関数に追加します．このL1とL2で作用が異なりますが，どちらも学習可能パラメータをできるだけ小さいノルムに抑えたまま損失関数を小さくしなければいけないという制約を機械学習モデルに与えます．これによって学習アルゴリズムがパラメータに与えることができる値は0から離れるほどに罰則が掛かり，0に近いほど利用しやすい状態になります．これはモデルの複雑性を抑える効果があり，モデルにバイアスを付加してバリアンスを減らしているともいえます．


## VAE

```python
 def reparametrizaion(self, mean, log_var, device):
        epsilon = torch.randn(mean.shape, device=device)
        return mean + epsilon*torch.exp(0.5 * log_var)
```

$$
\begin{align} \mathcal{L}({\bf{x}}, \phi, \theta)&= \mathbb{E}_{q_\phi({\bf{z}}|{\bf{x}})}\left[\log p_\theta({\bf{x}}|{\bf{z}})\right] – D_{KL}\left( q_\phi({\bf{z}}|{\bf{x}})||p_\theta({\bf{z}}) \right) \end{align}
$$

$$
D_{KL}\left( q_\phi({\bf{z}}|{\bf{x}})||p_\theta({\bf{z}}) \right)=-\frac{1}{2}\sum_{j=1}^{D}\left(1+\log \sigma^2_j – \mu_j^2 -\sigma^2\right)
$$
```python
def kl_divergence(mean,log_var):
    """KL[q(z|x)||p(z)]を計算"""
    return 0.5 * torch.sum(1+log_var - mean**2 - torch.exp(log_var))

loglikelihood = torch.sum(x * torch.log(x_hat+1e-8) + (1 - x) * torch.log(1 - x_hat  + 1e-8)) #E[log p(x|z)]
kld = kl_divergence(mean, log_var)

lower_bound = -(loglikelihood + kld) #変分下界(ELBO)=E[log p(x|z)] - KL[q(z|x)||p(z)]
```

