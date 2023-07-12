---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: datasci
  language: python
  name: datasci
---


## [課題] ae_fmnist_animation.py

このノートではmnistデータセットの次元圧縮を行うAEを実装しました．これの主要なパラメータをCLIオプションで変更できる形にしたプログラムae_fmnist_animation.pyを作成してください．
- このスクリプトは最終的に，任意のファイル名でアニメーション①に相当するgifファイルを作成することが目的です．
- データセットとして[Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist)を使ってください．これも有名なデータセットなので，PyTorchでは[torchvision.datasets.FashionMNIST](https://pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html)を利用することができます．
- ae_mnist.pyを更に修正し，--model 引数でSimpleAEとWeightTyingAEを選択できるようにしてください．ただし，--modelはクラス名を受け取ります．

:::{hint}

これどうすればいいのかな？と思ったら，自分で良さそうな設定にしてOKです．ただし，「このスクリプトでは**これこれこのような設定**で実験を行う」とcliの説明文に書いてください．

:::

::::{admonition} ae_fmnist_animation.py
:class: dropdown

<script src="https://gist.github.com/rsimd/ba9c259fee9f0490bb09bf1d6c72e0bb.js"></script>

::::

### 実行結果

#### Usage

argparserのdescriptionやhelpに説明を書き込んで，`--help` オプションで使い方が表示できるようにしてください．
```
(datasci) mriki@RikinoMac prml % python script/ae_fmnist_animation.py -h

```



#### 実行

```
(datasci) mriki@RikinoMac prml % python script/ae_fmnist_animation.py ...
...

```
