---
jupyter:
  jupytext:
    formats: md:myst
    text_representation:
      extension: .md
      format_name: myst
      format_version: 0.13
      jupytext_version: 1.11.5
  kernelspec:
    display_name: python
    language: python
    name: python
---

## [課題]MLPによるiris datasetのクラス分類プログラムのCLIアプリ化

```sh
$ python mlp_iris.py
```
で実行できる形のファイルmlp_iris.pyを作成して提出せよ。

jupyterファイルの内容を修正して、教師データとテストデータに対する正答率を標準出力に表示するようにしよう。

最低限必要な引数リスト（イコールの後ろの値はデフォルト値）:

- learning_rate = 0.01
- batch_size = 16
- max_epochs = 100
- device ="cpu" ... torchを動かしたいデバイス ( cpuとかgpuとか)
- seed =42... 疑似乱数のSEED
- test_size=0.3 

最終的に標準出力に出してほしいもの：

```py
print("教師データのACC",: train_acc)
print("テストデータのACC",: test_acc)
```

```{admonition} 心がけてほしい事項
- ArgumentParserの説明文をしっかり書くこと．
- コードにコメントを追加して読みやすくすること．（分量はお任せします）
- 関数にdocstringsを追加して読みやすくすること．
- linter/formatterなどを使って，コードを綺麗にすること．
- 自分で実行してみる or デバッガを使ってエラーが出ないプログラムを完成させよう．
```

Visual Studio Codeなどのテキストエディタを使ってmlp_iris.pyを編集してください．vscodeを利用する場合はpython用の設定（拡張機能のインストール）をすることで，コードを綺麗に書くことができます．  
また，vscode以外だとJetBrainが出しているPython用IDE「PyCharm」がお勧めです．（学生なら無料で使えるはず）  

### 実装

mlp_iris.pyのサンプルプログラムを以下に示します．

:::{admonition} mlp_iris.py
:class: dropdown
<script src="https://gist.github.com/rsimd/feac13929f5b5ce432fa85fcbc3466f8.js"></script>
:::

### 実行結果

#### Usage

```sh
(datasci) mriki@client132 prml % python script/mlp_iris.py -h
usage: mlp_iris.py [-h] [--learning_rate LEARNING_RATE] [--batch_size BATCH_SIZE] [--seed SEED]
                   [--max_epochs MAX_EPOCHS] [--device DEVICE] [--test_size TEST_SIZE]
                   [--validation_step VALIDATION_STEP]

三層のMLPでiris datasetのクラス分類を行う

options:
  -h, --help            show this help message and exit
  --learning_rate LEARNING_RATE
  --batch_size BATCH_SIZE
  --seed SEED
  --max_epochs MAX_EPOCHS
  --device DEVICE
  --test_size TEST_SIZE
  --validation_step VALIDATION_STEP
                        バリデーションの実行ステップ

```

#### 実行
実行例1
```sh
(datasci) mriki@client132 prml % python script/mlp_iris.py --seed=2023
Namespace(learning_rate=0.01, batch_size=16, seed=2023, max_epochs=100, device='cpu', test_size=0.3, validation_step=1)
教師データのACC: 0.957
テストデータのACC: 0.978
```

実行例2
```sh
(datasci) mriki@client132 prml % python script/mlp_iris.py --seed=2525
Namespace(learning_rate=0.01, batch_size=16, seed=2525, max_epochs=100, device='cpu', test_size=0.3, validation_step=1)
教師データのACC: 0.947
テストデータのACC: 0.911
```
