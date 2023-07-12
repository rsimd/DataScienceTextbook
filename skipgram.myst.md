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

## [発展課題]skip_gram.py

Skip-Gramを実装し，max_epochs=100, minibatch_size=512として訓練し，「サッカー」，「日本」，「女王」，「機械学習」について類似単語を類似度の高い順に上位5個表示するプログラムを作成してください．

- cbow.pyを参考にしてください．
- 学習にはja.text8を利用してください．

雛形:  
```python
class SkipGram(nn.Module):
    def __init__(self):
        super().__init__()
        ...

    def forward(self,x):
        ...
```

### 実装
::::{admonition} skipgram.py
:class: dropdown

<script src="https://gist.github.com/rsimd/ba9c259fee9f0490bb09bf1d6c72e0bb.js"></script>

::::

### 実行結果

#### Usage

argparserのdescriptionやhelpに説明を書き込んで，`--help` オプションで使い方が表示できるようにしてください．
```
(datasci) mriki@RikinoMac _prml % python skipgram.py -h
usage: skipgram.py [-h] [--learning_rate LEARNING_RATE] [--batch_size BATCH_SIZE] [--embedding_dim EMBEDDING_DIM] [--seed SEED] [--max_epochs MAX_EPOCHS] [--char_limit CHAR_LIMIT] [--device DEVICE] [--data_path DATA_PATH] [--save_path SAVE_PATH] [--window_size WINDOW_SIZE] [--query QUERY] [--topn TOPN]

Skip-Gramの訓練をja.text8で行う

options:
  -h, --help            show this help message and exit
  --learning_rate LEARNING_RATE
  --batch_size BATCH_SIZE
  --embedding_dim EMBEDDING_DIM
  --seed SEED
  --max_epochs MAX_EPOCHS
  --char_limit CHAR_LIMIT
                        ja.text8の先頭から何文字を利用するか．Noneの場合は全てを使う． ex. 1_000_000
  --device DEVICE
  --data_path DATA_PATH
                        訓練用コーパスの保存場所
  --save_path SAVE_PATH
                        学習済みモデルのファイル名．すでに存在していた場合はそれを読み込んで利用する
  --window_size WINDOW_SIZE
  --query QUERY         文字列を渡すと類似する単語をtopn個検索する
  --topn TOPN           検索単語数
```



#### 実行

初回学習時：
```sh
(datasci) mriki@RikinoMac _prml % python skipgram.py --char_limit 1000000 --seed 7012 --save_path ./skipgram.pkl --max_epochs 2
全文書の文字数が46507793あり，その内1000000だけを利用します．
前処理...
363003it [00:16, 21819.80it/s]
100%|███████████████████████| 80264/80264 [00:00<00:00, 900528.08it/s]
contextsのshape: (80264, 17871)
訓練開始...
  epoch    train_loss    train_ppl    valid_loss    valid_ppl      dur
-------  ------------  -----------  ------------  -----------  -------
      1        9.8043   18111.2683       10.1251   24961.8040  14.5564
      2        8.4743    4789.8352       10.6640   42786.7327  14.5319
```

学習済みの場合：
```sh
(datasci) mriki@RikinoMac _prml % python skipgram.py --char_limit 1000000 --seed 7012 --save_path ./skipgram.pkl --max_epochs 2
./skipgram.pklから学習済みモデルを読み込みます...
```

学習済みでクエリを検索する場合：
```sh
(datasci) mriki@RikinoMac _prml % python skipgram.py --save_path ./skipgram.pkl --query 日本
./skipgram.pklから学習済みモデルを読み込みます...
>>> 日本
1:古代  0.9472917318344116
2:文明  0.9328379034996033
3:社会  0.931919515132904
4:文化  0.9224883317947388
5:天皇  0.9139895439147949
(datasci) mriki@RikinoMac _prml % python skipgram.py --save_path ./skipgram.pkl --query ロボット
./skipgram.pklから学習済みモデルを読み込みます...
>>> ロボット
1:ロボティックス        0.8361095190048218
2:ステーション  0.8090811967849731
3:ぼう  0.8085721135139465
4:ロケット      0.7877843976020813
5:地球  0.7545166611671448
```