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

## cbow.pyのサンプルコード

### 実装

cbow.pyのサンプルプログラムを以下に示します．argparseパッケージを使ってCLI（Command Line Interface）アプリ化しています．

::::{admonition} ソースコード
:class: dropdown

```{code-block} python
:caption: cbow.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Any, Union, Callable, Type, TypeVar
from tqdm.std import trange, tqdm
import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import re
import argparse
import os
from sklearn.feature_extraction.text import CountVectorizer

def parse_args():
    parser = argparse.ArgumentParser(description="CBOWの訓練を行う")
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--embedding_dim", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--data_path", type=str,
                        default="./data/ja.text8", help="訓練用コーパスの保存場所")
    parser.add_argument("--save_path", type=str, help="学習済みモデルの保存場所")
    parser.add_argument("--window_size", type=int, default=11)
    parser.add_argument("--topn", type=int, default=5, help="検索単語数")
    parser.add_argument("--query", type=str, help="文字列を渡すと類似する単語をtopn個検索する")
    return parser


class CBoW(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        super().__init__()
        self.embeddingbag = nn.EmbeddingBag(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs: Any) -> Any:
        h = self.embeddingbag(inputs) / inputs.size(-1)
        return self.linear(h)


def build_dictionary(texts: list[str], min_df: int = 1) -> Any:
    """学習に使うコーパスから辞書とBoWを作る

    Args:
        texts (list[str]): コーパス，文書ごとに文字列として分けられたものがリストに入っている．
        min_df (int, optional): 最低出現文書数. Defaults to 1.

    Returns:
        Any: id2word, word2id, BoW(CSR_matrix)
    """
    def my_analyzer(text):
        tokens = text.split()
        tokens = filter(lambda token: re.search(
            r'[ぁ-ん]+|[ァ-ヴー]+|[一-龠]+', token), tokens)
        return tokens
    countvectorizer = CountVectorizer(min_df=min_df, analyzer=my_analyzer)

    X = countvectorizer.fit_transform(texts)
    id2word = {id: w for id, w in enumerate(
        countvectorizer.get_feature_names_out())}
    word2id = {w: id for id, w in id2word.items()}
    return id2word, word2id, X


def build_contexts_and_target(preprocessed_texts: list[list[str]], window_size: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """コンテキストとターゲットの配列を返す
    Args:
        preprocessed_texts (list[list[str]]): 単語ごとに半角スペースで区切られた前処理済みのコーパス
        window_size (int, optional): . Defaults to 5): ウィンドウサイズ

    Returns:
        tuple[np.ndarray, np.ndarray]: コンテキストとターゲット
    """
    contexts = []
    target = []
    a = window_size//2
    for text in preprocessed_texts:
        for i in range(a, len(text)-a):
            target.append(text[i])
            tmp = text[i-a:i]
            tmp += text[i+1:i+1+a]
            contexts.append(tmp)
    return np.array(contexts), np.array(target)


def get_batch(contexts: np.ndarray, target: np.ndarray, batch_size: int = 32, shuffle: bool = True):
    """バッチジェネレータ

    Args:
        contexts (np.ndarray): コンテキスト行列
        target (np.ndarray): ターゲットベクトル
        batch_size (int, optional): バッチサイズ. Defaults to 32.
        shuffle (bool, optional): シャッフルするかを決めるフラグ. Defaults to True.

    Yields:
        tuple(torch.Tensor, torch.Tensor): ミニバッチ
    """
    D = target.size
    index = np.arange(D)

    if shuffle:
        np.random.shuffle(index)

    n_batches = D // batch_size
    for minibatch_indexes in np.array_split(index, n_batches):
        a = torch.tensor(contexts[minibatch_indexes])
        b = torch.tensor(target[minibatch_indexes])
        yield a, b


def get_similar_words(query, word2id, word_embeddings, topn=5):
    """単語埋め込みベクトルを使って似た単語を検索する

    Args:
        query (str): 類似単語を検索したい単語
        word2id (dict[str,int], optional): 単語→単語idの辞書
        word_embeddings (np.ndarray, optional): 単語埋め込み行列．必ず(語彙数x埋め込み次元数)の行列であること.
        topn (int, optional): 検索結果の表示個数. Defaults to 5.
    """
    id = word2id[query]
    E = (word_embeddings.T / np.linalg.norm(word_embeddings,
         ord=2, axis=1)).T  # {(V,L).T / (V)}.T = (V,L)
    target_vector = E[id]
    cossim = E @ target_vector  # (V,L)@(L)=(V)
    sorted_index = np.argsort(cossim)[::-1][1:topn+1]  # 最も似たベクトルは自分自身なので先頭を除外

    print(f">>> {query}")
    _id2word = list(word2id.keys())
    for rank, i in enumerate(sorted_index):
        print(f"{rank+1}:{_id2word[i]} \t{cossim[i]}")


if __name__ == "__main__":
    args = parse_args()
    if args.save_path is None:
        save_path = f"./models/cbow_ep{args.max_epochs}_lr{args.learning_rate}_b{args.batch_size}_emb{args.embedding_dim}.pth"
    else:
        save_path = args.save_path

    with open(args.data_path) as f:
        text8 = f.read()
    texts = text8.split("。")
    id2word, word2id, X = build_dictionary(texts, 5)
    V = len(id2word)
    D = len(texts)
    print(f"文書数: {D}, 語彙数: {V}")

    preprocessed_texts = [[word2id[w]
                           for w in text.split() if w in word2id] for text in texts]
    preprocessed_texts = [
        text for text in preprocessed_texts if len(text) > args.window_size]
    contexts, target = build_contexts_and_target(
        preprocessed_texts, args.window_size)
    print("前処理後の文書数:", len(preprocessed_texts))
    print(f"contextsの数: {len(contexts)}")

    n_batches = len(target) // args.batch_size
    cbow = CBoW(V, args.embedding_dim)
    criterion = nn.CrossEntropyLoss()

    if not os.path.isfile(save_path):
        # 学習済みの重みがない場合
        print("training CBoW from scrach...")
        cbow.train()
        cbow = cbow.to(args.device)
        optimizer = optim.Adam(cbow.parameters(), lr=args.learning_rate)
        monitoring_loss = []

        for epoch in trange(args.max_epochs):
            with tqdm(total=n_batches) as tbar:
                for batch in get_batch(contexts, target, args.batch_size):
                    x, y = batch
                    x, y = x.to(args.device), y.to(args.device)

                    optimizer.zero_grad()
                    logits = cbow(x)
                    loss = criterion(logits, y)
                    loss.backward()
                    optimizer.step()

                    monitoring_loss.append(float(loss))
                    tbar.update(1)
        cbow.eval()
        cbow = cbow.cpu()
        with open(save_path+".vocab", "w") as f:
            f.write("\n".join(list(word2id.keys())))

        torch.save(cbow.state_dict(), save_path)
        plt.title("cross entropy of training set")
        plt.xlabel("iteration")
        plt.ylabel("cross entropy loss")
        plt.plot(monitoring_loss)

    else:
        # 学習済みの重みがある場合
        print("loading pretrained weights...")
        with open(save_path+".vocab") as f:
            vocab = f.read()
            id2word = vocab.splitlines()
            word2id = {word: id for id, word in enumerate(id2word)}

        cbow.load_state_dict(torch.load(save_path))
        cbow.eval()

    if args.query is not None:
        word_embeddings = list(cbow.embeddingbag.parameters())[
            0].data.detach().cpu().numpy()
        get_similar_words(args.query, word2id, word_embeddings, args.topn)

```
::::

### 実行結果

#### Usage
argparserのdescriptionやhelpに説明を書き込んだので，`--help` オプションを使えば説明が表示されます．
```
(datasci) mriki@RikinoMac prml % python script/cbow.py -h
usage: cbow.py [-h] [--learning_rate LEARNING_RATE] [--batch_size BATCH_SIZE] [--embedding_dim EMBEDDING_DIM]
               [--seed SEED] [--max_epochs MAX_EPOCHS] [--device DEVICE] [--data_path DATA_PATH]
               [--save_path SAVE_PATH] [--window_size WINDOW_SIZE] [--topn TOPN] [--query QUERY]

CBOWの訓練を行う

options:
  -h, --help            show this help message and exit
  --learning_rate LEARNING_RATE
  --batch_size BATCH_SIZE
  --embedding_dim EMBEDDING_DIM
  --seed SEED
  --max_epochs MAX_EPOCHS
  --device DEVICE
  --data_path DATA_PATH
                        訓練用コーパスの保存場所
  --save_path SAVE_PATH
                        学習済みモデルの保存場所
  --window_size WINDOW_SIZE
  --topn TOPN           検索単語数
  --query QUERY         文字列を渡すと類似する単語をtopn個検索する
```

自分自身しか使わないプログラムであっても，数週間も経てば使い方を忘れてしまうかもしれません．未来の自分のためにも，これらやスクリプト中のコメントを含めたドキュメントを必ず書くようにしましょう．

#### 実行
端末エミュレータから実行できる`ls`や`emacs`のようなコマンドと同様に，ここで作成したアプリも端末エミュレータからコマンドとして実行できます．
```
(datasci) mriki@RikinoMac prml % python script/cbow.py --save_path=models/CBoW_ep10_lr0.01_b512_emb50.pth --query=鳥
文書数: 564194, 語彙数: 63269
前処理後の文書数: 454833
contextsの数: 8109771
loading pretrained weights...
>>> 鳥
1:自室  0.6886861324310303
2:ラストシーン  0.685504674911499
3:世に  0.6764041185379028
4:不適切        0.6689269542694092
5:大洲  0.6603158712387085
```
(1epochも終わってないモデルの結果なので似た単語は全然出ていませんが，ちゃんと学習したモデルならもう少しまともな結果になるはずです...)