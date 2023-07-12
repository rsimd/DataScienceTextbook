## perceptron.py

先ほどのノートブックのコードから学習と簡単な可視化だけ取り出したPython scriptを以下に示します．（このプログラムをperceptron.pyとして保存してください．）


```python
# packageのimport
from typing import Any, Union, Callable, Type, TypeVar
import numpy as np 
import numpy.typing as npt
import pandas as pd 
import matplotlib.pyplot as plt 
import japanize_matplotlib
import argparse
plt.style.use("bmh")


def parse_args():
    parser = argparse.ArgumentParser(description="Perceptronで論理ゲートを再現する")
    parser.add_argument("--gate_type", type=str, default="or")
    parser.add_argument('--learning_rate', type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type=str, default="./perceptron.png")
    return parser.parse_args()


def sign(x:float)->float:
    if x > 0:
        return 1.
    return -1

def perceptron(
        X:list[float,float], 
        W:list[float,float],
        b:float)->np.ndarray:
    h = 0.
    for _x, _w in zip(X,W):
        h += _x * _w
    h += b
    return sign(h)

def plot_perceptron(w,b, truth_table, gate_type="or", need_output=False):
    linear = lambda x1,w,b: -(w[0]*x1 + b)/w[1]
    x1_sample = np.linspace(-2,2,100)

    fig, ax = plt.subplots()
    ax.scatter(truth_table.x1, truth_table.x2, c=truth_table[gate_type], label="入力データ")
    ax.plot(x1_sample, linear(x1_sample, W,b), label="閾値, 識別境界")
    ax.set_title(f"$x_1$-$x_2$平面上の{gate_type}問題")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    plt.legend()
    #ax.set_aspect('equal')
    if need_output:
        return fig
    
def update_perceptron(X,Y, lr=1.0, rng = np.random.default_rng(10)):
    W = rng.normal(0,1, [X.shape[-1]])
    b = float(rng.normal(0,1, 1))
    diff = np.inf
    while diff>0:
        diff = 0
        for (x,y) in zip(X,Y):
            y_hat = perceptron(x, W, b)
            W += lr*(y-y_hat)*x
            b += lr*(y-y_hat)
            diff += y != y_hat
    return W, b

if __name__ == "__main__":
    # 真偽値表の定義
    args = parse_args()
    truth_table = pd.DataFrame(
        np.array([[1,1,0,0],[1,0,1,0]]).T,
        columns=["x1","x2"]
    )
    truth_table["or"] = truth_table.x1 | truth_table.x2
    truth_table["and"] = truth_table.x1 & truth_table.x2
    truth_table["xor"] = np.logical_xor(truth_table.x1,truth_table.x2).astype(truth_table.x1.dtype)
    truth_table["nor"] = [0,0,0,1]

    truth_table[truth_table == 0] = -1

    W,b = update_perceptron(
        truth_table[["x1","x2"]].to_numpy(),
        truth_table[args.gate_type].to_numpy(),
        rng=np.random.default_rng(args.seed),
        lr=args.learning_rate
        )
    fig = plot_perceptron(W,b,truth_table=truth_table, gate_type=args.gate_type,need_output=True)

    Y_hat = []
    for x in truth_table[["x1","x2"]].to_numpy():
        y_hat = perceptron(x, W,b)
        Y_hat.append(y_hat)
    print("正解:",truth_table[args.gate_type].to_numpy(),)
    print("予測:",Y_hat)
    fig.savefig(args.save_path)
```


perceptron.pyは例えば以下のようにして，端末エミュレータから実行することができます．
```sh
python perceptron.py --gate_type=nor
```