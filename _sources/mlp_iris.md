## MLPによるiris datasetのクラス分類プログラムのCLIアプリ化

### mlp_iris.pyの実装
mlp_iris.pyのサンプルプログラムを以下に示します．

<script src="https://gist.github.com/rsimd/ba9c259fee9f0490bb09bf1d6c72e0bb.js"></script>

### 実行結果

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
(datasci) mriki@client132 prml % python script/mlp_iris.py --seed=2023
Namespace(learning_rate=0.01, batch_size=16, seed=2023, max_epochs=100, device='cpu', test_size=0.3, validation_step=1)
教師データのACC: 0.957
テストデータのACC: 0.978
(datasci) mriki@client132 prml % python script/mlp_iris.py --seed=2525
Namespace(learning_rate=0.01, batch_size=16, seed=2525, max_epochs=100, device='cpu', test_size=0.3, validation_step=1)
教師データのACC: 0.947
テストデータのACC: 0.911
```