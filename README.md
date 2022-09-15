# COPMAR

## **C**alculate **O**ne **P**rediction from **MA**ny **R**esults

指定したデータフレームから基準に合わせた時系列データを切り出してjsonに保存する処理と、複数の確信度出力結果から最も確信度が高いクラスを算定するライブラリ（？）です。

## How to use

作業しているディレクトリにCOPMARをクローンして、`import COPMAR`で宣言

### COPMAR.COPMAR.MakeTestData

指定したテストデータから、`num_frames_list`に指定したフレーム幅の時系列データを作る関数。不連続データを弾く基準は`num_frames_list`の最大値に依存。作られたデータはjsonファイルに格納される。

```python
import COPMAR
import pandas as pd

df = pd.read_csv(csv_path)
input_dim = 25
num_frames_list = [1,5,10,20]

COPMAR.COPMAR.MakeTestData(df, input_dim, num_frames_list)
```

### COPMAR.COPMAR.CalculatePredictions

確信度を記録した複数のcsvファイルから、確信度が最大のクラスを算出する関数。クラス数およびデータ数は各ファイルで同じでないと失敗するので注意。

想定するCSVファイルの構造は以下の通り：  
```csv
class pred_0 pred_1 pred_2 pred_3 pred_4 pred_5
    0   0.25   0.20   0.15   0.10   0.10   0.20
    4   0.20   0.20   0.20   0.10   0.25   0.05
    1   0.25   0.40   0.05   0.00   0.10   0.20
                    .
                    .
                    .
```

実行例は以下の通り。この例では10000データ分の推論結果に対して、4つの推論結果を記録したCSVファイルから6クラス分類の確信度を算出した結果を受け取っている。

```python
import COPMAR
import pandas as pd

data_length = 10000
num_class = 6
test_pred_list = ["pred_A.csv","pred_B.csv","pred_C.csv","pred_D.csv"]

pred_data = COPMAR.COPMAR.CalculatePredictions(data_length,num_class,test_pred_list)
```