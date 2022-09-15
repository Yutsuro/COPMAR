import os
import json

import numpy as np
import pandas as pd


# 各種フレーム幅に合わせたテストデータを用意する関数
#
# 引数：
#    df: 時系列化したいデータ
#    input_dim: データの特徴量の次元数
#    num_frames_list: 推論に用いるフレーム数を格納したリスト
def MakeTestData(df,input_dim,num_frames_list):

     # フレーム数のリストを大きさ順に揃える（一番大きいものが基準となる）
     num_frames_list = sorted(num_frames_list,reverse=True)
     max_frame = max(num_frames_list)

     # データとラベルに分割
     df_x = df.drop(['Activity'],axis=1)
     df_t = df['Activity']

     # 格納用の空の配列の用意
     df_x_seq = [np.zeros((len(df),nf,input_dim),dtype=float) for nf in num_frames_list]
     df_t_seq = np.full((len(num_frames_list),len(df),1),255,dtype=int)

     # num_frames_listで指定したフレーム数ごとに時系列データを作成
     # check_iは指定フレーム連続するデータのみを判定するための値
     check_i = 0
     for i in range(max_frame-1,len(df)):
         
         # num_frames_listの最大フレーム連続するデータのみを時系列データとする（不連続データは使わない）
         if len(list(set([df_t.iloc[j] for j in range(i-max_frame+1,i+1)])))==1:

             for ni,nf in enumerate(num_frames_list):

                 # 指定フレーム遡ったデータを用意
                 X = df_x.iloc[i-nf+1:i+1]
                 t = df_t.iloc[i]

                 # 用意したデータを先の配列に格納
                 df_x_seq[ni][check_i] = X
                 df_t_seq[ni,check_i] = t

             # 連続判定用の値をインクリメント（不連続の場合は足されない）
             check_i = check_i + 1

     # 指定フレームの最大値が1でない場合は、リスト末尾の余分な値を切り落とす
     # この処理をすることで、あらかじめ用意した空の配列からデータ以外の部分を排除
     # appendよりも代入を使うのは速度重視
     if check_i!=len(df):
         df_x_seq_list = [df_x_seq[i][:check_i] for i in range(len(num_frames_list))]
         df_t_seq_list = df_t_seq[:,:check_i]

     # 作ったデータをjsonにして書き出すためのディレクトリの指定
     # フレームの最大値ごとにデータが違うので、出力先は分ける
     json_dir = f"./data_json{str(max_frame)}"
     if not os.path.exists(json_dir):
         os.mkdir(json_dir)

     # 作成したデータの書き出し
     for i,nf in enumerate(num_frames_list):
         print(np.shape(df_x_seq_list[i]),np.shape(df_t_seq_list[i]))

         # "data"にデータを、"label"にラベルを格納
         df_dict = {"data":df_x_seq_list[i].tolist(),
                    "label":df_t_seq_list[i].tolist()}

         with open(json_dir+f"/testdata_frame{nf}.json","w",encoding="utf8") as f:
             json.dump(df_dict, f, indent=2)
     
     # 一応データ/ラベルと、データ数を返す
     # 主目的はjsonへの保存なのでおまけ（毎回時系列データを生成していると時間がかかる）
     return df_x_seq_list,df_t_seq_list,check_i


# jsonに書き込めているかの確認（おまけ、確認用）
def Check_json_Data(json_dir,num_frames_list):

     for nf in num_frames_list:
         with open(json_dir+f"/testdata_frame{nf}.json",encoding="utf8") as f:
             df = json.load(f)
         print(np.shape(df["data"]),np.shape(df["label"]))


# 複数の確信度を保存したファイル(CSV)から、最終的なクラスを算出する関数
# 
# 引数：
#    data_length: 推論するデータ数
#    num_class: クラス数
#    test_pred_list: 推論に使うCSVファイルのパスのリスト
def CalculatePredictions(data_length,num_class,test_pred_list):

     # 格納用の空の配列の用意（appendではなく代入のため）
     pred_data = np.zeros((data_length,num_class*len(test_pred_list)),dtype='float')

     # 複数の確信度のデータフレームを（クラスの列を除いて）横にくっつける
     for i,pl in enumerate(test_pred_list):
         df_pred = pd.read_csv(pl).to_numpy()[1:num_class+1]
         pred_data[:,i*num_class:(i+1)*num_class] = df_pred

     # くっつけたデータフレームについて、最もその行内での確信度が高い列番号を
     # クラス数で割った"あまり"がそのまま確信度の大きいクラスとなる
     pred_label = np.argmax(pred_data,axis=1) % num_class

     # 算出したクラスラベルを返す
     return pred_label


# 以下、動作確認用
def main():
     
     input_dim = 25
     num_frames_list = [5,10,20,30]
     
     df = pd.read_csv("data/test.csv",index_col=0)

     _,_,data_length = MakeTestData(df,input_dim,num_frames_list)

     print(data_length)
     
     # 書き込みテスト
     json_dir = "./data_json"
     Check_json_Data(json_dir,num_frames_list)


if __name__=="__main__":
     main()