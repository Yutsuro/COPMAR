import os
import json

import numpy as np
import pandas as pd


# 各種フレーム幅に合わせたテストデータの用意
def MakeTestData(df,input_dim,num_frames_list):

     num_frames_list = sorted(num_frames_list,reverse=True)

     max_frame = max(num_frames_list)

     # データとラベルに分割
     df_x = df.drop(['Activity'],axis=1)
     df_t = df['Activity']

     # 格納用の空の配列の用意
     df_x_seq = [np.zeros((len(df),nf,input_dim),dtype=float) for nf in num_frames_list]
     df_t_seq = np.full((len(num_frames_list),len(df),1),255,dtype=int)

     check_i = 0

     for i in range(max_frame-1,len(df)):
         if len(list(set([df_t.iloc[j] for j in range(i-max_frame+1,i+1)]))):
             for ni,nf in enumerate(num_frames_list):
                 X = df_x.iloc[i-nf+1:i+1]
                 t = df_t.iloc[i]

                 df_x_seq[ni][check_i] = X
                 df_t_seq[ni,check_i] = t

             check_i = check_i + 1

     if check_i!=len(df):
         df_x_seq_list = [df_x_seq[i][:check_i] for i in range(len(num_frames_list))]
         df_t_seq_list = df_t_seq[:,:check_i]

     json_dir = "./data_json"

     if not os.path.exists(json_dir):
         os.mkdir(json_dir)

     for i,nf in enumerate(num_frames_list):
         print(np.shape(df_x_seq_list[i]),np.shape(df_t_seq_list[i]))

         df_dict = {"data":df_x_seq_list[i].tolist(),
                    "label":df_t_seq_list[i].tolist()}

         with open(json_dir+f"/testdata_frame{nf}.json","w",encoding="utf8") as f:
             json.dump(df_dict, f, indent=2)
     
     return df_x_seq_list,df_t_seq_list


# jsonに書き込めているかの確認
def Check_json_Data(json_dir,num_frames_list):

     for nf in num_frames_list:
         with open(json_dir+f"/testdata_frame{nf}.json",encoding="utf8") as f:
             df = json.load(f)
         print(np.shape(df["data"]),np.shape(df["label"]))


def CalculatePredictions():

     return


def main():
     
     input_dim = 25
     num_frames_list = [5,10,20,30]
     
     df = pd.read_csv("data/test.csv",index_col=0)

     MakeTestData(df,input_dim,num_frames_list)
     
     # 書き込みテスト
     json_dir = "./data_json"
     Check_json_Data(json_dir,num_frames_list)


if __name__=="__main__":
     main()