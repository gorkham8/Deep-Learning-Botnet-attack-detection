import pandas as pd
import numpy as np

dataset =pd.read_csv('n_BaIoT_0.csv')
df_all=dataset.sample(frac=0.5)
del dataset
print(0)

dataset =pd.read_csv('n_BaIoT_1.csv')
df=dataset.sample(frac=0.05)
del dataset
df_all = pd.concat([df_all,df])
del df
print(1)

dataset =pd.read_csv('n_BaIoT_2.csv')
df=dataset.sample(frac=0.05)
del dataset
df_all = pd.concat([df_all,df])
del df
print(2)

dataset =pd.read_csv('n_BaIoT_3.csv')
df=dataset.sample(frac=0.05)
del dataset
df_all = pd.concat([df_all,df])
del df
print(3)

dataset =pd.read_csv('n_BaIoT_4.csv')
df=dataset.sample(frac=0.05)
del dataset
df_all = pd.concat([df_all,df])
del df
print(4)

dataset =pd.read_csv('n_BaIoT_5.csv')
df=dataset.sample(frac=0.05)
del dataset
df_all = pd.concat([df_all,df])
del df
print(5)

dataset =pd.read_csv('n_BaIoT_6.csv')
df=dataset.sample(frac=0.05)
del dataset
df_all = pd.concat([df_all,df])
del df
print(6)

dataset =pd.read_csv('n_BaIoT_7.csv')
df=dataset.sample(frac=0.05)
del dataset
df_all = pd.concat([df_all,df])
del df
print(7)

dataset =pd.read_csv('n_BaIoT_8.csv')
df=dataset.sample(frac=0.05)
del dataset
df_all = pd.concat([df_all,df])
del df
print(8)

dataset =pd.read_csv('n_BaIoT_9.csv')
df=dataset.sample(frac=0.05)
del dataset
df_all = pd.concat([df_all,df])
del df
print(9)

dataset =pd.read_csv('n_BaIoT_10.csv')
df=dataset.sample(frac=0.05)
del dataset
df_all = pd.concat([df_all,df])
del df
print(10)


df_all.to_csv('n_BaIoT_concentrated3.csv',index=False)



