import numpy as np
import pandas as pd

df = pd.read_csv('./30.csv')

arr_1 = df.loc[0:134, 'flux_idx'].to_numpy()
arr_2 = df.loc[:134, 'lon'].to_numpy()
print(df)


print(f'{arr_1.shape=}')

df_new = pd.DataFrame({
    'flux_idx': arr_1,
    'lon': arr_2
    })

df_new.to_csv('./test.csv', index=False)

