import pandas as pd
import numpy as np
import json

df = pd.read_csv('./FreqBand')
df = df.set_index('freqband')
df.index = df.index.astype(str)
beamDict = df['beam'].astype(float).to_dict()
df['T'] = df['mapdepth t']
df['P'] = df['T'] * np.sqrt(2)
noiseDict = df[['T', 'P']].to_dict('index')

with open('beamsize.json', 'w') as f:
    json.dump(beamDict, f)
with open('noise.json', 'w') as f:
    json.dump(noiseDict, f)
