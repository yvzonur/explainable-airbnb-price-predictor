import pandas as pd
import ast
from collections import Counter

df = pd.read_csv('listings.csv', low_memory=False)
all_amenities = []
for index, row in df.iterrows():
    am = row.get('amenities', '[]')
    if pd.notnull(am) and isinstance(am, str) and am.startswith('['):
        try:
            am_list = ast.literal_eval(am)
            for a in am_list:
                all_amenities.append(a.lower().strip())
        except:
            pass

counts = Counter(all_amenities)
for k, v in counts.most_common(25):
    print(k, v)
