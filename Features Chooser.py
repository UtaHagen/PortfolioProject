import pandas as pd
import statsmodels.api as sm
from itertools import chain, combinations



y = df['y']
X = df.drop('y', axis=1)

def powerset(iterable):
        s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))


results = []
for combo in powerset(X.columns):
    combo = list(combo)
    X_subset = X[combo]
    
    X_subset = sm.add_constant(X_subset)
    model = sm.OLS(y, X_subset)
    results.append((combo, model.fit().rsquared))


results.sort(key=lambda x: x[1], reverse=True)
top_10 = results[:10]

for combo, rsquared in top_10:
    print(f"Combo: {combo}, R-squared: {rsquared}")