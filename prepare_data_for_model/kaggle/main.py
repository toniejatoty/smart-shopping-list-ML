import pandas as pd
import sklearn
import numpy as np
import pandas as pd
import os
import pickle
merged_cache_file = 'merged_cache.pkl'
cache_file = 'data_cache.pkl'

if os.path.exists(cache_file):
    with open(cache_file, 'rb') as f:
        data = pickle.load(f)
else:
    aisles = pd.read_csv('archive/aisles.csv')
    departments = pd.read_csv('archive/departments.csv')
    prior = pd.read_csv('archive/order_products__prior.csv')
    train = pd.read_csv('archive/order_products__train.csv')
    orders = pd.read_csv('archive/orders.csv')
    products = pd.read_csv('archive/products.csv')

    data = {
        'aisles': aisles,
        'departments': departments,
        'prior': prior,
        'train': train,
        'orders': orders,
        'products': products
    }
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)

aisles = data['aisles']
departments = data['departments']
prior = data['prior']
train = data['train']
orders = data['orders']
products = data['products']

df = pd.concat([prior, train])

if os.path.exists(merged_cache_file):
    with open(merged_cache_file, 'rb') as f:
        df = pickle.load(f)
else:
    df = pd.concat([prior, train], ignore_index=True)

    df = df.merge(products, on='product_id')
    df = df.merge(aisles, on='aisle_id')
    df = df.merge(departments, on='department_id')

    with open(merged_cache_file, 'wb') as f:
        pickle.dump(df, f)

print(df.shape)

# koszyki = df.groupby('order_id').agg({
#  #   'product_name': lambda x: ', '.join(x),
#     'aisle': lambda x: ', '.join(x),
#     'department': lambda x: ', '.join(x)
# }).reset_index()

# koszyki = koszyki[['aisle', 'department']]
# koszyki.to_csv('koszyki.csv', index = False)

koszyki = pd.read_csv('koszyki.csv')

meannn = df.groupby(by='order_id').size().mean()
print(f"Średnia ilośc produktów w koszyku: {meannn}")
print(f"Liczba koszyków: {koszyki.shape[0]}")

