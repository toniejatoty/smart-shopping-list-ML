import pytorchmodel
import json
from datetime import datetime
data = json.load(open('example_input.json', 'r', encoding='utf-8'))

user_data = data['users_data']
for i in range(len(user_data)):
    user_data[i]['gender']=int(user_data[i]['gender']=='K')
    user_data[i]['timestamp'] = (datetime.now() - datetime.strptime(user_data[i]['timestamp'], "%Y-%m-%dT%H:%M:%S.%f")).days 
print(user_data)