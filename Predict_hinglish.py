from requests import request
import requests
import json
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
NUM_THREADS = 10

lang="hg"
model="hinglish_check_1june_2"
# model="zero_hinglish_15_april"

OUTPUT_FILE_PATH='/Users/abhishek.mathada/Documents/Hinglish/train_files/hello_jio/hg_prediction_set_part2_slow_spell_check_false.csv'
# OUTPUT_FILE_PATH='/Users/abhishek.mathada/Documents/Hinglish/train_files/Zero/prediction_zero_slow_hinglish_dict_19april.csv'

data=pd.read_csv("/Users/abhishek.mathada/Documents/Hinglish/train_files/hello_jio/Hinglish_Validation_set_part2.csv")
# data=pd.read_csv("/Users/abhishek.mathada/Documents/Hinglish/train_files/Zero/validation_zero_hinglish.csv")

queries=data['Expression']
actual_label=data['Intent']

resp=[]

def prediction(query_label):
  query,label = query_label
  url = "https://eva-enterprise-replica.engageapps.jio/api/v5/LR_FT_PREDICT?lang="+lang+"&model="+model+"&algo=lstm-lr&q="+query+"&nnp_check=False&greetings_check=False&spell_check=false"
  payload={}
  headers = {
      'Authorization': 'Basic aGVsbG9qaW8yOlElbnQ9eEZ2SiN3KyViQzdqWGpZQVFjMldmc1lAanB3P004',
      'Content-Type': 'application/json'
  } 
  response = requests.request("GET", url, headers=headers, data=payload)
  response=response.json()
  resp.append([query, label,response['textLabel'],response['labelprobability']])


with ThreadPoolExecutor(NUM_THREADS) as executor:
    args_list = list(zip(queries,actual_label))
    _ = list(tqdm(executor.map(prediction,args_list), total = len(args_list)))

df = pd.DataFrame(resp, columns = ['query', 'actual_label', 'i_label', 'i_prob'])
df.to_csv(OUTPUT_FILE_PATH,index=False)
