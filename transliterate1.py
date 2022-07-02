import pandas as pd
from tqdm import tqdm
from indictrans import Transliterator

trn = Transliterator(source='eng', target='hin', build_lookup=True)


# hin="आल इंडिया me ंर्क लागु करे w कश्मीर से dhara 370ko ख़तम करे हम इंडियन को आपसे यही उम्मीद है"
# INPUT_FILE_PATH = '/Users/abhishek.mathada/Documents/Common/greetings_dev.csv'
# OUTPUT_FILE_PATH = '/Users/abhishek.mathada/Documents/Common/greetings_temp_dev.csv'

INPUT_FILE_PATH = '/Users/abhishek.mathada/Desktop/country.csv'
OUTPUT_FILE_PATH = '/Users/abhishek.mathada/Desktop/country_hi.csv'


# def hindi_to_english(x):
#     return trn.transform(x)

def english_to_hindi(x):
    return trn.transform(x)


data = pd.read_csv(INPUT_FILE_PATH)
queries = data['place']
# actual_labels = data['label']

resp = []
# for query, actual_label in tqdm(list(zip(queries, actual_labels))):
#     english_words = hindi_to_english(query)
#     resp.append([query, actual_label, english_words])

for query in tqdm(queries):
    hindi_names = english_to_hindi(query)
    resp.append([query, hindi_names])
    # resp.append(hindi_names)
# print(resp)

df = pd.DataFrame(resp, columns=['query', 'hindi_names'])
df.to_csv(OUTPUT_FILE_PATH, index=False)
