import pandas as pd
import numpy as np
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager

names = ['class','sha256']
for i in range(2,107):
    names.append(f'api_seq_call_{i}')

df = pd.read_csv('./malware_dataset/malware_API_dataset.csv',on_bad_lines='skip',names=names)

api_columns = df.columns[2:]  # API call columns
unique_apis = set()

for col in api_columns:
    unique_apis.update(df[col].dropna().astype(str).unique())

# Setup Selenium WebDriver
options = webdriver.ChromeOptions()
options.add_argument("--headless")  # Run without opening browser (optional)
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")

service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

# Microsoft API search base URL
base_url = "https://learn.microsoft.com/en-us/search/?terms="

api_dict = {}
for i,api in enumerate(unique_apis):
    search_url = base_url + api.replace(" ", "%20")
    
    # Open search page
    print(i,search_url)
    driver.get(search_url)
    time.sleep(2)  # Allow page to load
    try:
        # Extract first search result description
        first_result = p_elements = driver.find_elements(By.TAG_NAME, "p")
        text_ele = []
        for i in first_result:
            text_ele.append(i.text)
        api_dict[api] = text_ele
    except Exception:
        api_dict[api] = "Failed to fetch"

# Close WebDriver
driver.quit()

# Print API descriptions
print("\nAPI Dictionary:")
for api, desc in api_dict.items():
    print(f"{api}: {desc}")

import pickle

f = open("file.pkl","wb")
pickle.dump(api_dict,f)
f.close()
