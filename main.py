import streamlit as st
import numpy as np

# for getting random links
import random
from serpapi import GoogleSearch
from openai import OpenAI

import embedding_bucketing.embedding_model_test as em

from config import openai_key
from config import SER_API_KEY

import ao_core as ao
from Arch__giftRecommender import arch

import json
import re
with open("google-countries.json") as f:
    data = json.load(f)


country_names = []
for country in data:
    country_names.append(country["country_name"])



client = OpenAI(api_key = openai_key,)

possible_genres = ["Clothes", "Electronics", "Books", "Toys", "Jewelry", "Home", "Beauty", "Sports", "Food", "Music", "Movies", "Games", "Art", "Travel", "Pets", "Health", "Fitness", "Tech", "DIY", "Gardening", "Cooking", "Crafts", "Cars", "Outdoors", "Office", "School", "Baby", "Party", "Wedding", "Holidays", "Grooming",]

em.config(openai_key)

cache, bucket = em.init("embedding_cache", possible_genres)

if "agent" not in st.session_state:
    st.session_state.agent = ao.Agent(arch, "agent")

if "started"    not in st.session_state:
    st.session_state.started = False

if "country_code" not in st.session_state:
    st.session_state.country_code = "us"

def llm_call(input_message): #llm call method 
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  
        messages=[
            {"role": "system", "content": "give a 5 options answer each only a couple of words long"},
            {"role": "user", "content": input_message}
        ],
         max_tokens=35,
        temperature=0.1
    )
    local_response = response.choices[0].message.content
    return local_response

def get_random_product(search_term):
    params = {
  "engine": "google_shopping",
  "q": f"{search_term}",
  "gl": f"{st.session_state.country_code}",
  "api_key": f"{SER_API_KEY}",
}

    search = GoogleSearch(params)
    results = search.get_dict()
    shopping_results = results["shopping_results"]

    random.shuffle(shopping_results)
    
    name = shopping_results[0]["title"]
    price = shopping_results[0]["price"]
    thumbnail = shopping_results[0]["thumbnail"]
    return name, price, thumbnail

def get_price_binary(price):
    # Convert price to binary
    match = re.search(r"[-+]?\d*\.\d+|\d+", price)

# Convert the match to a float
    if match:
        price = float(match.group())
    print(price)
    if price <25:
        price_binary = [0,0]
    elif price <50:
        price_binary = [0,1]
    elif price <100:
        price_binary = [1,0]
    else:
        price_binary = [1,1]
    return price_binary

def call_agent(input_to_agent):
    print(input_to_agent)
    response = st.session_state.agent.next_state(input_to_agent)
    return response

def train_agent(input_to_agent, label):
    if label == "Recommend":
        LABEL = [1,1,1,1,1,1,1,1,1,1]
        print("positive")
    else:
        print("negative")
        LABEL = [0,0,0,0,0,0,0,0,0,0]
    
    st.session_state.agent.next_state(input_to_agent, LABEL)
    

 

st.title("Gift Recommender")
st.write("Welcome to the gift recommender! We will help you find the perfect gift for your loved ones.")

x = st.selectbox("Country", country_names)
if x in country_names:
    for country in data:
        if country["country_name"] == str(x):
            st.session_state.country_code = country["country_code"]
            

age = st.number_input("How old is the person you are buying the gift for?", min_value=0, max_value=100, value=18)
gender = st.multiselect("What gender is the person you are buying the", ["Male", "Female", "Non-binary", "Prefer not to say"])
budget = st.number_input("What is your budget?", min_value=0, value=50)


request = str(("What are some gift catagories that meet the following: age: ",age, "gender: ", gender, "budget: ", budget))

if st.button("Find gifts"):
    st.session_state.started = True

if st.session_state.started:
    search_terms = []

    response = llm_call(request)

    search_terms = [line.split('. ', 1)[1] + ' ' for line in response.splitlines() if '. ' in line]

    random.shuffle(search_terms)    #Shuffle the search terms to get a random one

    search_term = search_terms[0]

    product_name, price, photos = get_random_product(search_term)

    cldis, genre, bucketid, genre_binary = em.auto_sort(cache, word=product_name, max_distance=10, bucket_array= bucket, type_of_distance_calc="COSINE SIMILARITY", amount_of_binary_digits=10)
 
    price_binary = np.array(get_price_binary(price))


    input_to_agent = np.concatenate([price_binary, genre_binary])

    response = call_agent(input_to_agent)

    counter = 0
    for element in response:
        if element == 1:
            counter += 1
    
    percentage_response = str((counter/len(response))*100) + "%"

    st.write("Agent Recommendation: ", percentage_response)

    if st.button("Recommend More"):
        train_agent(input_to_agent, "Recommend")
    if st.button("Recommend Less"):
        train_agent(input_to_agent, "Don't Recommend")
    

    

    st.write(f"Here is a gift idea for you: {product_name}")
    st.write(f"Price: {price}")
    st.write(f"Genre: {genre}")
    st.image(photos, caption=product_name)


