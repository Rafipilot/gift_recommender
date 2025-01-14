import streamlit as st
import numpy as np

from urllib.parse import quote
# for getting random links
import random
from openai import OpenAI

import embedding_bucketing.embedding_model_test as em

from config import openai_key, Rapid_key


import ao_core as ao
from Arch__giftRecommender import arch

import json
import re

import httpx
import http.client
from parsel import Selector
from urllib.parse import urlencode


with open("google-countries.json") as f:
    data = json.load(f)

st.set_page_config(
    page_title="Recommender Demo by AO Labs",
    page_icon="misc/ao_favicon.png",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://discord.gg/Zg9bHPYss5",
        "Report a bug": "mailto:eng@aolabs.ai",
        "About": "AO Labs builds next-gen AI models that learn after training; learn more at docs.aolabs.ai/docs/mnist-benchmark",
    },
)

st.title("Gift Recommender")
st.write("Welcome to the gift recommender! We will help you find the perfect gift for your loved ones.")


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


def get_random_product(query, budget):
    # Encode the query string to ensure no invalid characters
    encoded_query = quote(query)
    
    conn = http.client.HTTPSConnection("real-time-amazon-data.p.rapidapi.com")

    headers = {
        'x-rapidapi-key': f"{Rapid_key}",
        'x-rapidapi-host': "real-time-amazon-data.p.rapidapi.com"
    }

    # Use the encoded query in the request
    conn.request("GET", f"/search?query={encoded_query}&page=1&country=US&sort_by=RELEVANCE&product_condition=ALL&is_prime=false&deals_and_discounts=NONE", headers=headers)

    res = conn.getresponse()
    data = json.loads(res.read().decode("utf-8"))
    products = data["data"]["products"]

    random.shuffle(products)
    name = products[0]["product_title"]
    price = products[0]["product_original_price"]
    photo = products[0]["product_photo"]

    if price:
        pass
    else:
        price = 0

    return name, price, photo

def get_price_binary(price):
    # Convert price to binary
    match = re.search(r"[-+]?\d*\.\d+|\d+", price)
    print("price: ",price)

# Convert the match to a float
    if match:
        price = float(match.group())
    else:
        price = float(price)
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
    
col_left, col_right = st.columns(2)



with col_left:
    x = st.selectbox("Country", country_names)
    if x in country_names:
        for country in data:
            if country["country_name"] == str(x):
                st.session_state.country_code = country["country_code"]
                

    age = st.number_input("How old is the person you are buying the gift for?", min_value=0, max_value=100, value=18)
    gender = st.multiselect("What gender is the person you are buying the", ["Male", "Female", "Non-binary", "Prefer not to say"])
    budget = st.number_input("What is your budget?", min_value=0, value=50)


request = str(("What are some gift catagories that meet the following: age: ",age, "gender: ", gender, "budget: ", budget))

with col_right:
    if st.button("Find gifts"):
        st.session_state.started = True

    if st.session_state.started:
        search_terms = []

        response = llm_call(request)

        search_terms = [line.split('. ', 1)[1] + ' ' for line in response.splitlines() if '. ' in line]

        random.shuffle(search_terms)    #Shuffle the search terms to get a random one

        search_term = search_terms[0]

        product_name, price, photos = get_random_product(search_term, budget)

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

        col_left, col_right = st.columns(2)

        with col_left:
            if st.button("Recommend More"):
                train_agent(input_to_agent, "Recommend")
        with col_right:
            if st.button("Recommend Less"):
                train_agent(input_to_agent, "Don't Recommend")
            

        

        st.write(f"Here is a gift idea for you: {product_name}")
        st.write(f"Price: {price}")
        st.write(f"Genre: {genre}")
        st.image(photos, caption=product_name)


