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

import httpx
from parsel import Selector
from urllib.parse import urlencode


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

session = httpx.Client(
    headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36 Edg/113.0.1774.35",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
    },
    http2=True,
    follow_redirects=True
)

def parse_product(response):

    """Parse Ebay's product listing page for core product data"""
    sel = Selector(response.text)
    css_join = lambda css: "".join(sel.css(css).getall()).strip()  # join all selected elements
    css = lambda css: sel.css(css).get("").strip()  # take first selected element and strip of leading/trailing spaces

    item = {}
    item["url"] = css('link[rel="canonical"]::attr(href)')
    if item["url"]:
        item["id"] = item["url"].split("/itm/")[1].split("?")[0]  # extract ID from the URL
    item["price_original"] = css(".x-price-primary>span::text")
    item["price_converted"] = css(".x-price-approx__price ::text")  # ebay automatically converts price for some regions
    item["name"] = css_join("h1 span::text")
    item["seller_name"] = sel.xpath("//div[contains(@class,'info__about-seller')]/a/span/text()").get()
    item["seller_url"] = sel.xpath("//div[contains(@class,'info__about-seller')]/a/@href").get()
    if item["seller_url"]:
        item["seller_url"] = item["seller_url"].split("?")[0]
    item["photos"] = sel.css('.ux-image-filmstrip-carousel-item.image img::attr("src")').getall()
    item["photos"].extend(sel.css('.ux-image-carousel-item.image img::attr("src")').getall())
    item["description_url"] = css("iframe#desc_ifr::attr(src)")

    features = {}
    feature_table = sel.css("div.ux-layout-section--features")
    for feature in feature_table.css("dl.ux-labels-values"):
        label = "".join(feature.css(".ux-labels-values__labels-content > div > span::text").getall()).strip(":\n ")
        value = "".join(feature.css(".ux-labels-values__values-content > div > span *::text").getall()).strip(":\n ")
        features[label] = value
    item["features"] = features

    return item

def get_random_product(query):
    limit = 10
    country_domains = {
    "us": "ebay.com",
    "gb": "ebay.co.uk",
    "de": "ebay.de",
    "fr": "ebay.fr",
    "ca": "ebay.ca",
    "au": "ebay.com.au",
    "it": "ebay.it",
    "es": "ebay.es",
    "nl": "ebay.nl",
    "at": "ebay.at",
    "be": "ebay.be",
    "ch": "ebay.ch",
    "ie": "ebay.ie",
    "pl": "ebay.pl",
    "in": "ebay.in",
    "sg": "ebay.sg",
    "my": "ebay.com.my",
    "ph": "ebay.ph",
    "hk": "ebay.com.hk",
    "cn": "ebay.cn",
    "jp": "ebay.co.jp",
    "kr": "ebay.co.kr",
    "tw": "ebay.com.tw",
    "th": "ebay.co.th",
    "vn": "ebay.vn",
    "id": "ebay.co.id",
    "mx": "mx.ebay.com",
    "br": "ebay.com.br",
    "ar": "ebay.com.ar",
    "cl": "ebay.cl",
    "co": "ebay.com.co",
    "pe": "ebay.com.pe",
    "ve": "ebay.com.ve",
    "za": "ebay.co.za",
    "eg": "ebay.com.eg",
    "sa": "ebay.com.sa",
    "ae": "ebay.ae",
    "il": "ebay.co.il",
    "ru": "ebay.ru",
    "ua": "ebay.com.ua",
    "tr": "ebay.com.tr",
    "cz": "ebay.cz",
    "sk": "ebay.sk",
    "hu": "ebay.hu",
    "ro": "ebay.ro",
    "gr": "ebay.gr",
    "se": "ebay.se",
    "no": "ebay.no",

    # Add more country codes asap
}
    
    base_url = f"https://{country_domains.get(st.session_state.country_code, 'ebay.com')}/sch/i.html"

    print("using base url: ", base_url)
    params = {
        "_nkw": query,  # search keyword
        "_ipg": limit,  # items per page
        "_sop": 12,     # best match sort
    }
    search_url = f"{base_url}?{urlencode(params)}"
    response = session.get(search_url)

    if response.status_code != 200:
        raise Exception(f"Failed to fetch search results: {response.status_code}")

    sel = Selector(response.text)
    product_links = sel.css(".s-item__link::attr(href)").getall()[:limit]

    products = []
    for link in product_links:
        product_response = session.get(link)
        product_data = parse_product(product_response)
        products.append(product_data)

    random.shuffle(products)

    return products[0]["name"], products[0]["price_original"], products[0]["photos"][0]

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


