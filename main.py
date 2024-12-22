import streamlit as st

# for getting random links
import random
from serpapi import GoogleSearch
from openai import OpenAI

# for embedding model
import embedding_bucketing.embedding_model_test as em


from config import openai_key
from config import SER_API_KEY

client = OpenAI(api_key = openai_key,)

em.config(openai_key)


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
  "api_key": "6673a88cd7070e11560e328a12075f405db2d3a8b63c677d8dbeefa2a7a631ae"
}

    search = GoogleSearch(params)
    results = search.get_dict()
    shopping_results = results["shopping_results"]

    random.shuffle(shopping_results)
    
    name = shopping_results[0]["title"]
    price = shopping_results[0]["price"]
    thumbnail = shopping_results[0]["thumbnail"]
    return name, price, thumbnail

 

st.title("Gift Recommender")
st.write("Welcome to the gift recommender! We will help you find the perfect gift for your loved ones.")

age = st.number_input("How old is the person you are buying the gift for?", min_value=0, max_value=100, value=18)
gender = st.multiselect("What gender is the person you are buying the", ["Male", "Female", "Non-binary", "Prefer not to say"])
budget = st.number_input("What is your budget?", min_value=0, value=50)

request = str(("What are some gift catagories that meet the following: age: ",age, "gender: ", gender, "budget: ", budget))

if st.button("Find gifts"):
    search_terms = []

    response = llm_call(request)

    search_terms = [line.split('. ', 1)[1] + ' ' for line in response.splitlines() if '. ' in line]

    cache, buckets= em.init("genre_embedding_cache", search_terms)  #Initialize the embedding model with the search terms

    random.shuffle(search_terms)    #Shuffle the search terms to get a random one

    search_term = search_terms[0]

    st.write(f"Searching for gifts based on the following search term: {search_term}")

    product_name, price, photos = get_random_product(search_term)

    cldis, genre, bucketid, genre_binary = em.auto_sort(cache, word= product_name, max_distance=10, bucket_array= search_terms, type_of_distance_calc="COSINE SIMILARITY", amount_of_binary_digits=5)


    st.write(f"Here is a gift idea for you: {product_name}")
    st.write(f"Price: {price}")
    st.write(f"Genre: {genre}")
    st.image(photos, caption=product_name)

