import http.client
import json
import random

def get_random_product(query, budget):
    conn = http.client.HTTPSConnection("real-time-amazon-data.p.rapidapi.com")

    headers = {
        'x-rapidapi-key': "723e3b8057msh611f8822fbca1b7p12867djsnb164e99788e3",
        'x-rapidapi-host': "real-time-amazon-data.p.rapidapi.com"
    }

    conn.request("GET", "/search?query=Drone&page=1&country=US&sort_by=RELEVANCE&min_price=20&product_condition=ALL&is_prime=false&deals_and_discounts=NONE", headers=headers)

    res = conn.getresponse()
    data = json.loads(res.read().decode("utf-8"))
    products = data["data"]["products"]

    random.shuffle(products)
    name = products[0]["product_title"]
    price = products[0]["product_original_price"]
    photo = products[0]["product_photo"]

    return name, price, photo
name, price, photo = get_random_product("laptop", 100)
print(name, price, photo)