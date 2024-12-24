import json
import httpx
from parsel import Selector
from urllib.parse import urlencode

# establish our HTTP2 client with browser-like headers
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
    """Parse eBay's product listing page for core product data."""
    sel = Selector(response.text)
    css_join = lambda css: "".join(sel.css(css).getall()).strip()
    css = lambda css: sel.css(css).get("").strip()

    item = {}
    item["url"] = css('link[rel="canonical"]::attr(href)')
    if item["url"]:
        item["id"] = item["url"].split("/itm/")[1].split("?")[0]
    item["price_original"] = css(".x-price-primary>span::text")
    item["price_converted"] = css(".x-price-approx__price ::text")
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

def search_ebay(query: str, country: str, limit: int = 5):
    """Search eBay with the given query and parse the top results."""
    tld = country if country in ["com", "co.uk", "ca", "de", "com.au", "fr", "it", "es"] else "com"
    base_url = f"https://www.ebay.{tld}/sch/i.html"
    params = {
        "_nkw": query,
        "_ipg": limit,
        "_sop": 12,
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

    return products


if __name__ == "__main__":
    query = input("Enter search query: ").strip()
    country = input("Enter country TLD (e.g., 'com', 'co.uk', 'de', 'ca', 'com.au'): ").strip().lower()
    results = search_ebay(query, country)

    # Print the results in JSON format
    print(json.dumps(results, indent=2))
