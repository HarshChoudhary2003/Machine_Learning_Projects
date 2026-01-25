import json

def recommend_hotel(city):
    with open("data/hotels.json", "r") as f:
        hotels = json.load(f)

    matches = [h for h in hotels if h["city"].lower() == city.lower()]

    if not matches:
        return None

    return sorted(matches, key=lambda x: (-x["stars"], x["price_per_night"]))[0]
