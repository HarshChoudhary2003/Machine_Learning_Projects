import json

def suggest_places(city):
    with open("data/places.json", "r") as f:
        places = json.load(f)

    city_places = [p for p in places if p["city"].lower() == city.lower()]
    return sorted(city_places, key=lambda x: x["rating"], reverse=True)[:5]
    return sorted(matches, key=lambda x: (-x["stars"], x["price_per_night"]))[0]