import json

def search_flight(source, destination):
    with open("data/flights.json", "r") as f:
        flights = json.load(f)

    matches = [
        f for f in flights
        if f["from"].lower() == source.lower()
        and f["to"].lower() == destination.lower()
    ]

    if not matches:
        return None

    return min(matches, key=lambda x: x["price"])
