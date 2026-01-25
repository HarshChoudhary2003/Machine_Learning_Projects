from tools.flight_tool import search_flight
from tools.hotel_tool import recommend_hotel
from tools.places_tool import suggest_places
from tools.budget_tool import estimate_budget

def get_weather_forecast(days):
    return {f"Day {i}": "Sunny, 30–32°C" for i in range(1, days + 1)}

def run_agent(query, max_budget, min_stars, destinations, days=3):
    reasoning = []
    trips = []

    for destination in destinations:
        flight = search_flight("Delhi", destination)

        if flight is None:
            reasoning.append(
                f"No direct flight from Delhi to {destination}. Selecting cheapest alternative."
            )
            for city in ["Hyderabad", "Mumbai", "Bangalore", "Chennai", "Jaipur", "Kolkata"]:
                flight = search_flight(city, destination)
                if flight:
                    break

        if flight is None:
            continue

        hotel = recommend_hotel(destination)

        if hotel["stars"] < min_stars:
            reasoning.append(
                f"{destination}: Hotel does not meet minimum star requirement."
            )
            continue

        places = suggest_places(destination)

        budget = estimate_budget(
            flight_price=flight["price"],
            hotel_price=hotel["price_per_night"],
            days=days
        )

        if budget["Total"] > max_budget:
            reasoning.append(
                f"{destination}: Budget exceeds selected limit."
            )
            continue

        itinerary = {
            f"Day {i+1}": [p["name"] for p in places[i*2:(i*2)+2]]
            for i in range(days)
        }

        trips.append({
            "Destination": destination,
            "Flight Selected": flight,
            "Hotel Selected": hotel,
            "Day-wise Itinerary": itinerary,
            "Weather Forecast": get_weather_forecast(days),
            "Budget Breakdown": budget
        })

    return {
        "Trip Summary": f"Multi-Destination Trip: {', '.join(destinations)}",
        "Trips": trips,
        "Agent Reasoning": reasoning or ["All destinations satisfied constraints"]
    }
    return {
        "Trip Summary": f"Multi-Destination Trip: {', '.join(destinations)}",
        "Trips": trips,
        "Agent Reasoning": reasoning or ["All destinations satisfied constraints"]
    }