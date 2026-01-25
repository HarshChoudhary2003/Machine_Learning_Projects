def estimate_budget(flight_price, hotel_price, days):
    food_cost = days * 800
    return {
        "Flight": flight_price,
        "Hotel": hotel_price * days,
        "Food & Local Travel": food_cost,
        "Total": flight_price + (hotel_price * days) + food_cost
    }
