import requests


url = 'http://localhost:9696/predict'

flight = {
    "Unnamed: 0": 1,
    "airline": "Indigo",
    "flight": "6E-789",
    "source_city": "Delhi",
    "departure_time": "Morning",
    "stops": "zero",
    "arrival_time": "Afternoon",
    "destination_city": "Mumbai",
    "class": "Economy",
    "duration": 2.5,
    "days_left": 15
}


response = requests.post(url, json=flight).json()
print(response)
