import requests

url ='https://duration-pred-serve.fly.dev/predict'

trip = {    'PULocationID': '100',
    'DOLocationID': '102',
    'trip_distance': 30,
}

response = requests.post(url, json=trip).json()
print(response)