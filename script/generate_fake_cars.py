import pandas as pd
import numpy as np
from faker import Faker
import random

fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)

# Define possible values for categorical columns
manufacturers = ['toyota', 'ford', 'chevrolet', 'honda', 'nissan', 'bmw', 'mercedes-benz', 'hyundai', 'kia', 'volkswagen']
models = ['camry', 'civic', 'accord', 'corolla', 'f-150', 'silverado', 'altima', 'elantra', 'optima', 'jetta']
regions = ['los angeles', 'new york', 'chicago', 'houston', 'phoenix', 'philadelphia', 'san antonio', 'san diego', 'dallas', 'san jose']
fuels = ['gas', 'diesel', 'electric', 'hybrid', 'other']
transmissions = ['automatic', 'manual', 'other']
drives = ['fwd', 'rwd', '4wd', 'other']
types = ['sedan', 'suv', 'truck', 'coupe', 'convertible', 'wagon', 'van', 'hatchback', 'pickup', 'other']
paint_colors = ['white', 'black', 'silver', 'blue', 'red', 'gray', 'green', 'yellow', 'brown', 'other']
states = ['ca', 'ny', 'tx', 'fl', 'il', 'pa', 'oh', 'ga', 'nc', 'mi']

def generate_fake_car():
    year = random.randint(1995, 2023)
    posting_year = random.randint(year, 2024)
    odometer = int(np.clip(np.random.normal(80000, 40000), 0, 300000))
    car_age = posting_year - year
    return {
        "year": year,
        "car_age": car_age,
        "odometer": odometer,
        "manufacturer": random.choice(manufacturers),
        "model": random.choice(models),
        "region": random.choice(regions),
        "fuel": random.choice(fuels),
        "transmission": random.choice(transmissions),
        "drive": random.choice(drives),
        "type": random.choice(types),
        "paint_color": random.choice(paint_colors),
        "state": random.choice(states),
        # Optionally add a fake price for validation/testing
        "price": int(np.clip(np.random.normal(15000, 8000), 2000, 60000))
    }

def generate_fake_dataset(n=20, include_price=True):
    data = [generate_fake_car() for _ in range(n)]
    df = pd.DataFrame(data)
    if not include_price:
        df = df.drop(columns=['price'])
    return df

if __name__ == "__main__":
    # Generate 20 fake cars for batch prediction (no price column)
    df = generate_fake_dataset(n=20, include_price=False)
    df.to_csv("fake_cars_for_prediction.csv", index=False)
    print("Fake batch prediction data saved to fake_cars_for_prediction.csv")

    # Generate 20 fake cars for validation (with price column)
    df_val = generate_fake_dataset(n=20, include_price=True)
    df_val.to_csv("fake_cars_for_validation.csv", index=False)
    print("Fake validation data saved to fake_cars_for_validation.csv")