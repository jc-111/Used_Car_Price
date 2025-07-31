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

def generate_realistic_car(real_df):
    row = real_df.sample(n=1).iloc[0]
    return {
        "year": row["year"],
        "car_age": pd.Timestamp.now().year - row["year"],
        "odometer": row["odometer"],
        "manufacturer": row["manufacturer"],
        "model": row["model"],
        "region": row["region"],
        "fuel": row["fuel"],
        "transmission": row["transmission"],
        "drive": row["drive"],
        "type": row["type"],
        "paint_color": row["paint_color"],
        "state": row["state"]
        # Optionally: "price": row["price"]
    }

def generate_realistic_dataset(real_df, n=20):
    data = [generate_realistic_car(real_df) for _ in range(n)]
    return pd.DataFrame(data)

if __name__ == "__main__":
    real_df = pd.read_csv("data/vehicles.csv")
    df = generate_realistic_dataset(real_df, n=20)
    df.to_csv("realistic_fake_cars_for_prediction.csv", index=False)
    print("Realistic batch prediction data saved to realistic_fake_cars_for_prediction.csv")