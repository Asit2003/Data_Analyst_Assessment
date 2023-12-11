import pandas as pd
import numpy as np

## Question 1: Car Matrix Generation
def generate_car_matrix(df)->pd.DataFrame:
    # Pivot the DataFrame to create a matrix with id_1 as index, id_2 as columns, and car values as data
    car_matrix = df.pivot(index='id_1', columns='id_2', values='car').fillna(0)
    
    # Fill diagonal values with 0
    for idx in car_matrix.index:
        if idx in car_matrix.columns:
            car_matrix.loc[idx, idx] = 0
    
    return car_matrix

## Question 2: Car Type Count Calculation

def get_type_count(df)->dict:
    # Add a new categorical column 'car_type' based on values of the column 'car'
    conditions = [
        (df['car'] <= 15),
        ((df['car'] > 15) & (df['car'] <= 25)),
        (df['car'] > 25)
    ]
    choices = ['low', 'medium', 'high']
    df['car_type'] = pd.Series(np.select(conditions, choices), dtype="category")

    # Calculate the count of occurrences for each 'car_type' category
    type_counts = df['car_type'].value_counts().to_dict()

    # Sort the dictionary alphabetically based on keys
    sorted_type_counts = dict(sorted(type_counts.items()))

    return sorted_type_counts


## Question 3: Bus Count Index Retrieval
def get_bus_indexes(df)->list:
    # Calculate the mean value of the 'bus' column
    mean_bus = df['bus'].mean()

    # Identify indices where 'bus' values are greater than twice the mean value
    bus_indexes = df[df['bus'] > 2 * mean_bus].index.tolist()

    # Sort the indices in ascending order
    bus_indexes.sort()

    return bus_indexes

## Question 4: Route Filtering
def filter_routes(df)->list:
    # Group by 'route' and calculate the average of 'truck' values
    route_avg_truck = df.groupby('route')['truck'].mean()

    # Filter routes where the average of 'truck' values is greater than 7
    selected_routes = route_avg_truck[route_avg_truck > 7].index.tolist()

    # Sort the list of selected routes
    selected_routes.sort()

    return selected_routes

## Question 5: Matrix Value Modification

def multiply_matrix(df)->pd.DataFrame:
    matrix = generate_car_matrix(df)
    # Deep copy the input DataFrame to avoid modifying the original DataFrame
    modified_matrix = matrix.copy()

    # Apply the modification logic
    modified_matrix[modified_matrix > 20] *= 0.75
    modified_matrix[modified_matrix <= 20] *= 1.25

    # Round the values to 1 decimal place
    modified_matrix = modified_matrix.round(1)

    return modified_matrix

## Question 6: Time Check
def time_check(df)->pd.Series:
    # Convert start and end timestamps to datetime objects
    df['start_datetime'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'], format='%A %H:%M:%S')
    df['end_datetime'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'], format='%A %H:%M:%S')

    # Calculate the difference in time for each row
    df['time_difference'] = df['end_datetime'] - df['start_datetime']

    # Group by ('id', 'id_2') and check for completeness
    completeness_check = df.groupby(['id', 'id_2'])['time_difference'].agg(lambda x: x.sum() >= pd.Timedelta(days=7) and x.max() >= pd.Timedelta(hours=24))

    return completeness_check


