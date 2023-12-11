#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import networkx as nx

## Question 1: Distance Matrix Calculation

def calculate_distance_matrix(df)->pd.DataFrame():
    # Read the CSV file into a DataFrame
#     df = pd.read_csv(csv_file)

    # Create an undirected graph from the DataFrame
    G = nx.from_pandas_edgelist(df, 'id_start', 'id_end', ['distance'])

    # Get a sorted list of nodes
    nodes = sorted(list(G.nodes))

    # Initialize an empty distance matrix
    num_nodes = len(nodes)
    distance_matrix = [[float('inf')] * num_nodes for _ in range(num_nodes)]

    # Compute the shortest path lengths between all pairs of nodes
    for i in range(num_nodes):
        for j in range(num_nodes):
            try:
                distance_matrix[i][j] = nx.shortest_path_length(G, source=nodes[i], target=nodes[j], weight='distance')
            except nx.NetworkXNoPath:
                # Handle the case where there is no path between nodes
                distance_matrix[i][j] = float('inf')

    # Make the matrix symmetric
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            distance_matrix[j][i] = distance_matrix[i][j]

    # Convert the distance matrix to a DataFrame
    df = pd.DataFrame(distance_matrix, index=nodes, columns=nodes)

    return df


## Question 2: Unroll Distance Matrix

def unroll_distance_matrix(df)->pd.DataFrame():
    
    distance_matrix_df = calculate_distance_matrix(df)
    # Reset index to get 'id_start' as a column
    distance_matrix_df_reset = distance_matrix_df.reset_index()

    # Melt the DataFrame to convert it to long format
    melted_df = pd.melt(distance_matrix_df_reset, id_vars='index', var_name='id_end', value_name='distance')

    # Rename columns for consistency
    melted_df.columns = ['id_start', 'id_end', 'distance']

    # Exclude rows where 'id_start' is equal to 'id_end'
    melted_df = melted_df[melted_df['id_start'] != melted_df['id_end']]

    # Reset index and drop the old index column
    melted_df_reset = melted_df.reset_index(drop=True)

    return melted_df_reset


## Question 3: Finding IDs within Percentage Threshold

def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    unrolled_df = unroll_distance_matrix(df)
    # Ensure the input DataFrame has the correct format
    if not isinstance(unrolled_df, pd.DataFrame):
        raise ValueError("Input must be a DataFrame")

    # Filter rows for the given reference_id
    reference_data = unrolled_df[unrolled_df['id_start'] == reference_id]

    # Calculate the average distance for the reference_id
    average_distance = reference_data['distance'].mean()

    # Calculate the threshold for 10% (including ceiling and floor)
    lower_threshold = average_distance - (average_distance * 0.1)
    upper_threshold = average_distance + (average_distance * 0.1)

    # Filter rows within the threshold and extract unique id_start values
    filtered_ids = unrolled_df[(unrolled_df['distance'] >= lower_threshold) & (unrolled_df['distance'] <= upper_threshold)]['id_start'].unique()

    # Sort and return the list of id_start values
    sorted_filtered_ids = sorted(filtered_ids)

    return sorted_filtered_ids
# Example usage
# reference_id = 1001400  # Replace with the desired reference_id
# sorted_filtered_ids = find_ids_within_ten_percentage_threshold(df, reference_id)

# # print(f"Reference ID: {reference_id}")
# print(f"IDs within 10% threshold: {sorted_filtered_ids}")

## Question 4: Calculate Toll Rate

def calculate_toll_rate(df)->pd.DataFrame():
    unrolled_df = unroll_distance_matrix(df)
    # Ensure the input DataFrame has the correct format
    if not isinstance(unrolled_df, pd.DataFrame):
        raise ValueError("Input must be a DataFrame")

    # Define rate coefficients for each vehicle type
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    # Add columns for toll rates based on vehicle types
    for vehicle_type, rate_coefficient in rate_coefficients.items():
        unrolled_df[vehicle_type] = unrolled_df['distance'] * rate_coefficient
    unrolled_df =unrolled_df.sort_values(by='id_start')

    return unrolled_df

from datetime import datetime, timedelta, time


## Question 5: Calculate Time-Based Toll Rates


def calculate_time_based_toll_rates(df)->pd.DataFrame():
    toll_rate_df = calculate_toll_rate(df)
    # Ensure the input DataFrame has the correct format
    if not isinstance(toll_rate_df, pd.DataFrame):
        raise ValueError("Input must be a DataFrame")

    # Convert the index to DatetimeIndex
    toll_rate_df.index = pd.to_datetime(toll_rate_df.index)

    # Define time ranges and discount factors
    time_ranges_weekdays = [
        (time(0, 0, 0), time(10, 0, 0), 0.8),
        (time(10, 0, 0), time(18, 0, 0), 1.2),
        (time(18, 0, 0), time(23, 59, 59), 0.8)
    ]
    time_ranges_weekends = [
        (time(0, 0, 0), time(23, 59, 59), 0.7)
    ]

    # Initialize an empty list to store modified data
    modified_data = []

    # Iterate over unique ('id_start', 'id_end') pairs
    unique_pairs = toll_rate_df[['id_start', 'id_end']].drop_duplicates()
    for _, pair in unique_pairs.iterrows():
        for day in range(7):  # Iterate over days (0 for Monday, 6 for Sunday)
            for start_time, end_time, discount_factor in (time_ranges_weekdays if day < 5 else time_ranges_weekends):
                start_datetime = datetime.combine(datetime(2023, 1, 1), start_time) + timedelta(days=day)
                end_datetime = datetime.combine(datetime(2023, 1, 1), end_time) + timedelta(days=day)

                # Filter rows for the specific time range and ('id_start', 'id_end') pair
                mask = (toll_rate_df['id_start'] == pair['id_start']) & (toll_rate_df['id_end'] == pair['id_end'])
                mask &= (toll_rate_df.index.time >= start_datetime.time()) & (toll_rate_df.index.time <= end_datetime.time())
                selected_rows = toll_rate_df.loc[mask].copy()

                # Add start_day, start_time, end_day, and end_time columns
                selected_rows['start_day'] = start_datetime.strftime('%A')
                selected_rows['start_time'] = start_datetime.time()
                selected_rows['end_day'] = end_datetime.strftime('%A')
                selected_rows['end_time'] = end_datetime.time()

                # Apply the discount factor to the vehicle columns using .loc
                for vehicle_type in ['moto', 'car', 'rv', 'bus', 'truck']:
                    selected_rows.loc[:, vehicle_type] *= discount_factor

                # Update the modified_data list with the modified rows
                modified_data.extend(selected_rows.to_dict('records'))

    # Create a DataFrame from the modified data
    modified_df = pd.DataFrame(modified_data)

    return modified_df



# In[ ]:




