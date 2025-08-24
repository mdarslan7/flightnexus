import pandas as pd
import networkx as nx
import os

def find_critical_flights():
    data_path = os.path.join('data', 'cleaned_flight_data.csv')
    df = pd.read_csv(data_path, parse_dates=['STD_datetime', 'ATA_datetime'])

    df.dropna(subset=['Flight Number'], inplace=True)
    df = df[df['Flight Number'].str.strip() != '']

    df.sort_values(by='STD_datetime', inplace=True)
    df.reset_index(drop=True, inplace=True)

    G = nx.DiGraph()

    for idx, row in df.iterrows():
        G.add_node(idx, flight_number=row['Flight Number'], aircraft=row['Aircraft'])

    aircraft_groups = df.groupby('Aircraft')
    for _, group in aircraft_groups:
        for i in range(len(group) - 1):
            current_flight_idx = group.index[i]
            next_flight_idx = group.index[i+1]
            if df.loc[next_flight_idx, 'STD_datetime'] > df.loc[current_flight_idx, 'ATA_datetime']:
                 G.add_edge(current_flight_idx, next_flight_idx)

    centrality = nx.out_degree_centrality(G)
    df['centrality'] = df.index.map(centrality)

    critical_flights = df.sort_values(by='centrality', ascending=False).head(10)

    output_path = os.path.join('data', 'critical_flights.csv')
    critical_flights.to_csv(output_path, index=False)
    print(f"Critical flights analysis saved to {output_path}")

if __name__ == '__main__':
    find_critical_flights()