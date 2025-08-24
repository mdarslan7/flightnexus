import pandas as pd
import numpy as np
import networkx as nx
import os
def enhanced_critical_flights_analysis(df):
    print("--- Enhanced Critical Flights Analysis Started ---")
    G = nx.DiGraph()

    flight_nodes = {}
    for idx, row in df.iterrows():
        if pd.isna(row['Flight Number']) or row['Flight Number'].strip() == '':
            continue
        flight_id = f"{row['Flight Number']}_{row['Date']}_{row['STD_datetime'].strftime('%H%M')}"

        G.add_node(flight_id, 
                  aircraft=row['Aircraft'],
                  route=row['From'] + '_' + row['To'],
                  hour=row['STD_datetime'].hour,
                  delay=row.get('departure_delay', 0),
                  flight_number=row['Flight Number'],
                  date=row['Date'],
                  std_time=row['STD_datetime'],
                  ata_time=row['ATA_datetime'])
        flight_nodes[flight_id] = row

        same_aircraft_flights = df[
            (df['Aircraft'] == row['Aircraft']) & 
            (df['STD_datetime'] > row['ATA_datetime']) &
            (df['STD_datetime'] <= row['ATA_datetime'] + pd.Timedelta(hours=6))
        ]
        for _, next_flight in same_aircraft_flights.iterrows():
            if pd.isna(next_flight['Flight Number']) or next_flight['Flight Number'].strip() == '':
                continue
            next_flight_id = f"{next_flight['Flight Number']}_{next_flight['Date']}_{next_flight['STD_datetime'].strftime('%H%M')}"
            if next_flight_id != flight_id and next_flight_id not in G.nodes():
                continue
            turnaround_time = (next_flight['STD_datetime'] - row['ATA_datetime']).total_seconds() / 3600
            if turnaround_time > 0:
                weight = 1 / max(turnaround_time, 0.5)
                G.add_edge(flight_id, next_flight_id, 
                          weight=weight,
                          connection_type='aircraft_routing',
                          turnaround_hours=turnaround_time)

        if 'Mumbai (BOM)' in row['From']:
            connecting_flights = df[
                (df['STD_datetime'] >= row['STA_datetime'] + pd.Timedelta(minutes=60)) &
                (df['STD_datetime'] <= row['STA_datetime'] + pd.Timedelta(hours=4)) &
                (df['From'].str.contains('Mumbai', na=False))
            ]
            for _, conn_flight in connecting_flights.iterrows():
                if pd.isna(conn_flight['Flight Number']) or conn_flight['Flight Number'].strip() == '':
                    continue
                conn_flight_id = f"{conn_flight['Flight Number']}_{conn_flight['Date']}_{conn_flight['STD_datetime'].strftime('%H%M')}"
                if conn_flight_id != flight_id:
                    G.add_edge(flight_id, conn_flight_id, 
                              weight=0.4,
                              connection_type='passenger_transfer')
    print(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    centrality_measures = {}
    if len(G.nodes()) > 0:
        print("Calculating centrality measures...")
        try:
            centrality_measures['degree'] = nx.degree_centrality(G)
        except:
            centrality_measures['degree'] = {node: 0 for node in G.nodes()}
        try:
            centrality_measures['betweenness'] = nx.betweenness_centrality(G)
        except:
            centrality_measures['betweenness'] = {node: 0 for node in G.nodes()}
        try:
            centrality_measures['closeness'] = nx.closeness_centrality(G)
        except:
            centrality_measures['closeness'] = {node: 0 for node in G.nodes()}
        try:
            centrality_measures['eigenvector'] = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-06)
        except:
            centrality_measures['eigenvector'] = {node: 0 for node in G.nodes()}
        try:
            centrality_measures['pagerank'] = nx.pagerank(G)
        except:
            centrality_measures['pagerank'] = {node: 0 for node in G.nodes()}

    combined_scores = {}
    for node in G.nodes():
        score = (
            centrality_measures.get('degree', {}).get(node, 0) * 0.3 +
            centrality_measures.get('betweenness', {}).get(node, 0) * 0.3 +
            centrality_measures.get('pagerank', {}).get(node, 0) * 0.4
        )
        combined_scores[node] = score

    critical_flights_data = []
    sorted_nodes = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    for node, score in sorted_nodes[:20]:
        if node in flight_nodes:
            flight_data = flight_nodes[node]
            critical_flights_data.append({
                'flight_id': node,
                'Flight Number': flight_data['Flight Number'],
                'Date': flight_data['Date'],
                'Aircraft': flight_data['Aircraft'],
                'To': flight_data['To'],
                'STD_datetime': flight_data['STD_datetime'],
                'combined_centrality': score,
                'degree_centrality': centrality_measures.get('degree', {}).get(node, 0),
                'betweenness_centrality': centrality_measures.get('betweenness', {}).get(node, 0),
                'pagerank_score': centrality_measures.get('pagerank', {}).get(node, 0),
                'connections_out': G.out_degree(node),
                'connections_in': G.in_degree(node)
            })
    critical_df = pd.DataFrame(critical_flights_data)
    if len(critical_df) > 0:

        simple_critical = critical_df[['Flight Number', 'Date', 'Aircraft', 'To', 'STD_datetime']].copy()
        simple_critical['centrality'] = critical_df['combined_centrality']

        output_path = os.path.join('data', 'critical_flights.csv')
        simple_critical.to_csv(output_path, index=False)
        enhanced_output_path = os.path.join('data', 'enhanced_critical_flights.csv')
        critical_df.to_csv(enhanced_output_path, index=False)
        print(f"Critical flights analysis saved to {output_path}")
        print(f"Enhanced analysis saved to {enhanced_output_path}")

        print(f"\n--- Top 5 Critical Flights ---")
        for i, row in critical_df.head(5).iterrows():
            print(f"{row['Flight Number']} on {row['Date']}: {row['combined_centrality']:.4f}")
    else:
        print("No critical flights identified - creating empty files")
        empty_df = pd.DataFrame(columns=['Flight Number', 'Date', 'Aircraft', 'To', 'STD_datetime', 'centrality'])
        empty_df.to_csv(os.path.join('data', 'critical_flights.csv'), index=False)
    return critical_df if len(critical_df) > 0 else None, G
def find_critical_flights():

    data_path = os.path.join('data', 'cleaned_flight_data.csv')
    df = pd.read_csv(data_path, parse_dates=['STD_datetime', 'ATA_datetime', 'STA_datetime', 'ATD_datetime'])

    df['departure_delay'] = (df['ATD_datetime'] - df['STD_datetime']).dt.total_seconds() / 60

    critical_df, graph = enhanced_critical_flights_analysis(df)
    print("--- Critical Flights Analysis Complete ---")
if __name__ == '__main__':
    find_critical_flights()