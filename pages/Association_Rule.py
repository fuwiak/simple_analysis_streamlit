import pandas as pd
import streamlit as st
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.colors as mcolors

st.title("Advanced Association Rule Mining Dashboard")

@st.experimental_singleton
def load_data(path):
    """ Load data selectively with memory optimization. """
    return pd.read_csv(path)

@st.experimental_singleton
def preprocess_data(df, top_n):
    """ Convert all data to string to prevent type errors and prepare transactions. """
    # Select top n columns with the most unique values
    col_unique_counts = df.nunique().sort_values(ascending=False)
    top_columns = col_unique_counts.head(top_n).index.tolist()
    df = df[top_columns].applymap(str)
    transactions = df.apply(lambda row: row.dropna().tolist(), axis=1).tolist()
    encoder = TransactionEncoder()
    encoded_array = encoder.fit(transactions).transform(transactions)
    return pd.DataFrame(encoded_array, columns=encoder.columns_)

@st.experimental_memo
def compute_rules(encoded_df, min_support, min_confidence):
    """ Compute frequent itemsets and association rules based on user-defined thresholds. """
    frequent_itemsets = apriori(encoded_df, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
    return rules

# Sidebar - Dataset selection
st.sidebar.header("Dataset Selection")
file_paths = {
    'Air Rates': 'data/rates_air.csv',
    'Land Rates': 'data/rates_land.csv',
    'Sea Rates': 'data/rates_sea.csv',
    'Merged Leads (default)': 'data/merged_leads_land_not_null.csv'
}
selected_file = st.sidebar.selectbox("Choose a dataset:", options=list(file_paths.keys()), index=3)
path = file_paths[selected_file]

# Load data with optimizations
df = load_data(path)

# User input for number of top columns to consider
top_n = st.sidebar.slider("Number of top columns to consider", 1, len(df.columns), 5)

# Data preprocessing and association rule mining
encoded_df = preprocess_data(df, top_n)
min_support = st.sidebar.slider("Minimum Support", 0.01, 0.5, 0.05, 0.01)
min_confidence = st.sidebar.slider("Minimum Confidence", 0.0, 1.0, 0.5, 0.1)
rules = compute_rules(encoded_df, min_support, min_confidence)

# Displaying results
if not rules.empty:
    st.write("Generated Association Rules:")
    st.dataframe(rules)

    # Visualization: Network graph visualization
    if st.button("Show Network Graph"):
        st.write("Network Graph of Item Associations:")
        G = nx.from_pandas_edgelist(rules, 'antecedents', 'consequents', edge_attr=True)
        pos = nx.spring_layout(G, seed=42)
        plt.figure(figsize=(10, 10))
        norm = plt.Normalize(vmin=rules['lift'].min(), vmax=rules['lift'].max())
        cmap = plt.cm.Blues
        colors = [cmap(norm(value)) for value in rules['lift']]
        nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2500, edge_color=colors, edge_cmap=cmap, width=2.0)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, label='Lift')
        plt.title('Item Association Graph')
        st.pyplot(plt)
else:
    st.write("No rules found with the current settings.")
