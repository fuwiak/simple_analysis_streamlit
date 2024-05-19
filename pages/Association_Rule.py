import pandas as pd
import streamlit as st
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from streamlit_agraph import agraph, Node, Edge, Config
import networkx as nx

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
def compute_rules(encoded_df, min_support, min_confidence, max_rules):
    """ Compute frequent itemsets and association rules based on user-defined thresholds. """
    frequent_itemsets = apriori(encoded_df, min_support=min_support,
                                use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence",
                              min_threshold=min_confidence)
    rules['antecedents'] = rules['antecedents'].apply(
        lambda x: ', '.join(list(x)))
    rules['consequents'] = rules['consequents'].apply(
        lambda x: ', '.join(list(x)))

    # Limit the number of rules
    rules = rules.head(max_rules)

    return rules


def draw_interactive_graph(rules, level_separation, node_spacing,
                           tree_spacing):
    """ Draw an interactive graph using streamlit-agraph. """
    G = nx.DiGraph()

    for index, row in rules.iterrows():
        antecedents = row['antecedents']
        consequents = row['consequents']
        lift = row['lift']
        confidence = row['confidence']
        G.add_edge(antecedents, consequents, lift=lift, confidence=confidence)

    nodes = []
    edges = []

    for node in G.nodes:
        nodes.append(
            Node(id=node, label=node, size=25, shape="ellipse", title=node))

    for edge in G.edges:
        source, target = edge
        lift = G[source][target]['lift']
        confidence = G[source][target]['confidence']
        edges.append(Edge(source=source,
                          label=f"Lift: {lift:.2f}, Confidence: {confidence:.2f}",
                          target=target))

    config = Config(
        width=950,
        height=700,
        directed=True,
        physics=False,
        hierarchical=True,
        layout={"hierarchical": {"enabled": True,
                                 "levelSeparation": level_separation,
                                 "nodeSpacing": node_spacing,
                                 "treeSpacing": tree_spacing,
                                 "blockShifting": True,
                                 "edgeMinimization": True,
                                 "parentCentralization": True,
                                 "direction": "LR", "sortMethod": "hubsize"}},
        interaction={"hover": True},
        edges={
            "smooth": {"type": "cubicBezier", "forceDirection": "horizontal",
                       "roundness": 0.5}}
    )

    return agraph(nodes=nodes, edges=edges, config=config)


# Sidebar - Dataset selection
st.sidebar.header("Dataset Selection")
file_paths = {
    'Air Rates': 'data/rates_air.csv',
    'Land Rates': 'data/rates_land.csv',
    'Sea Rates': 'data/rates_sea.csv',
    'Merged Leads (default)': 'data/merged_leads_land_not_null.csv'
}
selected_file = st.sidebar.selectbox("Choose a dataset:",
                                     options=list(file_paths.keys()), index=3)
path = file_paths[selected_file]

# Load data with optimizations
df = load_data(path)

# User input for number of top columns to consider
top_n = st.sidebar.slider("Number of top columns to consider", 1,
                          len(df.columns), 5)

# User input for graph layout settings
level_separation = st.sidebar.slider("Level Separation", 100, 1000, 300, 50)
node_spacing = st.sidebar.slider("Node Spacing", 50, 500, 200, 50)
tree_spacing = st.sidebar.slider("Tree Spacing", 100, 1000, 300, 50)

# User input for rule filtering
max_rules = st.sidebar.slider("Max Number of Rules", 10, 100, 50, 10)

# Data preprocessing and association rule mining
encoded_df = preprocess_data(df, top_n)
min_support = st.sidebar.slider("Minimum Support", 0.01, 0.5, 0.05, 0.01)
min_confidence = st.sidebar.slider("Minimum Confidence", 0.0, 1.0, 0.5, 0.1)
rules = compute_rules(encoded_df, min_support, min_confidence, max_rules)

# Displaying results
if not rules.empty:
    st.write("Generated Association Rules:")
    st.dataframe(rules)

    # Visualization: Interactive Network graph visualization
    if st.button("Show Interactive Network Graph"):
        st.write("Interactive Network Graph of Item Associations:")
        draw_interactive_graph(rules, level_separation, node_spacing,
                               tree_spacing)
else:
    st.write("No rules found with the current settings.")
