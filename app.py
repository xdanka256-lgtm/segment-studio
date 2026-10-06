import numpy as np
import streamlit as st
import pandas as pd
from altair.datasets import data
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from ollama import Client
import requests
from sklearn.metrics import silhouette_score


st.title("Segment Studio")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Step 1: CSV Table")
    st.dataframe(df)

    st.subheader("Step 2: WCSS (Elbow)")

    numeric_df = df.select_dtypes(include="number")

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)

    min_k = st.slider("Select Min K-Means",2,20,2)
    max_k = st.slider("Select Max K-Means",2,20,10)

    if st.button("Run WCSS"):
        k_values = list(range(min_k, max_k + 1))
        wcss = []
        silhouette_scores = []

        for k in range(min_k, max_k + 1):
            model = KMeans(n_clusters=k, random_state=42)
            model.fit(scaled_data)
            wcss.append(model.inertia_)
            label = model.labels_
            if len(set(label)) > 1:
                score = silhouette_score(scaled_data, label)
            else:
                score = None
            silhouette_scores.append(score)

        results = pd.DataFrame({
            "k":range(min_k, max_k + 1),
             "WCSS":wcss,
            "silhouette":silhouette_scores
         })
        st.dataframe(results)
        valid_results = results.dropna(subset=["silhouette"])
        if not valid_results.empty:
            best_k = int(
                valid_results.loc[
                    valid_results["silhouette"].idxmax(),
                    "k"
                ]
            )
            st.write("Best K by Silhouette Score:",best_k)

        fig, ax = plt.subplots()
        ax.plot(results["k"], results["WCSS"], marker="o")
        ax.set_xlabel("k")
        ax.set_ylabel("WCSS")
        ax.set_title("Elbow Plot")

        st.pyplot(fig)

        fig2, ax2 = plt.subplots()
        ax2.plot(
            results["k"],
            results["silhouette"],
            marker="o"
        )
        ax2.set_xlabel("k")
        ax2.set_ylabel("Silhouette Score")
        ax2.set_title("Silhouette Score")

        st.pyplot(fig2)

    st.subheader("Step 3: Clusters")

    final_k = st.slider("Select Final K-Means")

    if st.button("Create Clusters"):
        final_model = KMeans(
            n_clusters=final_k,
            random_state=42
        )

        cluster_ids = final_model.fit_predict(scaled_data)

        df['cluster_id'] = cluster_ids

        cluster_table = (
            df.groupby('cluster_id')
            .size()
            .reset_index(name='count')
        )
        cluster_table['name'] = ""
        cluster_table['description'] = ""

        st.dataframe(cluster_table)

        st.subheader("Step 4: Cluster Summaries")

        numeric_columns = [
            col for col in df.select_dtypes(include="number").columns
            if col != "cluster_id"
        ]

        categorical_columns = [
            col for col in df.select_dtypes(include="object").columns
            if col != "cluster_id"
        ]
        cluster_summaries = {}

        for cluster_id in sorted(df["cluster_id"].unique()):
            cluster_data = df[df["cluster_id"] == cluster_id]

            summary = {
                "count": len(cluster_data),
                "numeric_means":{},
                "categorical_modes":{},
            }
            for column in numeric_columns:
                summary["numeric_means"][column] = round(cluster_data[column].mean(), 2)

            for column in categorical_columns:
                mode_value = cluster_data[column].mode()

                if not mode_value.empty:
                    summary["categorical_modes"][column] = mode_value.iloc[0]

            cluster_summaries[cluster_id] = summary

        for cluster_id, summary in cluster_summaries.items():
            st.write(f"### Cluster {cluster_id}")
            st.write("Count:", summary["count"])
            st.write("Numeric Averages:")
            st.json(summary["numeric_means"])
            st.write("Most comon categorical values:")
            st.json(summary["categorical_modes"])

        for cluster_id, summary in cluster_summaries.items():
            prompt =f"""
            You are analyzing a customer cluster
            Cluster ID: {cluster_id}
            Count: {summary["count"]}
            Numeric Means: {summary["numeric_means"]}
            Most comon categorical values: {summary["categorical_modes"]}
            Give this cluster:
            1. A short name.
            2. A one-sentence description.
            Return exactly in this format:
            Name:... 
            Description:...
            """

            response = requests.post(
                "http://127.0.0.1:11434/api/generate",
                json={
                    "model": "llama3.2",
                    "prompt": prompt,
                    "stream": False
                }

            )
            result = response.json()

            llm_text = result["response"]
            name = ""
            description = ""

            for line in llm_text.splitlines():
                if line.startswith("Name:"):
                    name = line.replace("Name:", "").strip()

                if line.startswith("Description:"):
                    description = line.replace("Description:", "").strip()

            cluster_table.loc[
                cluster_table["cluster_id"] == cluster_id,
                "name"
            ] = name
            cluster_table.loc[
                cluster_table["cluster_id"] == cluster_id,
                "description"
            ] = description

        st.subheader("Named Clusters")
        st.dataframe(cluster_table)

        cluster_new_map = dict(
            zip(
                cluster_table["name"],
                cluster_table["description"]
            )
        )
        df[f"name_cluster"] = df["cluster_id"].map(cluster_new_map)

        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="clusters.csv",
            mime = "text/csv"
        )

