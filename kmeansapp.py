import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from sklearn.cluster import KMeans

df = pd.read_csv('Mall_Customers.csv')
df.rename(index=str,columns={
    'Annual Income (k$)' : 'Income' , 'Spending Score (1-100)' : 'Score'
}, inplace=True)
x = df.iloc[:,[2,3,4]].values

st.header("Isi Dari Dataset")
st.write(df)

clusters = []
for i in range(1,11):
    km = KMeans(n_clusters=i)
    km.fit(x)
    clusters.append(km.inertia_)

fig, ax = plt.subplots(figsize=(12,8))
sns.lineplot(x=list(range(1,11)),y=clusters,ax=ax)
ax.set_title('Mencari Elbow')
ax.set_xlabel('Clusters')
ax.set_ylabel('Inertia')

ax.annotate('Posibble Elbow Point', xy=(4,100000), xytext=(2,80000), xycoords='data',
            arrowprops=dict(arrowstyle='->',connectionstyle='arc3',color='blue',lw=2))
ax.annotate('Posibble Elbow Point', xy=(5,75000), xytext=(5,100000), xycoords='data',
            arrowprops=dict(arrowstyle='->',connectionstyle='arc3',color='blue',lw=2))

st.set_option('deprecation.showPyplotGlobalUse', False)
elbowplot = st.pyplot()

st.sidebar.subheader("Nilai Jumlah K")
clust = st.sidebar.slider("Pilih Jumlah Cluster:",2, 10, 5, 1)

def k_means(n_clust):
    kmean =KMeans(n_clusters=n_clust, init = "k-means++", max_iter = 300, n_init = 10, random_state = 0)
    y_clusters = kmean.fit_predict(x)

    fig = plt.figure(figsize = (15,15))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x[y_clusters == 0,0],x[y_clusters == 0,1],x[y_clusters == 0,2], s = 40 , color = 'blue', label = "cluster 0")
    ax.scatter(x[y_clusters == 1,0],x[y_clusters == 1,1],x[y_clusters == 1,2], s = 40 , color = 'orange', label = "cluster 1")
    ax.scatter(x[y_clusters == 2,0],x[y_clusters == 2,1],x[y_clusters == 2,2], s = 40 , color = 'green', label = "cluster 2")
    ax.scatter(x[y_clusters == 3,0],x[y_clusters == 3,1],x[y_clusters == 3,2], s = 40 , color = '#D12B60', label = "cluster 3")
    ax.scatter(x[y_clusters == 4,0],x[y_clusters == 4,1],x[y_clusters == 4,2], s = 40 , color = 'purple', label = "cluster 4")
    ax.set_xlabel('Age of a customer-->')
    ax.set_ylabel('Anual Income-->')
    ax.set_zlabel('Spending Score-->')
    ax.legend()
    plt.show()

    st.header('Cluster Plot')
    st.pyplot()

    Scene = dict(xaxis = dict(title  = 'Age -->'),yaxis = dict(title  = 'Spending Score--->'),zaxis = dict(title  = 'Annual Income-->'))

    labels = kmean.labels_
    trace = go.Scatter3d(x=x[:, 0], y=x[:, 1], z=x[:, 2], mode='markers',marker=dict(color = labels, size= 10, line=dict(color= 'black',width = 10)))
    layout = go.Layout(margin=dict(l=0,r=0),scene = Scene,height = 800,width = 800)
    data = [trace]
    fig = go.Figure(data = data, layout = layout)
    fig.show()

    st.header('Cluster Plot 2')
    st.plotly_chart(fig, use_container_width=True)

    clusteringdata = pd.DataFrame(x, columns=['Age', 'Income', 'Score'])
    clusteringdata['Labels']=kmean.labels_

    plt.figure(figsize=(10,8))
    sns.scatterplot(x='Income', y='Score', hue='Labels', marker='o', size='Labels', palette=sns.color_palette('hls', n_colors=n_clust), data=clusteringdata)
    
    for label in clusteringdata['Labels'].unique():
        plt.annotate(label,
                (clusteringdata[clusteringdata['Labels']==label]['Income'].mean(),
                clusteringdata[clusteringdata['Labels']==label]['Score'].mean()),
                horizontalalignment='center',
                verticalalignment='center',
                size=20, weight='bold',
                color='black')
    st.header('Cluster Plot 3')
    st.pyplot()
    st.header('Hasil Final Data Clustering')
    st.write(clusteringdata)


k_means(clust)

