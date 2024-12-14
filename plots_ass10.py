import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.io as pio
from matplotlib.backends.backend_pdf import PdfPages

#initialkize kaleido
pio.kaleido.scope.default_width = 1000
pio.kaleido.scope.default_height = 600

#visualisations to pdf
pp = PdfPages('all_plots.pdf')

#use dataframe from hulu_movies.csv
df=pd.read_csv('hulu_movies.csv')
print(df['Streaming Start'].head())

#streaming start to usable formal and delete colomn with index
df['Streaming Start'] = pd.to_datetime(df['Streaming Start'], format="%d.%m.%Y")
df = df.drop('Unnamed: 0', axis=1)

#create bar chart
top_10_movies = df.nlargest(10, 'Tomatometer')['Title'].tolist()
top_10_scores = df.nlargest(10, 'Tomatometer')['Tomatometer'].tolist()

plt.figure(figsize=(10,6))
plt.barh(top_10_movies, top_10_scores)
plt.xlabel('Tomatometer Score')
plt.title('Top 10 Movies with Highest Tomatometer Scores')
pp.savefig()  # Save the current figure to the PDF
plt.close()

#scatter plot
plt.figure(figsize=(10,6))
plt.scatter(df['Tomatometer'], df['Popcornmeter'])
plt.xlabel('Tomatometer Score')
plt.ylabel('Popcornmeter Score')
plt.title('Comparison of Tomatometer and Popcornmeter Scores')
pp.savefig()  # Save the current figure to the PDF
plt.close()

#histogram
plt.figure(figsize=(12,6))

plt.subplot(1, 2, 1)
plt.hist(df['Tomatometer'], bins=10, edgecolor='black')
plt.xlabel('Tomatometer Score')
plt.title('Distribution of Tomatometer Scores')

plt.subplot(1, 2, 2)
plt.hist(df['Popcornmeter'], bins=10, edgecolor='black')
plt.xlabel('Popcornmeter Score')
plt.title('Distribution of Popcornmeter Scores')

plt.tight_layout()
pp.savefig()  # Save the current figure to the PDF
plt.close()

#lineplot
df['Release Month'] = df['Streaming Start'].dt.month
release_counts = df['Release Month'].value_counts().sort_index()

plt.figure(figsize=(10,6))
plt.plot(release_counts.index, release_counts.values)
plt.xticks(range(1, 13))
plt.xlabel('Month')
plt.ylabel('Number of Movies Released')
plt.title('Movies Released Monthly')
plt.grid(True)
pp.savefig()  # Save the current figure to the PDF
plt.close()

#creating data frame with data that i want to use
data = df[['Title', 'Tomatometer', 'Popcornmeter']]
print(data.head())

#initilize kmean object
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(data[['Tomatometer', 'Popcornmeter']])

#add kmeans cluster to dataframe
data.loc[:, 'Cluster'] = kmeans.labels_
data.loc[:, 'Cluster'] = kmeans.labels_
#create scatter plot with plotly

fig = px.scatter(data, x='Tomatometer', y='Popcornmeter', color='Cluster', hover_data=['Title'], title='Movie Clusters')
fig.write_image("clusters_with_plotly.pdf")


#close pdf
pp.close()

