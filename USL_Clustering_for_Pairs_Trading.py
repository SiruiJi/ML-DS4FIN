# Load libraries
import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv, set_option
from pandas.plotting import scatter_matrix
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import datetime
import pandas_datareader as dr

#Import Model Packages
from sklearn.cluster import KMeans, AgglomerativeClustering,AffinityPropagation, DBSCAN
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_mutual_info_score
from sklearn import cluster, covariance, manifold
from scipy.cluster.hierarchy import dendrogram, linkage, ward
from statsmodels.tsa.stattools import coint
from sklearn.manifold import TSNE
import matplotlib.cm as cm

#Other Helper Packages and functions
import matplotlib.ticker as ticker
from itertools import cycle
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def data_preprocessing(ticker_set, start_date, end_date):
    end_analysis_date = (start_date + timedelta(days=15)).strftime("%Y-%m-%d")
    end_analysis_date = datetime.strptime(end_analysis_date, "%Y-%m-%d")

    dataset = yf.download(ticker_set, start_date, end_date)['Adj Close']
    print('Null Values =', dataset.isnull().values.any())
    missing_fractions = dataset.isnull().mean().sort_values(ascending=False)

    missing_fractions.head(10)

    drop_list = sorted(list(missing_fractions[missing_fractions > 0.3].index))

    dataset.drop(labels=drop_list, axis=1, inplace=True)
    print(dataset.shape)
    dataset = dataset.fillna(method='ffill')

    returns = dataset.pct_change().mean() * 252
    returns = pd.DataFrame(returns)
    returns.columns = ['Returns']
    returns['Volatility'] = dataset.pct_change().std() * np.sqrt(252)
    data = returns

    scaler = StandardScaler().fit(data)
    rescaledDataset = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)
    # summarize transformed data
    rescaledDataset.head(2)
    X = rescaledDataset
    return end_analysis_date, dataset, X


def find_optimal_number_of_cluster_for_kmeans(X):
    distorsions = []
    max_loop = 20
    for k in range(2, max_loop):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        distorsions.append(kmeans.inertia_)
    fig = plt.figure(figsize=(15, 5))
    plt.plot(range(2, max_loop), distorsions)
    plt.xticks([i for i in range(2, max_loop)], rotation=75)
    plt.grid(True)
    plt.show()

    silhouette_score = []
    for k in range(2, max_loop):
        kmeans = KMeans(n_clusters=k, random_state=10, n_init=10)
        kmeans.fit(X)
        silhouette_score.append(metrics.silhouette_score(X, kmeans.labels_, random_state=10))
    fig = plt.figure(figsize=(15, 5))
    plt.plot(range(2, max_loop), silhouette_score)
    plt.xticks([i for i in range(2, max_loop)], rotation=75)
    plt.grid(True)
    plt.show()


def k_means_training(X, km_nclust):
    k_means = cluster.KMeans(n_clusters=km_nclust)
    k_means.fit(X)
    target_labels = k_means.predict(X)
    centroids = k_means.cluster_centers_
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111)
    scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=k_means.labels_, cmap="rainbow", label=X.index)
    ax.set_title('k-Means results')
    ax.set_xlabel('Mean Return')
    ax.set_ylabel('Volatility')
    plt.colorbar(scatter)

    plt.plot(centroids[:, 0], centroids[:, 1], 'sg', markersize=11)

    # show number of stocks in each cluster
    clustered_series = pd.Series(index=X.index, data=k_means.labels_.flatten())
    # clustered stock with its cluster label
    clustered_series_all = pd.Series(index=X.index, data=k_means.labels_.flatten())
    clustered_series = clustered_series[clustered_series != -1]

    plt.figure(figsize=(12, 7))
    plt.barh(
        range(len(clustered_series.value_counts())),  # cluster labels, y axis
        clustered_series.value_counts()
    )
    plt.title('Cluster Member Counts')
    plt.xlabel('Stocks in Cluster')
    plt.ylabel('Cluster Number')
    plt.show()
    return clustered_series, k_means, centroids


def find_optimal_number_of_cluster_for_hc(X):
    # Calulate linkage
    Z = linkage(X, method='ward')
    print(Z[0])
    # Plot Dendogram
    plt.figure(figsize=(10, 7))
    plt.title("Stocks Dendrograms")
    dendrogram(Z, labels=X.index)
    plt.show()
    distance_threshold = 13
    clusters = fcluster(Z, distance_threshold, criterion='distance')
    chosen_clusters = pd.DataFrame(data=clusters, columns=['cluster'])
    print(chosen_clusters['cluster'].unique())


def hierarchical_clustering_training(X, hc_nclust):
    hc = AgglomerativeClustering(n_clusters=hc_nclust, affinity='euclidean', linkage='ward')
    clust_labels1 = hc.fit_predict(X)
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111)
    scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clust_labels1, cmap="rainbow")
    ax.set_title('Hierarchical Clustering')
    ax.set_xlabel('Mean Return')
    ax.set_ylabel('Volatility')
    plt.colorbar(scatter)
    plt.show()
    return hc

def affinity_propagation_training(X, clustered_series):
    ap = AffinityPropagation()
    ap.fit(X)
    clust_labels2 = ap.predict(X)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clust_labels2, cmap="rainbow")
    ax.set_title('Affinity')
    ax.set_xlabel('Mean Return')
    ax.set_ylabel('Volatility')
    plt.colorbar(scatter)

    cluster_centers_indices = ap.cluster_centers_indices_
    labels = ap.labels_

    no_clusters = len(cluster_centers_indices)
    print('Estimated number of clusters: %d' % no_clusters)
    # Plot exemplars

    X_temp = np.asarray(X)
    plt.close('all')
    plt.figure(1)
    plt.clf()

    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(no_clusters), colors):
        class_members = labels == k
        cluster_center = X_temp[cluster_centers_indices[k]]
        plt.plot(X_temp[class_members, 0], X_temp[class_members, 1], col + '.')
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=14)
        for x in X_temp[class_members]:
            plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

    plt.show()

    # show number of stocks in each cluster
    clustered_series_ap = pd.Series(index=X.index, data=ap.labels_.flatten())
    # clustered stock with its cluster label
    clustered_series_all_ap = pd.Series(index=X.index, data=ap.labels_.flatten())
    clustered_series_ap = clustered_series_ap[clustered_series != -1]

    plt.figure(figsize=(12, 7))
    plt.barh(
        range(len(clustered_series_ap.value_counts())),  # cluster labels, y axis
        clustered_series_ap.value_counts()
    )
    plt.title('Cluster Member Counts')
    plt.xlabel('Stocks in Cluster')
    plt.ylabel('Cluster Number')
    plt.show()
    return ap, clustered_series_ap


def model_evaluation(X, k_means, hc, ap):
    print("km", metrics.silhouette_score(X, k_means.labels_, metric='euclidean'))
    print("hc", metrics.silhouette_score(X, hc.fit_predict(X), metric='euclidean'))
    print("ap", metrics.silhouette_score(X, ap.labels_, metric='euclidean'))


def visualizing_time_series_within_a_cluster(X, dataset, ap, clustered_series_ap):
    # all stock with its cluster label (including -1)
    global data
    clustered_series = pd.Series(index=X.index, data=ap.fit_predict(X).flatten())
    # clustered stock with its cluster label
    clustered_series_all = pd.Series(index=X.index, data=ap.fit_predict(X).flatten())
    clustered_series = clustered_series[clustered_series != -1]
    # get the number of stocks in each cluster
    counts = clustered_series_ap.value_counts()

    # let's visualize some clusters
    cluster_vis_list = list(counts[(counts < 25) & (counts > 1)].index)[::-1]
    print(cluster_vis_list)

    CLUSTER_SIZE_LIMIT = 9999
    counts = clustered_series.value_counts()
    ticker_count_reduced = counts[(counts > 1) & (counts <= CLUSTER_SIZE_LIMIT)]
    print("Clusters formed: %d" % len(ticker_count_reduced))
    print("Pairs to evaluate: %d" % (ticker_count_reduced * (ticker_count_reduced - 1)).sum())
    # plot a handful of the smallest clusters

    print(cluster_vis_list[0:min(len(cluster_vis_list), 4)])
    for clust in cluster_vis_list[0:min(len(cluster_vis_list), 4)]:
        tickers = list(clustered_series[clustered_series == clust].index)
        means = np.log(dataset.loc[:end_analysis_date].mean())
        data = np.log(dataset.loc[:end_analysis_date]).sub(means)
        data.plot(title='Stock Time Series for Cluster %d' % clust)
    plt.show()
    return ticker_count_reduced, data, clustered_series


def find_cointegrated_pairs(data, significance=0.05):
    # This function is from https://www.quantopian.com/lectures/introduction-to-pairs-trading
    n = data.shape[1]
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    keys = data.keys()
    pairs = []
    for i in range(1):
        for j in range(i+1, n):
            S1 = data[keys[i]]
            S2 = data[keys[j]]
            result = coint(S1, S2)
            score = result[0]
            pvalue = result[1]
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue
            if pvalue < significance:
                pairs.append((keys[i], keys[j]))

    return score_matrix, pvalue_matrix, pairs


def pairs_visualizing(pairs, X, clustered_series, centroids):

    stocks = np.unique(pairs)
    X_df = pd.DataFrame(index=X.index, data=X).T
    in_pairs_series = clustered_series.loc[stocks]
    stocks = list(np.unique(pairs))
    X_pairs = X_df.T.loc[stocks]
    X_tsne = TSNE(learning_rate=50, perplexity=3, random_state=1337).fit_transform(X_pairs)
    plt.figure(1, facecolor='white', figsize=(16, 8))
    plt.clf()
    plt.axis('off')
    for pair in pairs:
        # print(pair[0])
        ticker1 = pair[0]
        loc1 = X_pairs.index.get_loc(pair[0])
        x1, y1 = X_tsne[loc1, :]
        # print(ticker1, loc1)

        ticker2 = pair[0]
        loc2 = X_pairs.index.get_loc(pair[1])
        x2, y2 = X_tsne[loc2, :]

        plt.plot([x1, x2], [y1, y2], 'k-', alpha=0.3, c='gray');

    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=220, alpha=0.9, c=in_pairs_series.values, cmap=cm.Paired)
    plt.title('T-SNE Visualization of Validated Pairs');

    # zip joins x and y coordinates in pairs
    for x, y, name in zip(X_tsne[:, 0], X_tsne[:, 1], X_pairs.index):
        label = name

        plt.annotate(label,  # this is the text
                     (x, y),  # this is the point to label
                     textcoords="offset points",  # how to position the text
                     xytext=(0, 10),  # distance from text to points (x,y)
                     ha='center')  # horizontal alignment can be left, right or center

    plt.plot(centroids[:, 0], centroids[:, 1], 'sg', markersize=11)
    plt.show()

if __name__ == '__main__':
    ticker_set = ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'GOOG', 'META', 'BRK-B', 'TSLA', 'UNH', 'LLY', 'JPM', 'XOM',
                  'JNJ', 'V', 'PG', 'AVGO', 'MA', 'HD', 'CVX', 'MRK', 'ABBV', 'PEP', 'COST', 'ADBE', 'KO', 'CSCO',
                  'WMT', 'TMO', 'MCD', 'PFE', 'CRM', 'BAC', 'ACN', 'CMCSA', 'LIN', 'NFLX', 'ABT', 'ORCL', 'DHR', 'AMD',
                  'WFC', 'DIS', 'TXN', 'PM', 'VZ', 'INTU', 'COP', 'CAT', 'AMGN', 'NEE', 'INTC', 'UNP', 'LOW', 'IBM',
                  'BMY', 'SPGI', 'RTX', 'HON', 'BA', 'UPS', 'GE', 'QCOM', 'AMAT', 'NKE', 'PLD', 'NOW', 'BKNG', 'SBUX',
                  'MS', 'ELV', 'MDT', 'GS', 'DE', 'ADP', 'LMT', 'TJX', 'T', 'BLK', 'ISRG', 'MDLZ', 'GILD', 'MMC', 'AXP',
                  'SYK', 'REGN', 'VRTX', 'ETN', 'LRCX', 'ADI', 'SCHW', 'CVS', 'ZTS', 'CI', 'CB', 'AMT', 'SLB', 'C',
                  'BDX', 'MO', 'PGR', 'TMUS', 'FI', 'SO', 'EOG', 'BSX', 'CME', 'EQIX', 'MU', 'DUK', 'PANW', 'PYPL',
                  'AON', 'SNPS', 'ITW', 'KLAC', 'ATVI', 'ICE', 'APD', 'SHW', 'CDNS', 'CSX', 'NOC', 'CL', 'MPC', 'HUM',
                  'FDX', 'WM', 'MCK', 'TGT', 'ORLY', 'HCA', 'FCX', 'EMR', 'PXD', 'MMM', 'MCO', 'ROP', 'CMG', 'PSX',
                  'MAR', 'PH', 'APH', 'GD', 'USB', 'NXPI', 'AJG', 'NSC', 'PNC', 'VLO',  'F', 'MSI', 'GM', 'TT',
                  'EW', 'CARR', 'AZO', 'ADSK', 'TDG', 'ANET', 'SRE', 'ECL', 'OXY', 'PCAR', 'ADM', 'MNST', 'KMB', 'PSA',
                  'CCI', 'CHTR', 'MCHP', 'MSCI', 'CTAS']

    start_date = datetime(2022, 10, 1)
    end_date = datetime(2023, 10, 1)
    end_analysis_date, dataset_, X_ = data_preprocessing(ticker_set, start_date, end_date)
    #find_optimal_number_of_cluster_for_kmeans(X_)
    km_clust=3
    clustered_series_, k_means_, centroids_ = k_means_training(X_, km_clust)

    #find_optimal_number_of_cluster_for_hc(X_)
    hc_ncluster=2
    hc_ = hierarchical_clustering_training(X_, hc_ncluster)

    ap_, clustered_series_ap_ = affinity_propagation_training(X_, clustered_series_)

    model_evaluation(X_, k_means_, hc_, ap_)
    ticker_count_reduced_, data_, clustered_series_ = visualizing_time_series_within_a_cluster(X_, dataset_, ap_, clustered_series_ap_)
    #score_matrix_, pvalue_matrix_, pairs_ = find_cointegrated_pairs(data_,  significance=0.05)

    cluster_dict = {}
    for i, which_clust in enumerate(ticker_count_reduced_.index):
        tickers = clustered_series_[clustered_series_ == which_clust].index
        score_matrix, pvalue_matrix, pairs = find_cointegrated_pairs(
            dataset_[tickers]
        )
        cluster_dict[which_clust] = {}
        cluster_dict[which_clust]['score_matrix'] = score_matrix
        cluster_dict[which_clust]['pvalue_matrix'] = pvalue_matrix
        cluster_dict[which_clust]['pairs'] = pairs

    pairs = []
    for clust in cluster_dict.keys():
        pairs.extend(cluster_dict[clust]['pairs'])

    print("Number of pairs found : %d" % len(pairs))
    print("In those pairs, there are %d unique tickers." % len(np.unique(pairs)))
    print(pairs)

    pairs_visualizing(pairs, X_, clustered_series_, centroids_)
