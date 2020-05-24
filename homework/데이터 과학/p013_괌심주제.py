from collections import Counter

interests = [
    (0, "Hadoop"),(0, "Big Data"),(0, "HBase"),(0, "Java"),
    (0, "Spark"),(0, "Storm"),(0, "Cassandra"),
    (1, "NoSQL"),(1, "MongoDB"),(1, "Cassandra"),(1, "HBase"),
    (1, "Postgres"),(2, "Python"),(2, "scikit-learn"),(2, "scipy"),
    (2, "numpy"),(2, "statsmodels"),(2, "pandas"),(3, "R"),(3, "Python"),
    (3, "statistics"),(3, "regression"),(3, "probability"),
    (4, "machine learning"),(4, "regression"),(4, "decision trees"),
    (4, "libsvm"),(5, "Python"),(5, "R"),(5, "Java"),(5, "C++"),
    (5, "Haskell"),(5, "programming languages"),(6, "statistics"),
    (6, "probability"),(6, "mathematics"),(6, "theory"),
    (7, "machine learing"),(7, "scikit-learn"),(7, "Mahout"),
    (7, "nerual network"),(8, "nerual network"),(8, "deep learning"),
    (8, "Big Data"),(8, "artificial intelligence"),(9, "Hadoop"),
    (9, "Java"),(9, "MapReduce"),(9, "Big Data"),
]


words_and_counts = Counter(word
                           for user, interest in interests
                           for word in interest.lower().split())
print(words_and_counts)
# Counter({'big': 3, 'data': 3, 'java': 3, 'python': 3, 'hadoop': 2, 
#          'hbase': 2, 'cassandra': 2, 'scikit-learn': 2, 'r': 2, 'statistics': 2, 
#          'regression': 2, 'probability': 2, 'machine': 2, 'learning': 2, 'nerual': 2, 
#          'network': 2, 'spark': 1, 'storm': 1, 'nosql': 1, 'mongodb': 1, 
#          'postgres': 1, 'scipy': 1, 'numpy': 1, 'statsmodels': 1, 'pandas': 1, 
#          'decision': 1, 'trees': 1, 'libsvm': 1, 'c++': 1, 'haskell': 1, 
#          'programming': 1, 'languages': 1, 'mathematics': 1, 'theory': 1, 
#          'learing': 1, 'mahout': 1, 'deep': 1, 'artificial': 1, 'intelligence': 1, 
#          'mapreduce': 1})


# 한 번을 초과해서 등장하는 단어들만 출력
for word, count in words_and_counts.most_common():
    if count > 1:
        print(word, count)
        # big 3, data 3, java 3, python 3, hadoop 2, hbase 2, cassandra 2
        # scikit-learn 2, r 2, statistics 2, regression 2, probability 2
        # machine 2, learning 2 ,nerual 2, network 2