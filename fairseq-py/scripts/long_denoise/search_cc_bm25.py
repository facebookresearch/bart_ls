import json
import sys
import lucene
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import random

from multiprocessing import Pool as ProcessPool
from multiprocessing.util import Finalize
import os

from java.io import File
from java.nio.file import Paths
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field, FieldType
from org.apache.lucene.store import MMapDirectory, ByteBuffersDirectory, RAMDirectory
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.search.similarities import BM25Similarity
from org.apache.lucene.index import \
    FieldInfo, IndexWriter, IndexWriterConfig, IndexOptions




def reduce_query(query, max_len = 48):
    query_len = len(query.split())
    if query_len < max_len:
        return query
    else:
        start = random.randint(0, query_len - max_len)
        query_tokens = query.split()
        return " ".join(query_tokens[start:start + max_len])



class Indexer(object):

    def __init__(self, corpusPath, storeDir=None, in_memory=False):

        if not in_memory and not os.path.exists(storeDir):
            os.mkdir(storeDir)

        if in_memory:
            # self.store = ByteBuffersDirectory()
            self.store = RAMDirectory()
        else:
            self.store = MMapDirectory(Paths.get(storeDir))
        self.analyzer = StandardAnalyzer()
        config = IndexWriterConfig(self.analyzer)
        config.setOpenMode(IndexWriterConfig.OpenMode.CREATE)
        writer = IndexWriter(self.store, config)

        self.indexDocs(corpusPath, writer)
        writer.commit()
        writer.close()
        print('done')

    def indexDocs(self, corpusPath, writer):

        metaType = FieldType()
        metaType.setStored(True)
        metaType.setTokenized(False)

        contextType = FieldType()
        contextType.setStored(True)
        contextType.setTokenized(True)
        contextType.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS)

        # adding corpus
        with open(corpusPath) as g:
            for row in tqdm(g.readlines()):
                row = json.loads(row)
                doc_id, text = row['id'], row['contents']

                # doc_id, text, title = row[:3]
                doc = Document()
                # doc.add(Field('Title', title, metaType))
                doc.add(Field('ID', str(doc_id), metaType))
                doc.add(Field('Context', text, contextType))
                writer.addDocument(doc)

        return 

# def bm25_index_in_cluster(cluster_id, cluster_dir='/data/home/xwhan/data/long_c4/clusters'):
#     """
#     build a bm25 index in each cluster
#     """

#     index_path = f'{cluster_dir}/cluster_{cluster_id}/bm25'
#     print(f'Index all documents in cluster {cluster_id} to {index_path}...')

#     lucene.initVM(vmargs=['-Djava.awt.headless=true'])
#     index = Indexer(index_path)

#     for shard in range(10):
#         print(f'Indexing shard {shard}')
#         start = datetime.now()
#         document_path = f'{cluster_dir}/cluster_{cluster_id}/documents_{shard}.jsonl'
#         index.indexDocs(document_path)
#         end = datetime.now()
#         print(end - start)

#         searchDir = MMapDirectory(Paths.get(index_path), MMapDirectory.DEFAULT_MAX_CHUNK_SIZE)
#         searcher = IndexSearcher(DirectoryReader.open(searchDir))
#         reader = searcher.getIndexReader()
#         print(reader.numDocs())

#     return cluster_id

# def bm25_search_in_cluster(cluster_id, cluster_dir='/data/home/xwhan/data/long_c4/clusters'):
#     """
#     create long documents with bm25
#     """
#     lucene.initVM(vmargs=['-Djava.awt.headless=true'])
#     index_path = f'{cluster_dir}/cluster_{cluster_id}/bm25'

#     searchDir = MMapDirectory(Paths.get(index_path), MMapDirectory.DEFAULT_MAX_CHUNK_SIZE)
#     searcher = IndexSearcher(DirectoryReader.open(searchDir))
#     analyzer = StandardAnalyzer()
    
#     print(f'Read all documents in cluster {cluster_id}...')
#     id2doc = {}
#     for shard in range(10):
#         documents = [json.loads(l) for l in tqdm(open(f'{cluster_dir}/cluster_{cluster_id}/documents_{shard}.jsonl').readlines())]
#         for doc in documents:
#             id2doc[doc['id']] = doc['raw']

#     parser = QueryParser('Context', analyzer)

#     print(f"Start assembling long docs...")
#     seen_docs = defaultdict(int)
#     long_docs = [] # each line is a list of docs
#     doc_lens = []


#     max_search_steps = 50
#     for doc_id in tqdm(id2doc.keys()):

#         if doc_id in seen_docs:
#             continue

#         curr_docs = set()

#         seen_docs[doc_id] += 1
#         curr_docs.add(doc_id)

#         text = id2doc[doc_id]
#         doc_len = len(text.split())

#         if doc_len > 10000:
#             long_docs.append([text.strip()])
#         else:
#             # recursive research
#             doc_list = [text]
#             for step in range(max_search_steps):
#                 query_raw = ' '.join(doc_list)
#                 query = parser.parse(QueryParser.escape(query_raw))
#                 scoreDocs = searcher.search(query, 2*max_search_steps).scoreDocs
#                 success = False
#                 for hit in scoreDocs:
#                     item = searcher.doc(hit.doc)
#                     knn_id = item.get("ID")
#                     if knn_id in curr_docs:
#                         continue

#                     success = True
#                     knn_doc = item.get('Context')
#                     seen_docs[knn_id] += 1
#                     curr_docs.add(knn_id)
#                     doc_list.append(knn_doc)
#                     doc_len += len(knn_doc.split())
#                     break
                
#                 breakpoint()
#                 if doc_len > 16000:
#                     break
                
                
#     #         knn_docs = knn_map[doc_id]
#     #         doc_list = [id2doc[doc_id].strip()]
#     #         for knn_id in knn_docs:
#     #             if seen_docs[knn_id] >= 3: # max repeition
#     #                 continue
#     #             seen_docs[knn_id] += 1
#     #             knn_doc = id2doc[knn_id]
#     #             doc_list.append(knn_doc.strip())
#     #             doc_len += len(knn_doc.split())
#     #             if doc_len > 16000:
#     #                 break
#     #         long_docs.append(doc_list)
#     #     doc_lens.append(doc_len)

#     # freqs = np.array(list(seen_docs.values()))
#     # doc_lens = np.array(doc_lens)

#     # pass


# from java.nio.file import Paths
# from org.apache.lucene.index import DirectoryReader
# from org.apache.lucene.analysis.standard import StandardAnalyzer
# from org.apache.lucene.document import Document, Field, FieldType
# from org.apache.lucene.index import FieldInfo, IndexWriter, IndexWriterConfig, IndexOptions
# from org.apache.lucene.search import IndexSearcher
# from org.apache.lucene.store import MMapDirectory
# from org.apache.lucene.queryparser.classic import QueryParser
# from scripts.long_denoise.make_c4_longer import _score_ngrams

# class Ticker(object):

#     def __init__(self):
#         self.tick = True

#     def run(self):
#         while self.tick:
#             sys.stdout.write('.')
#             sys.stdout.flush()
#             time.sleep(1.0)


# class Indexer(object):

#     def __init__(self, storeDir):

#         if not os.path.exists(storeDir):
#             os.mkdir(storeDir)

#         self.store = MMapDirectory(Paths.get(storeDir))
#         self.analyzer = StandardAnalyzer()

#     def indexDocs(self, corpusPath):

#         config = IndexWriterConfig(self.analyzer)
#         config.setOpenMode(IndexWriterConfig.OpenMode.CREATE_OR_APPEND)
#         writer = IndexWriter(self.store, config)
#         metaType = FieldType()
#         metaType.setStored(True)
#         metaType.setTokenized(False)
#         # metaType.setIndexOptions(IndexOptions.DOCS_AND_FREQS)

#         contextType = FieldType()
#         contextType.setStored(True)
#         contextType.setTokenized(True)
#         contextType.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS)

#         # adding corpus
#         with open(corpusPath) as g:
#             for row in tqdm(g.readlines()):
#                 row = json.loads(row)
#                 doc_id, text = row['id'], row['raw']

#                 # doc_id, text, title = row[:3]
#                 doc = Document()
#                 # doc.add(Field('Title', title, metaType))
#                 doc.add(Field('ID', str(doc_id), metaType))
#                 doc.add(Field('Context', text, contextType))
#                 writer.addDocument(doc)

#         ticker = Ticker()
#         print('commit index')
#         threading.Thread(target=ticker.run).start()
#         writer.commit()
#         writer.close()
#         ticker.tick = False
#         print('done')
    
#         return 


if __name__ == '__main__':

    shard_id = int(sys.argv[1])

    lucene.initVM(vmargs=['-Djava.aws.headless=true'])

    index_path = f'/data/home/xwhan/data/long_c4/indexes/shard{shard_id}'
    corpus_path = f'/data/home/xwhan/data/long_c4/jsonl_files/shard{shard_id}/sample.jsonl'

    print("Build Lucene Index ...")
    index = Indexer(corpus_path, in_memory=False, storeDir=index_path)

    # searchDir = MMapDirectory(Paths.get(index_path), MMapDirectory.DEFAULT_MAX_CHUNK_SIZE)

    analyzer = index.analyzer

    # searcher = IndexSearcher(DirectoryReader.open(searchDir))

    searcher = IndexSearcher(DirectoryReader.open(index.store))


    print('Loading documents...')
    corpus_path = f'/data/home/xwhan/data/long_c4/jsonl_files/shard{shard_id}/sample.jsonl'
    id2docs = {str(json.loads(line)['id']):json.loads(line)['contents'] for line in tqdm(open(corpus_path).readlines())}

    parser = QueryParser('Context', analyzer)


    print('Searching documents...')
    for doc_id in tqdm(list(id2docs.keys())):

        q = reduce_query(id2docs[doc_id])
        query = parser.parse(QueryParser.escape(q))

        scoreDocs = searcher.search(query, 10).scoreDocs
        topkDocs = []
        for hit in scoreDocs:
            doc = searcher.doc(hit.doc)
            topkDocs.append({
                "text": doc.get("Context")
            })
        # breakpoint()
    #     retrieved.append(topkDocs)
    
    # qas_docs = list(zip(questions, answers, retrieved))


