import xml.etree.ElementTree as ET
import numpy as np
from sklearn.model_selection import train_test_split
import random

'''
This file contains classes for dataset processing.
Each temprel_ee is an instance of event relation annotation.
temprel_set is the collection of instances
'''


class temprel_ee:
    def __init__(self, xml_element):
        self.xml_element = xml_element
        self.label = xml_element.attrib['LABEL']
        self.sentdiff = int(xml_element.attrib['SENTDIFF'])
        self.docid = xml_element.attrib['DOCID']
        self.source = xml_element.attrib['SOURCE']
        self.target = xml_element.attrib['TARGET']
        self.data = xml_element.text.strip().split()
        self.token = []
        self.lemma = []
        self.part_of_speech = []
        self.position = []
        self.length = len(self.data)
        self.event_ix = []
        self.verbs = []
        for i,d in enumerate(self.data):
            tmp = d.split('///')
            self.part_of_speech.append(tmp[-2])
            self.position.append(tmp[-1])
            if tmp[-1] == 'E1':
                self.event_ix.append(i)
                self.verbs.append(tmp[0])
            elif tmp[-1] == 'E2':
                self.event_ix.append(i)
                self.verbs.append(tmp[0])
            self.token.append(tmp[0])
            self.lemma.append(tmp[1])


class temprel_set:
    def __init__(self, xmlfname, datasetname="matres", is_train=0):
        self.xmlfname = xmlfname
        self.datasetname = datasetname
        tree = ET.parse(xmlfname)
        root = tree.getroot()
        self.size = len(root)
        self.temprel_ee = []
        self.docid2idx = {}
        self.idx2docid = []
        for e in root:
            temprel = temprel_ee(e)
            self.temprel_ee.append(temprel)
            if self.docid2idx.get(temprel.docid) is None:
                self.docid2idx[temprel.docid] = len(self.idx2docid)
                self.idx2docid.append(temprel.docid)

        if is_train > 0:
            self.temprel_ee, self.dev = train_test_split(self.temprel_ee,
            test_size=0.2,random_state=is_train)
        self.adj_matrix = None
        self.key_to_eid = {}
        self.eid_to_key = []
        self.eid_to_relid = []
        self.trans_pairs = None
        self.eid_to_docidx = []

    # This method builds a adjacency matrix for negative sampling in eventPoincare embedding
    def build_matrix(self):
        if self.adj_matrix is None:
            row = []# source
            col = []# target
            val = []
            trans_pairs = set()
            self.docidx2eid = [set() for i in range(len(self.idx2docid))]
            for i,temprel in enumerate(self.temprel_ee):
                source_key = temprel.docid+temprel.source
                target_key = temprel.docid+temprel.target

                if self.key_to_eid.get(source_key) is None:
                    self.key_to_eid[source_key] = len(self.eid_to_key)
                    self.eid_to_key.append(source_key)
                    self.eid_to_relid.append((i,True)) # [relid,is_source?]
                    self.eid_to_docidx.append(self.docid2idx[temprel.docid])
                    self.docidx2eid[self.docid2idx[temprel.docid]].add(self.key_to_eid[source_key])

                if self.key_to_eid.get(target_key) is None:
                    self.key_to_eid[target_key] = len(self.eid_to_key)
                    self.eid_to_key.append(target_key)
                    self.eid_to_relid.append((i,False)) # [relid,is_source?]
                    self.eid_to_docidx.append(self.docid2idx[temprel.docid])
                    self.docidx2eid[self.docid2idx[temprel.docid]].add(self.key_to_eid[target_key])

                docidx = self.docid2idx[temprel.docid]
                row.append(self.key_to_eid[source_key])
                col.append(self.key_to_eid[target_key])
                val.append(docidx+1)
                row.append(self.key_to_eid[target_key])
                col.append(self.key_to_eid[source_key])
                val.append(docidx+1)

                if temprel.label != 'VAGUE':
                    if temprel.label == 'AFTER':
                        trans_pairs.add((self.key_to_eid[target_key], self.key_to_eid[source_key]))
                    else:
                        trans_pairs.add((self.key_to_eid[source_key], self.key_to_eid[target_key]))

            self.adj_matrix = np.diag(-np.ones(len(self.eid_to_key), dtype=np.int16))
            
            for i in range(len(val)):
                self.adj_matrix[row[i],col[i]] = val[i]
                other_eve_in_doc = self.docidx2eid[val[i]-1]
                for k in other_eve_in_doc:
                    self.adj_matrix[row[i],k] = val[i]

            self.trans_pairs = transitive_closure(trans_pairs)
            print('Trans pairs number: %d' % (len(self.trans_pairs)))

            for s, t in self.trans_pairs:
                self.adj_matrix[s,t] = -1
                self.adj_matrix[t,s] = -1

    # for negative sampling in eventPoincare embedding
    def gen_neg_samples(self, eid, max_neg=1):
        docidx = self.eid_to_docidx[eid]
        sample_pool, = np.where(self.adj_matrix[eid] == docidx+1)
        if len(sample_pool) <= max_neg:
            return sample_pool # numpy array
        else:
            samples = set()
            while len(samples) < max_neg:
                samples.add(choose(sample_pool))
            return list(samples)
        

def transitive_closure(closure):
    #closure = set(closure)
    while True:
        new_relations = set((x,w) for x,y in closure for q,w in closure if q == y)
        closure_until_now = closure | new_relations
        if closure_until_now == closure:
            break
        closure = closure_until_now
    return closure


# a method for quick random choosing
def choose(pool):
    probs = np.random.rand(len(pool))
    probs /= probs.sum()
    choice = pool[np.searchsorted(probs.cumsum(), random.random())]
    return choice

