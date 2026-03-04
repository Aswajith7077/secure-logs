import re
import os
import pandas as pd
from datetime import datetime

class LogCluster:
    def __init__(self, log_template, cluster_id):
        self.log_template = log_template
        self.cluster_id = cluster_id
        self.size = 1

class Drain:
    def __init__(self, depth=4, st=0.4, max_child=100, rex=[]):
        """
        depth: depth of the parsing tree (excluding root)
        st: similarity threshold
        max_child: max number of children for each node
        rex: regex patterns for preprocessing
        """
        self.depth = depth - 2
        self.st = st
        self.max_child = max_child
        self.rex = rex
        self.root_node = {}
        self.id_to_cluster = {}

    def preprocess(self, line):
        for current_rex in self.rex:
            line = re.sub(current_rex, '<*>', line)
        return line

    def get_similarity(self, seq1, seq2):
        assert len(seq1) == len(seq2)
        count = 0
        for token1, token2 in zip(seq1, seq2):
            if token1 == token2:
                count += 1
        return count / len(seq1)

    def fast_match(self, cluster_list, tokens):
        match_cluster = None
        max_sim = -1
        max_num_of_params = -1

        for cluster in cluster_list:
            cur_sim = self.get_similarity(cluster.log_template.split(), tokens)
            if cur_sim > max_sim or (cur_sim == max_sim and len(re.findall('<*>', cluster.log_template)) > max_num_of_params):
                max_sim = cur_sim
                max_cluster = cluster
                max_num_of_params = len(re.findall('<*>', cluster.log_template))

        if max_sim >= self.st:
            return max_cluster
        return None

    def get_template(self, seq1, seq2):
        assert len(seq1) == len(seq2)
        res = []
        for i in range(len(seq1)):
            if seq1[i] == seq2[i]:
                res.append(seq1[i])
            else:
                res.append('<*>')
        return " ".join(res)

    def add_log_line(self, log_line):
        log_line = self.preprocess(log_line)
        tokens = log_line.split()
        log_len = len(tokens)

        # Level 1: Log message length
        if log_len not in self.root_node:
            self.root_node[log_len] = {}
        parent_node = self.root_node[log_len]

        # Level 2 to depth: First tokens
        for i in range(min(log_len, self.depth)):
            token = tokens[i]
            if re.search(r'\d', token): # If contains digits, use generic <*>
                token = '<*>'
            
            if token not in parent_node:
                if i == min(log_len, self.depth) - 1:
                    parent_node[token] = []
                else:
                    parent_node[token] = {}
            parent_node = parent_node[token]
        
        # parent_node is now a list of LogCluster objects (leaves)
        match_cluster = self.fast_match(parent_node, tokens)

        if match_cluster is None:
            new_cluster_id = len(self.id_to_cluster) + 1
            new_cluster = LogCluster(log_line, new_cluster_id)
            parent_node.append(new_cluster)
            self.id_to_cluster[new_cluster_id] = new_cluster
            return f"E{new_cluster_id}", log_line
        else:
            new_template = self.get_template(tokens, match_cluster.log_template.split())
            match_cluster.log_template = new_template
            match_cluster.size += 1
            return f"E{match_cluster.cluster_id}", new_template
