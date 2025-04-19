from collections import defaultdict
from itertools import chain, combinations

class Apriori:
    def __init__(self, transactions, min_support=0.5, min_confidence=0.5):
        self.transactions = transactions
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.frequent_itemsets = {}

    def find_rules(self):
        self.frequent_itemsets = self.__find_frequent_itemsets(self.transactions, self.min_support)

        return self.__find_rules(self.frequent_itemsets, self.min_confidence, self.transactions)

    def __find_frequent_itemsets(self, transactions, min_support):
        n_transactions = len(transactions)
        candidates = defaultdict(int)

        for transaction in transactions:
            for item in transaction:
                candidates[frozenset([item])] += 1

        current_itemsets = {
            itemset: appearance_cnt
            for itemset, appearance_cnt in candidates.items() if appearance_cnt / n_transactions >= min_support
        }

        all_frequent_itemsets = dict(current_itemsets)
        k = 2

        while current_itemsets:
            candidates = defaultdict(int)
            itemsets = list(current_itemsets.keys())

            for i in range(len(itemsets)):
                for j in range(i + 1, len(itemsets)):
                    candidate = itemsets[i] | itemsets[j]

                    if len(candidate) == k:
                        candidates[candidate] = self.__eval_support(candidate, transactions)

            current_itemsets = {
                itemset: appearance_cnt
                for itemset, appearance_cnt in candidates.items() if appearance_cnt / n_transactions >= min_support
            }

            all_frequent_itemsets.update(current_itemsets)
            k += 1

        return all_frequent_itemsets

    def __find_rules(self, frequent_itemsets, min_confidence, transactions):
        rules = []

        for itemset, appearance_cnt in frequent_itemsets.items():
            for subset in self.__get_subsets(itemset):
                if subset and itemset.difference(subset):
                    subset_support = frequent_itemsets[subset] if subset in frequent_itemsets else self.__eval_support(subset, transactions)
                    confidence = appearance_cnt / subset_support

                    if confidence >= min_confidence:
                        rules.append((subset, itemset.difference(subset), confidence))

        return rules

    def __get_subsets(self, itemset):
        return [frozenset(subset) for subset in chain.from_iterable(combinations(itemset, r) for r in range(len(itemset) + 1))]

    def __eval_support(self, subset, transactions):
        return sum(1 for t in transactions if subset.issubset(t))

    def print_frequent_itemsets(self):
        n_transactions = len(self.transactions)

        for frequent_set, appearance_cnt in self.frequent_itemsets.items():
            print(f'{set(frequent_set)}, support: {appearance_cnt / n_transactions}')