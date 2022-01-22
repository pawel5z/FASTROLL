import numpy as np
import graphviz
import pandas as pd
import random


def entropy(counts):
    P = counts/(np.sum(counts))
    return -np.sum(P*np.log2(P + 1e-10))


def gini(counts):
    return 1 - np.sum((counts / np.sum(counts))**2)


def mean_err_rate(counts):
    return 1 - np.max(counts)/np.sum(counts)


class AbstractSplit:
    """Split the examples in a tree node according to a criterion."""

    def __init__(self, attr):
        self.attr = attr
        self.purity_gain = 0.0

    def mean_purity(self, df, purity_function):
        """Return mean purity across all split's children."""
        def purity(child):
            return purity_function(child['target'].value_counts()) * len(child)
        return sum(purity(child) for _, child in df.groupby(self.attr)) / len(df)

    def __call__(self, x):
        """Return the subtree corresponding to x."""
        raise NotImplementedError

    def build_subtrees(self, df, subtree_kwargs):
        """Recuisively build the subtrees."""
        raise NotImplementedError

    def iter_subtrees(self):
        """Return an iterator over subtrees."""
        raise NotImplementedError

    def add_to_graphviz(self, dot):
        """Add the split to the graphviz vizalization."""
        raise NotImplementedError

    def __str__(self):
        return f"{self.__class__.__name__}: {self.attr}"


def get_split(df, purity_function=entropy, nattrs=None):
    target_value_counts = df["target"].value_counts()
    if len(target_value_counts) == 1:
        return None

    possible_splits = [attr for attr in df.columns if attr !=
                       'target' and len(df[attr].unique()) > 1]
    if not possible_splits:
        return None

    # Random Forest support
    # restrict possible_splits to a few radomly selected attributes
    if nattrs is not None:
        random.shuffle(possible_splits)
        possible_splits = possible_splits[:nattrs]

    splits = [
        CategoricalMultivalueSplit(attr)
        for attr in possible_splits
    ]

    best_split = min(splits, key=lambda s: s.mean_purity(df, purity_function))
    base_purity = purity_function(target_value_counts)
    best_split.purity_gain = base_purity - \
        best_split.mean_purity(df, purity_function)
    if best_split.purity_gain >= 0:
        return best_split
    return None


class CategoricalMultivalueSplit(AbstractSplit):
    def __call__(self, x):
        a = x[self.attr]
        if a in self.subtrees:
            return self.subtrees[a]
        return None

    def build_subtrees(self, df, subtree_kwargs):
        self.subtrees = {}
        for group_name, group_df in df.groupby(self.attr):
            child = Tree(group_df, **subtree_kwargs)
            self.subtrees[group_name] = child

    def iter_subtrees(self):
        return self.subtrees.values()

    def add_to_graphviz(self, dot, parent, print_info):
        for split_name, child in self.subtrees.items():
            child.add_to_graphviz(dot, print_info)
            dot.edge(f"{id(parent)}", f"{id(child)}", label=f"{split_name}")


class Tree:
    def __init__(self, df, **kwargs):
        super().__init__()

        # Technicality:
        # We need to let subtrees know about all targets to properly color nodes
        # We pass this in subtree arguments.
        if "all_targets" not in kwargs:
            kwargs["all_targets"] = sorted(df["target"].unique())

        # Save keyword arguments to build subtrees
        kwargs_orig = dict(kwargs)

        # Get kwargs we know about, remaning ones will be used for splitting
        self.all_targets = kwargs.pop("all_targets")

        # Save debug info for visualization
        # Debugging tip: contents of self.info are printed in tree visualizations!
        self.counts = df["target"].value_counts()
        self.info = {
            "num_samples": len(df),
            "entropy": entropy(self.counts),
            "gini": gini(self.counts),
        }

        self.split = get_split(df, **kwargs)
        if self.split:
            self.split.build_subtrees(df, kwargs_orig)

    def __call__(self, sample):
        return self.get_target_distribution(sample).idxmax()

    def prune_confidence(self):
        if self.split is None:
            return 0.0

        z = 1.96  # 95% confidence level

        def confidence(counts):
            err = mean_err_rate(counts)
            return err + z * np.sqrt(err*(1-err)/np.sum(counts))

        node_confidence = self.info['confidence_interval'] = confidence(
            self.counts)
        if any(node_confidence <= tree.prune_confidence() for tree in self.split.iter_subtrees()):
            self.split = None
        return node_confidence

    def leaf(self, sample):
        if self.split is None:
            return self
        subtree = self.split(sample)
        return subtree.leaf(sample) if subtree else self

    def measure_purity_gain(self, attr):
        if self.split == None:
            return 0.0
        purity = sum(subtree.measure_purity_gain(attr)
                     for subtree in self.split.iter_subtrees())
        if self.split.attr == attr:
            purity += self.split.purity_gain
        return purity

    def get_target_distribution(self, sample):
        # descend into subtrees and return the leaf target distribution
        return self.leaf(sample).counts

    def draw(self, print_info=True):
        dot = graphviz.Digraph()
        self.add_to_graphviz(dot, print_info)
        return dot

    def add_to_graphviz(self, dot, print_info):
        freqs = self.counts / self.counts.sum()
        freqs = dict(freqs)
        colors = []
        freqs_info = []
        for i, c in enumerate(self.all_targets):
            freq = freqs.get(c, 0.0)
            if freq > 0:
                colors.append(f"{i%9 + 1};{freq}")
                freqs_info.append(f"{c}:{freq:.2f}")
        colors = ":".join(colors)
        labels = [" ".join(freqs_info)]
        if print_info:
            for k, v in self.info.items():
                labels.append(f"{k} = {v}")
        if self.split:
            labels.append(f"split by: {self.split.attr}")
        dot.node(
            f"{id(self)}",
            label="\n".join(labels),
            shape="box",
            style="striped",
            fillcolor=colors,
            colorscheme="set39",
        )
        if self.split:
            self.split.add_to_graphviz(dot, self, print_info)


def error(tree, test: pd.DataFrame):
    predcts = test.apply(lambda row: tree(row), axis=1).values
    targets = test['target'].values
    return sum(predcts[i] != targets[i] for i in range(len(test))) / len(test)


class RandomForest(list):
    def __init__(self, training_set, target_column_name="target", nattrs=1, size=0, pruning=False):
        self.train = training_set.rename(
            columns={target_column_name: "target"}, inplace=False)
        self.targets = target_column_name
        self.nattrs = nattrs

        # Initialize with size
        for _ in range(size):
            self.add_tree()

        if pruning:
            for tree in self:
                tree.prune_confidence()

    def __call__(self, sample):
        votes = [tree(sample) for tree in self]
        return pd.Series(data=votes).value_counts().idxmax()

    def add_tree(self):
        bootstrap = self.train.sample(frac=1.0, replace=True)
        tree = Tree(bootstrap, nattrs=self.nattrs)
        self.append(tree)

    def forest_error(self, test):
        return error(self, test.rename(columns={self.targets: "target"}, inplace=False))

    def trees_agreement(self, test_df):
        agreement = 0
        for _, sample in test_df.iterrows():
            votes = [tree(sample) for tree in self]
            pred = pd.Series(data=votes).value_counts().idxmax()
            agreement += np.sum(pred == votes)/len(votes)
        return agreement/len(test_df)
