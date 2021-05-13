# https://github.com/mdhiebert/clustercheck/blob/master/graph.py

class ClusterVertex:
    def __init__(self, name, neighbors=None):
        self.name = name
        
        if neighbors is not None:
            self.neighbors = neighbors
        else:
            self.neighbors = set()

    def connected_to(self, other_vertex):
        """
            Returns True if this vertex is connected to `other_vertex`, else False.
        """
        raise NotImplementedError

    def add_neighbor(self, other_vertex):
        """
            Add a neighbor `other_vertex` to this vertex and vice versa.
        """
        raise NotImplementedError

    def get_neighbors(self):
        """
            Returns a set of the neighbors of this vertex.
        """

    def isolate(self):
        """
            Remove all neighbors from this vertex.
        """
        raise NotImplementedError

    def json(self):
        """
            A toJSON() method.
        """
        raise NotImplementedError

    def __str__(self):
        return '{} :: {}'.format(self.name, [n.name for n in self.get_neighbors()])

    def __eq__(self, other):
        return isinstance(self, ClusterVertex) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

class ClusterGraph:
    """
        A graph of clusters
    """
    def __init__(self, vertices):
        """
            Every ClusterGraph has a set of vertices of abstract type `ClusterVertex`
        """
        self.vertices = vertices

    def add_vertex(self, vertex):
        """
            Adds a vertex to this `ClusterGraph`
        """
        raise NotImplementedError

    def get_clusters(self):
        """
            Return a list of sets of `ClusterVertex`, where each set represents a distinct cluster in this graph.
        """
        raise NotImplementedError

    def get_vertex(self, vertex_name):
        """
            Returns the vertex with the vertex_name if it exists, else False
        """
        raise NotImplementedError

class CGMetricsWrapper:
    """
        A meta-graph with two subgraphs: predicted and actual.
        We compare the nodes and edges of these two subgraphs to produce metrics.
    """

    def __init__(self, predicted=None, actual=None):
        self.predicted = predicted
        self.actual = actual

    def load_from_text_file(self, text_file_path):
        """
            Constructs a `ClusterGraph` based on text-file input from
            the binary labeler tool.
        """

        raise NotImplementedError

    def save_to_json_file(self, json_file_path):
        """
            Saves this CGMetricsWrapper to a json file.
        """
        raise NotImplementedError

    def load_from_json_file(self, json_file_path):
        """
            Constructs a `ClusterGraph` based on json-file input from the
            meta-labeler tool.
        """
        raise NotImplementedError

    def metrics(self):
        """
            Return the precision, recall, and fscore of this meta graph.
        """
        raise NotImplementedError

    def get_actual_vertex(self, vertex_name):
        return self.actual.get_vertex(vertex_name)

# # # SUPERNODE # # #
import os
from os.path import isfile, join

# Implementations of the Abstract Classes in `graph.py`

class SuperNodeCV():
    """
        A Supernode class that supports `NodeCV`
    """

    def __init__(self, neighbors=None):
        if neighbors is None:
            self.neighbors = set()
        else:
            self.neighbors = neighbors

class NodeCV(ClusterVertex):
    """
        An implementation of ClusterVertex that represents
        actual nodes in a cluster, connected to exactly one other
        node - the cluster node of the cluster it belongs to.
        There must ALWAYS be a connected supernode.
    """
    def __init__(self, name, supernode=None):
        super(NodeCV, self).__init__(name, set())

        if supernode is not None:
            self.supernode = supernode
        else:
            self.supernode = SuperNodeCV(neighbors=set([self]))


    def connected_to(self, other_vertex):
        return other_vertex in self.supernode.neighbors

    def add_neighbor(self, other_vertex):
        self.supernode.neighbors.update(other_vertex.supernode.neighbors)

        for vertex in other_vertex.supernode.neighbors:
            vertex.supernode = self.supernode

    def get_neighbors(self):
        return self.supernode.neighbors

    def isolate(self):
        self.supernode.neighbors.remove(self)
        self.supernode = SuperNodeCV(neighbors=set([self]))
    
    def json(self):
        return {
            'name': self.name,
            'neighbors': [v.name for v in self.get_neighbors()]
        }

class NodeCG(ClusterGraph):
    def __init__(self, vertices):
        self.vertices = vertices

    def add_vertex(self, vertex):
        self.vertices.add(vertex)

    def get_clusters(self):
        clusters = []

        vertices = set([vertex for vertex in self.vertices])

        while len(vertices) > 0:
            vertex = vertices.pop()
            cluster = set([vertex]).union(vertex.get_neighbors())
            clusters.append(cluster)
            vertices.difference_update(cluster)

        return clusters

    def get_vertex(self, vertex_name):
        for vertex in self.vertices:
            if vertex.name == vertex_name:
                return vertex
        return False