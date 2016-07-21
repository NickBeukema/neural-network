import networkx as nx
import matplotlib.pyplot as plt
import pdb
import numpy

from networkx.drawing.nx_agraph import write_dot
from networkx.drawing.nx_agraph import graphviz_layout

def graph_line(data, title):
    plt.plot(range(len(data['train'])), data['train'], linestyle='--', color='r', label='Training Set')
    plt.plot(range(len(data['test'])), data['test'], label='Testing Set')
    plt.plot(range(len(data['categorize_accuracy'])), data['categorize_accuracy'], color='g', label="Categorize")

    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.title(title)
    plt.legend()
    plt.ylim([-.1, 1])
    plt.show()
