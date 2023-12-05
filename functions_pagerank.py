import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random


def make_graph(G):
    '''
    Função para preparar o grafo para o algoritmo PageRank.

    '''

    # Convertendo o grafo para direcionado caso não seja
    if not nx.is_directed(G):
        print('O grafo está sendo convertido para direcionado..')
        G = G.to_directed()    

    # Renomeando os nós
    n_unique_nodes = len(set(G.nodes()))
    node2int = dict(zip(set(G.nodes()), range(n_unique_nodes)))
    int2node = {v:k for k,v in node2int.items()}

    G = nx.relabel_nodes(G, node2int)

    # salvando a label original dos nós como o atributo 'label'
    for node in G.nodes():
        G.nodes[node]['label'] = int2node[node]

    # Removendo nós isolados
    nodes = G.nodes()
    for node in nodes:
        if len(G.edges(node))==0:
            G.remove_node(node)
    return G, int2node

def plot_graph(G, final_probs, int2node, bool_final_probs=False):
    '''
    Função para plotar o grafo.       
    '''

    # plot_graph_pyvis(G, final_probs, int2node, bool_final_probs)

    # obtendo a label e pagerank de cada nó
    labels = {}
    pageranks = []
    probs = []
    for node in G.nodes():
        labels[node] = G.nodes[node]['label']
        pageranks.append(G.nodes[node]['pagerank'])

    try:
        clubs = np.array(list(map(lambda x: G.nodes[x]['club'], G.nodes())))
        # labels = dict(zip(G.nodes(), clubs)) 
    except:
        pass   

    pos = nx.spring_layout(G, k=2)  # k regulates the distance between nodes

    # ploting curvy edges
    nx.draw_networkx_edges(
        G, pos, arrows=True, arrowstyle="->", arrowsize=10, alpha=0.3,
        connectionstyle="arc3,rad=0.1"
    )

    font_size = 10
    node_size = 600

    if not bool_final_probs:
        nx.draw_networkx_nodes(G, pos, node_size=node_size, alpha=0.8)
        nx.draw_networkx_labels(G, pos, labels, font_size=font_size)
        # nx.draw(G, with_labels=True, alpha=0.8, arrows=False, labels=labels)
    else:
        # nx.draw(G, with_labels=True, alpha=0.8, arrows=False, node_color = pageranks, \
        #                                                                                 cmap=plt.get_cmap('Purples'), labels=labels)
        nx.draw_networkx_nodes(G, pos, node_size=node_size,
                alpha=0.8, node_color = pageranks, cmap=plt.get_cmap('Purples'))
        nx.draw_networkx_labels(G, pos, labels, font_size=font_size)

        # Adicionando a barra de cores com os valores de ranking
        sm = plt.cm.ScalarMappable(cmap=plt.get_cmap('Purples'), norm=plt.Normalize(vmin = min(pageranks), vmax=max(pageranks)))
        sm._A = []
        plt.colorbar(sm)
    return plt

def make_pagerank_matrix(G, d):
    '''
    Função para construir a matriz de transição do PageRank.
    '''

    n_nodes = len(G.nodes())

    # Construindo matriz de adjacência
    adj_matrix = np.zeros(shape=(n_nodes, n_nodes))

    for edge in G.edges():
        adj_matrix[edge[0], edge[1]] = 1

    # Construindo a matriz P de probabilidade de transição entre os nós
    tran_matrix = adj_matrix / np.sum(adj_matrix, axis=1).reshape(-1,1)

    # Construindo a matriz de navegação aleatória, onde todos os nós tem a mesma probabilidade de serem visitados
    random_surf = np.ones(shape = (n_nodes, n_nodes)) / n_nodes    

    # Construindo a matriz de transição para nós absorventes
    absorbing_nodes = np.zeros(shape = (n_nodes,))
    for node in G.nodes():
        if len(G.out_edges(node))==0:
            absorbing_nodes[node] = 1
    
    absorbing_node_matrix = np.outer(absorbing_nodes, np.ones(shape = (n_nodes,))) / n_nodes

    # Matriz estocástica
    stochastic_matrix = tran_matrix + absorbing_node_matrix

    # Matriz de transição modificada P' = d * P + (1-d) * [1]/n
    pagerank_matrix = d * stochastic_matrix + (1-d) * random_surf
    return pagerank_matrix

def random_walk(G, d, n_iter):
    '''
    Função para realizar o random walk no grafo, retornando a distribuição estacionária.
    '''


    n_nodes = len(G.nodes())

    # Vetor inicial com probabilidades iguais para todos os nós
    initial_state = np.ones(shape=(n_nodes,)) / n_nodes
    pagerank_matrix = make_pagerank_matrix(G, d)

    new_initial_state = initial_state
    NORM = []
    for i in range(n_iter):
        final_state = np.dot(np.transpose(pagerank_matrix), new_initial_state)
        
        prev_initial_state = new_initial_state
        new_initial_state = final_state

        L2 = np.linalg.norm(new_initial_state-prev_initial_state)
        NORM.append(L2)

        if np.allclose(new_initial_state, prev_initial_state):
            print(f'Convergiu em {i+1} iterações.')
            break

    plt.figure(figsize=(8,4))
    plt.plot(range(i+1), NORM)
    plt.xlabel('Iterações')
    plt.ylabel('Norma Euclidiana')
    plt.title('Convergência do PageRank')
    plt.show()
    return final_state

def run(G, d, n_iter):
    '''
    Encontrando o ranqueamento dos nós por meio do PageRank
    '''

    G, int2node = make_graph(G)
    ranks = {}

    final_probs = random_walk(G, d, n_iter)

    # Garantindo que as dimensões estão corretas
    assert len(final_probs) == len(G.nodes())

    # Garantindo que as probabilidades somam 1
    assert np.allclose(np.sum(final_probs), 1)
    
    # Printando o ranking dos nós com os nomes originais 
    print('PageRank:')
    for i in np.argsort(final_probs)[::-1]:
        ranks[int2node[i]] = final_probs[i]
        # print(f'{int2node[i]}: {final_probs[i]}')
        G.nodes[i]['pagerank'] = final_probs[i]

    # adicionando em cada nó o atributo 'pagerank'

    
   
    plt.figure(figsize=(40,10))
    plt.subplot(121)
    plot_graph(G, None, int2node, bool_final_probs=False)
    plt.figure(figsize=(40,10))
    plt.subplot(122)
    plot_graph(G, final_probs, int2node, bool_final_probs=True)
    plt.show()
    return ranks, G