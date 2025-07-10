# 초기 네트워크 생성 파라미터 설정
import networkx as nx
import numpy as np
def get_ER(N,mk):
    p = mk / (N - 1)

    def generate_network(N, p):
        while True:
            # ER 네트워크 생성
            G = nx.erdos_renyi_graph(N, p)
            
            # 외부 링크가 없는 노드 제거
            G.remove_nodes_from(list(nx.isolates(G)))
            
            # 노드 수가 1000개가 되면 종료
            if len(G.nodes()) == N:
                break
        
        return G

    # 네트워크 생성
    Gs = []
    mk_ER = 1
    while mk_ER != mk:
        GER = generate_network(N, p)
        A = nx.adjacency_matrix(GER)
        Aij_ER = np.array(A.todense())
        Deg_ER = Aij_ER.sum(axis=1)
        mk_ER = Deg_ER.mean()
    return GER,Aij_ER,Deg_ER,mk

def get_SF(N,m):
    GSF = nx.barabasi_albert_graph(N,m)
    A = nx.adjacency_matrix(GSF)
    Aij_SF = np.array(A.todense())
    Deg_SF = Aij_SF.sum(axis=1)
    mk_SF = Deg_SF.mean()
    return GSF,Aij_SF,Deg_SF,mk_SF
