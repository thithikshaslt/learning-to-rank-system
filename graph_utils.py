import requests
import networkx as nx
import time
import os
import re

class CitationGraph:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        self.headers = {"x-api-key": api_key} if api_key else {}
        self.cache = {}

    def normalize_id(self, paper_id):
        """Strip ArXiv version suffixes (e.g., 2101.12345v2 -> 2101.12345)"""
        if not paper_id: return None
        return re.sub(r'v\d+$', '', paper_id)

    def fetch_paper_data_batch(self, paper_ids):
        """
        Fetch references, citations, and metadata for a list of ArXiv IDs using Semantic Scholar Batch API.
        """
        # Prefix with 'ARXIV:' as required by Semantic Scholar
        ids_to_fetch = [f"ARXIV:{self.normalize_id(pid)}" for pid in paper_ids if pid]
        
        url = f"{self.base_url}/paper/batch"
        # Increased fields to include citations and better metadata
        params = {"fields": "paperId,externalIds,title,year,citationCount,references.paperId,references.title,references.externalIds,citations.paperId,citations.title,citations.externalIds,citations.year"}
        
        try:
            response = requests.post(url, json={"ids": ids_to_fetch}, params=params, headers=self.headers)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error fetching batch data: {response.status_code}")
                return []
        except Exception as e:
            print(f"Exception during batch fetch: {e}")
            return []

    def build_graph(self, top_papers_metadata):
        """
        Build a local citation graph from paper metadata.
        """
        G = nx.DiGraph()
        
        # Mapping for easy lookup (S2 ID -> Normalized ArXiv ID)
        s2_to_arxiv = {}
        arxiv_to_title = {}
        
        # 1. Add our main result nodes
        for paper in top_papers_metadata:
            if not paper: continue
            s2_id = paper.get('paperId')
            ext_ids = paper.get('externalIds', {})
            arxiv_id = self.normalize_id(ext_ids.get('ArXiv'))
            
            if s2_id and arxiv_id:
                s2_to_arxiv[s2_id] = arxiv_id
                arxiv_to_title[arxiv_id] = paper.get('title')
                G.add_node(arxiv_id, 
                           title=paper.get('title'),
                           year=paper.get('year'),
                           citation_count=paper.get('citationCount', 0),
                           is_query_result=True)

        # 2. Add edges and potentially "Bridge" nodes
        # We look for shared ancestors to create more connectivity
        reference_frequency = {} # S2_ID -> count
        
        for paper in top_papers_metadata:
            if not paper: continue
            current_arxiv = self.normalize_id(paper.get('externalIds', {}).get('ArXiv'))
            if not current_arxiv: continue
            
            # Case A/B: References (Outgoing)
            references = paper.get('references', [])
            for ref in references:
                ref_s2_id = ref.get('paperId')
                if not ref_s2_id: continue
                
                if ref_s2_id in s2_to_arxiv:
                    G.add_edge(current_arxiv, s2_to_arxiv[ref_s2_id])
                else:
                    reference_frequency[ref_s2_id] = reference_frequency.get(ref_s2_id, [])
                    reference_frequency[ref_s2_id].append({
                        'source': current_arxiv,
                        'title': ref.get('title'),
                        'arxiv': self.normalize_id(ref.get('externalIds', {}).get('ArXiv'))
                    })
            
            # Case C: Citations (Incoming) - This breaks the "Uniform PageRank" issue
            citations = paper.get('citations') or []
            # Limit citations to top 20 to keep graph performant
            for cit in citations[:20]:
                if not cit: continue
                cit_s2_id = cit.get('paperId')
                ext_ids = cit.get('externalIds') or {}
                cit_arxiv = self.normalize_id(ext_ids.get('ArXiv'))
                if not cit_s2_id: continue
                
                cit_node_id = cit_arxiv or cit_s2_id
                if cit_node_id not in G:
                    G.add_node(cit_node_id, 
                               title=cit.get('title'),
                               year=cit.get('year'),
                               is_query_result=False,
                               is_citation_source=True)
                
                # Edge goes FROM the citation source TO our result paper
                G.add_edge(cit_node_id, current_arxiv)

        # 3. Add "Bridge" nodes (references cited by at least 2 of our papers)
        for ref_s2_id, sources in reference_frequency.items():
            if len(sources) >= 2:
                # This is a shared foundation! Add it to the graph.
                ref_info = sources[0]
                bridge_id = ref_info['arxiv'] or ref_s2_id # Use ArXiv ID if available, else S2 ID
                
                if bridge_id not in G:
                    G.add_node(bridge_id, 
                               title=ref_info['title'],
                               is_query_result=False, # It's a bridge, not a direct search result
                               is_bridge=True)
                
                for src in sources:
                    G.add_edge(src['source'], bridge_id)
        
        return G

    def compute_metrics(self, G):
        """
        Compute PageRank and Centrality for all nodes in the graph.
        """
        if len(G.nodes) == 0:
            return {}
            
        try:
            # PageRank (authority) - handles cyclic graphs well
            pagerank = nx.pagerank(G, alpha=0.85)
            
            # In-degree centrality (how many papers cite this one)
            in_degree = nx.in_degree_centrality(G)
            
            metrics = {}
            for node in G.nodes:
                metrics[node] = {
                    "pagerank": pagerank.get(node, 0),
                    "centrality": in_degree.get(node, 0),
                    "year": G.nodes[node].get('year') # Include year in metrics for sync
                }
            return metrics
        except Exception as e:
            print(f"Error computing metrics: {e}")
            return {node: {"pagerank": 0, "centrality": 0} for node in G.nodes}

    def get_pipeline_data(self, paper_ids):
        """
        Unified wrapper for Step 5, 6, 7.
        """
        print(f"  Fetching citation data for {len(paper_ids)} papers...")
        data = self.fetch_paper_data_batch(paper_ids)
        
        print("  Building graph...")
        G = self.build_graph(data)
        
        print("  Computing graph metrics...")
        metrics = self.compute_metrics(G)
        
        # Prepare graph data for frontend (nodes and edges)
        graph_json = {
            "nodes": [{"id": n, **G.nodes[n], **metrics.get(n, {})} for n in G.nodes],
            "links": [{"source": u, "target": v} for u, v in G.edges]
        }
        
        return metrics, graph_json
