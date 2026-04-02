import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Search, Loader2, Clock, Award, Star, X, MousePointer2, GitBranch, HelpCircle, Layers, Fingerprint, ChevronDown, ChevronUp, Maximize, Info } from 'lucide-react';
import ForceGraph2D from 'react-force-graph-2d';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';

const THEME = {
  primary: '#10b981',
  secondary: '#7dd3fc',
  accent: '#f59e0b',
  bg: '#050505',
  glass: 'rgba(255, 255, 255, 0.02)',
  glassBorder: 'rgba(255, 255, 255, 0.06)',
  textDim: '#94a3b8'
};

const Logo = () => (
  <svg width="24" height="24" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
    <linearGradient id="logoGrad" x1="0%" y1="100%" x2="100%" y2="0%">
      <stop offset="0%" style={{stopColor: '#10b981', stopOpacity: 1}} />
      <stop offset="100%" style={{stopColor: '#7dd3fc', stopOpacity: 1}} />
    </linearGradient>
    <path d="M4 24L12 16L18 22L28 8" stroke="url(#logoGrad)" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
    <path d="M20 8H28V16" stroke="url(#logoGrad)" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
    <circle cx="4" cy="24" r="2" fill="#10b981" />
    <circle cx="12" cy="16" r="2" fill="#10b981" />
    <circle cx="18" cy="22" r="2" fill="#10b981" />
    <circle cx="28" cy="8" r="2" fill="#7dd3fc" />
  </svg>
);

// Generates an aesthetic color string from a seed (used for rough clustering)
const stringToColor = (str) => {
  let hash = 0;
  for (let i = 0; i < str.length; i++) hash = str.charCodeAt(i) + ((hash << 5) - hash);
  const hue = Math.abs(hash) % 360;
  return `hsl(${hue}, 70%, 65%)`;
};

const App = () => {
  // Input tracking
  const [searchTerm, setSearchTerm] = useState('');
  
  // Last successfully executed query
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState([]);
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  
  const [selectedNodeId, setSelectedNodeId] = useState(null);
  const [hoveredNode, setHoveredNode] = useState(null);

  // Graph Controls (Changed default limit to 15)
  const [showEdges, setShowEdges] = useState(true);
  const [clusterView, setClusterView] = useState(false);
  
  const fgRef = useRef();
  const listRefs = useRef({});

  // Trigger search ONLY on form submission (enter key or button click)
  const performSearch = async (q) => {
    setLoading(true);
    setResults([]);
    setGraphData({ nodes: [], links: [] });
    setSelectedNodeId(null);
    try {
      // Backend fetches top 50, but we filter display client-side using nodeCount slider limit
      const response = await axios.post('http://localhost:8001/search', { query: q, top_k: 50 });
      setResults(response.data.results || []);
      setGraphData(response.data.graph || { nodes: [], links: [] });
    } catch (error) {
      console.error('Search error:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleManualSearch = (e) => {
    e.preventDefault();
    if (searchTerm && searchTerm.trim() !== '') {
      setQuery(searchTerm.trim());
      performSearch(searchTerm.trim());
    }
  };

  // Sync List to Graph
  const focusOnNode = useCallback((nodeId) => {
    if (fgRef.current && graphData.nodes.length > 0) {
      const gNode = graphData.nodes.find(n => n.id === nodeId);
      if (gNode && typeof gNode.x === 'number') {
        fgRef.current.centerAt(gNode.x, gNode.y, 800);
        fgRef.current.zoom(8, 800);
      }
    }
  }, [graphData]);

  const handleItemClick = (paperId) => {
    setSelectedNodeId(paperId);
    focusOnNode(paperId);
  };

  // Sync Graph to List
  const handleNodeClick = (node) => {
    if (!node || !node.id) return;
    setSelectedNodeId(node.id);
    focusOnNode(node.id);
    
    // Scroll list
    const el = listRefs.current[node.id];
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  };

  // Filter local display based on top 15 results
  const displayedResults = React.useMemo(() => results.slice(0, 15), [results]);
  
  const filteredGraphData = React.useMemo(() => {
    const topPaperIds = new Set(displayedResults.map(p => p.paper_id));
    // Keep nodes that are within the top nodeCount results, AND all bridge nodes connected to them
    const validNodes = graphData.nodes.filter(n => n.is_bridge || topPaperIds.has(n.id));
    const validNodeIds = new Set(validNodes.map(n => n.id));
    // Keep links where both source and target exist
    const validLinks = graphData.links.filter(l => 
      validNodeIds.has(typeof l.source === 'object' ? l.source.id : l.source) &&
      validNodeIds.has(typeof l.target === 'object' ? l.target.id : l.target)
    );
    return { nodes: validNodes, links: validLinks };
  }, [graphData, displayedResults]);

  return (
    <div className="search-container">
      <header className="header">
        <div className="header-content">
          <div className="logo-area">
            <div className="logo-icon">
              <Logo />
            </div>
            <h1 className="logo-text">LTR</h1>
          </div>

          <form onSubmit={handleManualSearch} className="search-bar-wrapper">
             <Search size={20} className="search-icon-inline" />
             <input
               type="text"
               value={searchTerm}
               onChange={(e) => setSearchTerm(e.target.value)}
               placeholder="Enter research query (e.g. contrastive learning)..."
               className="search-input"
             />
             <button type="submit" className="search-submit-btn" disabled={loading}>
                {loading ? <Loader2 size={16} className="animate-spin" /> : <MousePointer2 size={16} />}
                Search
             </button>
          </form>
        </div>
      </header>

      <main className="main-grid">
        <div className="results-column">
          <div className="results-header">
            <span style={{ fontSize: '0.85rem', color: THEME.textDim, letterSpacing: '1px', fontWeight: 600 }}>
              RELEVANCE RANKED
            </span>
            {displayedResults.length > 0 && (
              <span style={{ fontSize: '0.8rem', color: THEME.primary, fontWeight: 600 }}>Top {displayedResults.length} Results</span>
            )}
          </div>

          <div className="scroll-area">
            {loading && [1,2,3,4,5].map(k => (
              <div key={k} className="paper-card" style={{ height: '140px' }}>
                 <div className="skeleton" style={{ width: '80%', height: '20px', marginBottom: '12px' }} />
                 <div className="skeleton" style={{ width: '60%', height: '20px', marginBottom: '16px' }} />
                 <div style={{ display: 'flex', gap: '8px' }}>
                    <div className="skeleton" style={{ width: '60px', height: '20px' }} />
                    <div className="skeleton" style={{ width: '80px', height: '20px' }} />
                 </div>
              </div>
            ))}
            {!loading && results.length === 0 && (
              <div style={{ padding: '4rem 0', textAlign: 'center', opacity: 0.3 }}>
                <Layers size={56} style={{ margin: '0 auto 1.5rem', color: THEME.primary }} />
                <p style={{ fontSize: '1rem', color: '#fff' }}>Execute a search query to see recommendations...</p>
              </div>
            )}
            <AnimatePresence>
              {!loading && displayedResults.map((paper, idx) => {
                const isSelected = selectedNodeId === paper.paper_id;
                return (
                  <motion.div
                    key={paper.paper_id}
                    ref={el => listRefs.current[paper.paper_id] = el}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: Math.min(idx * 0.05, 0.5), type: "spring" }}
                    onClick={() => handleItemClick(paper.paper_id)}
                    className={`paper-card ${isSelected ? 'active' : ''}`}
                  >
                    <h3 className="paper-title">{paper.title}</h3>
                    <div className="badge-row">
                      <Badge icon={Clock} label={paper.year || 'N/A'} />
                      <Badge icon={Award} label={`Rank #${paper.final_rank || paper.rank_init}`} />
                      {paper.pagerank > 0 && (
                         <Badge icon={Star} label={`PR: ${paper.pagerank.toFixed(4)}`} color={THEME.secondary} />
                      )}
                    </div>
                    
                    <AnimatePresence>
                      {isSelected && (
                        <motion.div
                          initial={{ height: 0, opacity: 0 }}
                          animate={{ height: 'auto', opacity: 1 }}
                          exit={{ height: 0, opacity: 0 }}
                          style={{ overflow: 'hidden' }}
                        >
                          <div className="paper-abstract">
                            {paper.abstract}
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>
                    
                    <div style={{ position: 'absolute', top: '1.25rem', right: '1.25rem', opacity: isSelected ? 1 : 0.2, transition: '0.3s' }}>
                       {isSelected ? <ChevronUp size={18} color={THEME.primary} /> : <ChevronDown size={18} color="#fff" />}
                    </div>
                  </motion.div>
                );
              })}
            </AnimatePresence>
          </div>
        </div>

        <div className="graph-column">
           <div className="graph-top-bar" style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: '0.8rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', width: '100%', alignItems: 'center' }}>
                <h2 className="graph-title">Citation Connections Graph</h2>
                <div className="graph-controls">
                   <label className="control-item" title="Toggle citation relationships">
                      <input type="checkbox" checked={showEdges} onChange={e => setShowEdges(e.target.checked)} />
                      Edges
                   </label>
                   <label className="control-item" title="Color code nodes by publication year">
                      <input type="checkbox" checked={clusterView} onChange={e => setClusterView(e.target.checked)} />
                      Clusters
                   </label>
                   <button 
                      className="focus-btn" 
                      onClick={() => {
                          if (selectedNodeId) focusOnNode(selectedNodeId);
                          else if (fgRef.current) fgRef.current.zoomToFit(400);
                      }}
                   >
                      <Maximize size={14} /> Focus
                   </button>
                </div>
              </div>
              
              <div style={{ display: 'flex', justifyContent: 'space-between', width: '100%', alignItems: 'center', background: 'rgba(0,0,0,0.2)', padding: '0.5rem 0.8rem', borderRadius: '8px' }}>
                 <p style={{ margin: 0, fontSize: '0.8rem', color: THEME.textDim, opacity: 0.9, display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <Info size={14} color={THEME.secondary} />
                    <strong>Graph Guide:</strong> Nodes represent research papers. Larger nodes indicate high Authority Index. Lines are citations.
                 </p>
                 <div style={{ display: 'flex', gap: '1rem' }}>
                    <div className="legend-chip">
                       <span className="dot" style={{ background: 'rgba(255,255,255,0.85)' }}></span> Query Result
                    </div>
                    <div className="legend-chip">
                       <GitBranch size={10} color={THEME.textDim} style={{marginRight: '2px'}}/> 
                       <span className="dot" style={{ background: THEME.textDim, opacity: 0.8 }}></span> Shared Bridge
                    </div>
                 </div>
              </div>
           </div>

           <div className="graph-container">
              {filteredGraphData.nodes.length > 0 ? (
                <>
                  <ForceGraph2D
                    ref={fgRef}
                    graphData={filteredGraphData}
                    // Edges
                    linkColor={() => showEdges ? THEME.primary + '30' : 'transparent'}
                    linkWidth={showEdges ? 1.5 : 0}
                    linkDirectionalArrowLength={showEdges ? 4 : 0}
                    linkDirectionalArrowRelPos={1}
                    linkCurvature={0.2}
                    // Force Engine
                    d3AlphaDecay={0.02}
                    d3VelocityDecay={0.1}
                    backgroundColor={THEME.bg}
                    onNodeHover={(node) => setHoveredNode(node)}
                    onNodeClick={handleNodeClick}
                    nodeCanvasObject={(node, ctx, globalScale) => {
                      if (!node || !isFinite(node.x) || !isFinite(node.y)) return;

                      const isSelected = selectedNodeId === node.id;
                      const isHovered = hoveredNode?.id === node.id;
                      const isBridge = node.is_bridge;

                      // Size based on relevance/PageRank if not bridge
                      const baseRadius = isBridge ? 4 : 6 + (node.pagerank ? node.pagerank * 10 : 0);
                      const radius = Math.min(Math.max(baseRadius, 3), 12); // Clamped

                      // Cluster Color Logic
                      let nodeColor = 'rgba(255,255,255,0.7)';
                      if (clusterView) {
                         const clusterSeed = node.year || (isBridge ? 'bridge' : 'recent');
                         nodeColor = isBridge ? THEME.textDim : stringToColor(String(clusterSeed));
                      } else {
                         nodeColor = isBridge ? THEME.textDim : 'rgba(255,255,255,0.85)';
                      }

                      // Glow for active/hovered
                      if (isSelected || isHovered) {
                          ctx.shadowBlur = 15;
                          ctx.shadowColor = isSelected ? THEME.primary : '#fff';
                      } else {
                          ctx.shadowBlur = 0;
                      }

                      ctx.beginPath(); 
                      ctx.arc(node.x, node.y, radius, 0, 2 * Math.PI, false); 
                      ctx.fillStyle = isSelected ? THEME.primary : nodeColor;
                      ctx.fill();
                      
                      if (isSelected || isBridge) {
                          ctx.strokeStyle = isSelected ? '#fff' : 'rgba(255,255,255,0.2)';
                          ctx.lineWidth = 1.5/globalScale;
                          ctx.stroke();
                      }

                      // Draw Label if zoomed in or target
                      if (globalScale > 1.2 || isSelected || isHovered) {
                        const label = node.title?.length > 40 ? node.title.substring(0, 40) + '...' : node.title;
                        const fontSize = 11/globalScale;
                        ctx.font = `500 ${fontSize}px Inter`;
                        ctx.textAlign = 'center';
                        ctx.textBaseline = 'top';
                        ctx.fillStyle = isSelected || isHovered ? '#fff' : 'rgba(255,255,255,0.6)';
                        ctx.shadowBlur = 0;
                        ctx.fillText(label, node.x, node.y + radius + 4/globalScale);
                      }
                    }}
                  />
                  
                  {hoveredNode && hoveredNode.id !== selectedNodeId && (
                     <div 
                       className="graph-tooltip"
                       style={{ 
                         left: window.innerWidth < 800 ? '50%' : fgRef.current?.graph2ScreenCoords(hoveredNode.x, hoveredNode.y)?.x || 0, 
                         top: fgRef.current?.graph2ScreenCoords(hoveredNode.x, hoveredNode.y)?.y || 0
                       }}
                     >
                        <strong>{hoveredNode.year || 'Date N/A'}</strong> - {hoveredNode.title?.substring(0,60) + (hoveredNode.title?.length>60?'...':'')}
                     </div>
                  )}
                </>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', height: '100%', alignItems: 'center', justifyContent: 'center', opacity: 0.1 }}>
                  <HelpCircle size={80} color="#fff" />
                  <p style={{ marginTop: '1rem', color: '#fff' }}>Interactive citation relationships will appear here.</p>
                </div>
              )}
           </div>
        </div>
      </main>
    </div>
  );
};

const Badge = ({ icon: Icon, label, color = "#fff" }) => (
  <div className="badge" style={{ color: color }}>
    <Icon size={12} />
    {label}
  </div>
);

export default App;
