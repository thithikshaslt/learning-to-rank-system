import React, { useState, useEffect } from 'react';
import { Search, Loader2, Share2, Info, Book, Clock, Award, Star, X, MousePointer2, GitBranch, HelpCircle, Layers, Fingerprint } from 'lucide-react';
import ForceGraph2D from 'react-force-graph-2d';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';

const App = () => {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState([]);
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  const [selectedNode, setSelectedNode] = useState(null);

  // Premium Next-Gen Palette
  const THEME = {
    primary: '#b794f4',
    secondary: '#63b3ed',
    accent: '#f687b3',
    bg: '#0b0e14',
    glass: 'rgba(255, 255, 255, 0.03)',
    glassBorder: 'rgba(255, 255, 255, 0.08)',
    textDim: 'rgba(255, 255, 255, 0.7)'
  };

  const handleSearch = async (e) => {
    if (e) e.preventDefault();
    if (!query) return;

    setLoading(true);
    setResults([]); // Clear previous results to trigger entry animations
    try {
      const response = await axios.post('http://localhost:8001/search', { query });
      setResults(response.data.results);
      setGraphData(response.data.graph);
    } catch (error) {
      console.error('Search error:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="search-container">
      <header className="header">
        <div className="header-content">
          <div className="logo-area">
            <div className="logo-icon">
              <Fingerprint size={24} color="#0b0e14" />
            </div>
            <h1 className="logo-text">GraphSearch</h1>
          </div>

          <form onSubmit={handleSearch} className="search-bar-wrapper">
             <Search size={22} className="search-icon-inline" />
             <input
               type="text"
               value={query}
               onChange={(e) => setQuery(e.target.value)}
               placeholder="Enter research query..."
               className="search-input"
             />
             <button type="submit" className="search-submit-btn">
                {loading ? <Loader2 size={18} className="animate-spin" /> : <MousePointer2 size={18} />}
                Search
             </button>
          </form>
        </div>
      </header>

      <main className="main-grid">
        <div className="results-column">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
            <span style={{ fontSize: '0.9rem', color: THEME.textDim, opacity: 0.8, letterSpacing: '1px' }}>
              RELEVANCE RANKED
            </span>
            {results.length > 0 && (
              <span style={{ fontSize: '0.8rem', color: THEME.primary, fontWeight: 600 }}>{results.length} Found</span>
            )}
          </div>

          <div className="scroll-area">
            {results.length === 0 && !loading && (
              <div style={{ padding: '4rem 0', textAlign: 'center', opacity: 0.3 }}>
                <Layers size={64} style={{ margin: '0 auto 1.5rem', color: THEME.primary }} />
                <p style={{ fontSize: '1.1rem', color: '#fff' }}>Start your multi-stage search...</p>
              </div>
            )}
            <AnimatePresence>
              {results.map((paper, idx) => (
                <motion.div
                  key={paper.paper_id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: idx * 0.08, type: "spring", stiffness: 100 }}
                  onClick={() => setSelectedNode({ ...paper, id: paper.paper_id })}
                  className={`paper-card ${selectedNode?.paper_id === paper.paper_id ? 'active' : ''}`}
                >
                  <h3 className="paper-title">{paper.title}</h3>
                  <div className="badge-row">
                    <Badge icon={Clock} label={paper.year || 'N/A'} />
                    <Badge icon={Award} label={`LTR Rank #${paper.final_rank || paper.rank_init}`} />
                    {paper.pagerank > 0 && (
                       <Badge icon={Star} label={`Authority PR: ${paper.pagerank.toFixed(4)}`} color={THEME.secondary} />
                    )}
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        </div>

        <div className="graph-column">
           <div className="graph-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <h2 style={{ fontSize: '1.4rem', margin: 0, fontWeight: 700, backgroundImage: `linear-gradient(to right, #fff, ${THEME.primary})`, WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                    Citation Intelligence
                </h2>
                <p style={{ fontSize: '0.9rem', color: THEME.textDim, opacity: 0.6, marginTop: '0.4rem' }}>
                    Visualizing shared heritage and authority clusters.
                </p>
              </div>
              
              <div className="graph-legend-overlay">
                 <div className="legend-chip">
                    <span className="dot" style={{ background: '#fff' }}></span> Results
                 </div>
                 <div className="legend-chip">
                    <span className="dot" style={{ background: THEME.primary + '80' }}></span> Bridge
                 </div>
              </div>
           </div>

           <div className="graph-container">
              {graphData.nodes.length > 0 ? (
                <ForceGraph2D
                  graphData={graphData}
                  linkColor={() => THEME.primary + '20'}
                  linkDirectionalArrowLength={5}
                  linkDirectionalArrowRelPos={1}
                  linkCurvature={0.25}
                  nodeRelSize={7}
                  backgroundColor={THEME.bg}
                  nodeLabel={node => `${node.title} (${node.year || 'N/A'})`}
                  nodeCanvasObject={(node, ctx, globalScale) => {
                    // SAFETY CHECK: Ensure coordinates are finite before drawing
                    if (!node || !isFinite(node.x) || !isFinite(node.y)) return;

                    const label = node.title?.split(' ').slice(0, 3).join(' ') + '...';
                    const fontSize = 12/globalScale;
                    ctx.font = `500 ${fontSize}px Outfit`;
                    const textWidth = ctx.measureText(label).width;

                    const isSelected = selectedNode?.id === node.id;
                    const isBridge = node.is_bridge;

                    // Shadow Casting
                    if (isSelected) {
                        ctx.shadowBlur = 20;
                        ctx.shadowColor = THEME.primary;
                    }

                    // Node Style
                    try {
                        const gradient = ctx.createRadialGradient(node.x, node.y, 0, node.x, node.y, 7);
                        if (isSelected) {
                            gradient.addColorStop(0, '#fff');
                            gradient.addColorStop(1, THEME.primary);
                        } else if (isBridge) {
                            gradient.addColorStop(0, THEME.primary + '50');
                            gradient.addColorStop(1, 'rgba(255,255,255,0.1)');
                        } else {
                            gradient.addColorStop(0, 'rgba(255,255,255,0.9)');
                            gradient.addColorStop(1, 'rgba(255,255,255,0.4)');
                        }

                        ctx.fillStyle = gradient;
                        ctx.beginPath(); 
                        ctx.arc(node.x, node.y, isBridge ? 4.5 : 7, 0, 2 * Math.PI, false); 
                        ctx.fill();
                    } catch (e) {
                        // Fallback if gradient fails
                        ctx.fillStyle = isSelected ? '#fff' : 'rgba(255,255,255,0.5)';
                        ctx.beginPath(); 
                        ctx.arc(node.x, node.y, isBridge ? 4.5 : 7, 0, 2 * Math.PI, false); 
                        ctx.fill();
                    }
                    
                    if (isSelected) {
                        ctx.strokeStyle = '#fff';
                        ctx.lineWidth = 2/globalScale;
                        ctx.stroke();
                    }

                    // Text Label
                    if (globalScale > 1) {
                      ctx.shadowBlur = 0;
                      ctx.fillStyle = isSelected ? '#fff' : 'rgba(255,255,255,0.6)';
                      ctx.fillText(label.toUpperCase(), node.x - textWidth / 2, node.y + 16);
                    }
                  }}
                  onNodeClick={(node) => {
                    const paper = results.find(r => r.paper_id === node.id);
                    if (paper) {
                        setSelectedNode({ ...paper, id: node.id });
                    } else {
                        setSelectedNode({ 
                            title: node.title, 
                            abstract: "This paper acts as a 'Bridge Paper'—a fundamental research node that links multiple results in your current exploration.",
                            is_bridge: true,
                            id: node.id,
                            year: node.year,
                            pagerank: node.pagerank
                        });
                    }
                  }}
                />
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', height: '100%', alignItems: 'center', justifyContent: 'center', opacity: 0.1 }}>
                  <HelpCircle size={80} color="#fff" />
                  <p style={{ marginTop: '1rem', color: '#fff' }}>Click "Search" to explore intelligence</p>
                </div>
              )}

              <AnimatePresence>
                {selectedNode && (
                  <motion.div
                    initial={{ scale: 0.95, opacity: 0, x: 50 }}
                    animate={{ scale: 1, opacity: 1, x: 0 }}
                    exit={{ scale: 0.95, opacity: 0, x: 50 }}
                    className="detail-pane"
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                            {selectedNode.is_bridge ? <GitBranch size={16} color={THEME.primary} /> : <Fingerprint size={16} color={THEME.secondary} />}
                            <span style={{ fontSize: '0.75rem', textTransform: 'uppercase', color: THEME.primary, fontWeight: 700, letterSpacing: '1px' }}>
                                {selectedNode.is_bridge ? "Foundational Link" : "Core Finding"}
                            </span>
                        </div>
                        <button 
                            onClick={() => setSelectedNode(null)}
                            style={{ background: 'rgba(255,255,255,0.05)', border: 'none', color: '#fff', borderRadius: '50%', width: '32px', height: '32px', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
                        >
                            <X size={18} />
                        </button>
                    </div>

                    <h3 style={{ margin: '0', fontSize: '1.6rem', fontWeight: 700, lineHeight: 1.4, color: '#fff' }}>{selectedNode.title}</h3>
                    
                    <div className="metric-grid">
                      <div className="metric-box">
                         <div className="metric-label">Authority Index</div>
                         <div className="metric-value">{selectedNode.pagerank?.toFixed(4) || 'N/A'}</div>
                      </div>
                      {!selectedNode.is_bridge && (
                        <div className="metric-box">
                            <div className="metric-label">Relevance Score</div>
                            <div className="metric-value">{selectedNode.ltr_score?.toFixed(3)}</div>
                        </div>
                      )}
                    </div>

                    <div style={{ padding: '1.5rem', background: 'rgba(255,255,255,0.02)', borderRadius: '16px', border: '1px solid rgba(255,255,255,0.05)' }}>
                      <div className="metric-label" style={{ marginBottom: '1rem' }}>Expert Synthesis</div>
                      <p className="abstract" style={{ margin: 0 }}>{selectedNode.abstract}</p>
                    </div>

                    <div className="badge-row" style={{ marginTop: 'auto' }}>
                        <div className="badge" style={{ background: THEME.primary + '15', color: THEME.primary }}>
                            <Clock size={14} /> {selectedNode.year || 'N/A'}
                        </div>
                        {!selectedNode.is_bridge && (
                            <div className="badge" style={{ background: THEME.secondary + '15', color: THEME.secondary }}>
                                <Award size={14} /> Ranked #{selectedNode.final_rank || selectedNode.rank_init}
                            </div>
                        )}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
           </div>
        </div>
      </main>

      <style>{`
        .graph-legend-overlay {
            position: absolute;
            top: 2rem;
            right: 2rem;
            display: flex;
            gap: 1rem;
            z-index: 10;
        }
        .legend-chip {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            padding: 0.5rem 1rem;
            border-radius: 100px;
            font-size: 0.75rem;
            color: #fff;
            display: flex;
            align-items: center;
            gap: 0.6rem;
            border: 1px solid rgba(255, 255, 255, 0.08);
        }
        .legend-chip .dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
        }
        .animate-spin {
          animation: spin 1s linear infinite;
        }
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
};

const Badge = ({ icon: Icon, label, color = "#fff" }) => (
  <div className="badge" style={{ color: color }}>
    <Icon size={14} />
    {label}
  </div>
);

export default App;
