import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Search, Loader2, Clock, Award, Star, X, MousePointer2, GitBranch, HelpCircle, Layers, Fingerprint, ChevronDown, ChevronUp, Maximize, Info, ExternalLink, Copy, Check, User, Sparkles, ArrowUpDown, History } from 'lucide-react';
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

const stringToColor = (str) => {
  let hash = 0;
  for (let i = 0; i < str.length; i++) hash = str.charCodeAt(i) + ((hash << 5) - hash);
  const hue = Math.abs(hash) % 360;
  return `hsl(${hue}, 70%, 65%)`;
};

// ─── Search History (localStorage) ───────────────────────────────────
const HISTORY_KEY = 'arxivsense_search_history';
const MAX_HISTORY = 8;

function getSearchHistory() {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY)) || [];
  } catch { return []; }
}

function addToSearchHistory(query) {
  const history = getSearchHistory().filter(q => q.toLowerCase() !== query.toLowerCase());
  history.unshift(query);
  localStorage.setItem(HISTORY_KEY, JSON.stringify(history.slice(0, MAX_HISTORY)));
}

function removeFromHistory(query) {
  const history = getSearchHistory().filter(q => q !== query);
  localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
}

// ─── Author helpers ─────────────────────────────────────────────────
function parseAuthors(authorsStr) {
  if (!authorsStr) return [];
  return authorsStr.split(',').map(a => a.trim()).filter(Boolean);
}

// ─── BibTeX generator ────────────────────────────────────────────────
function generateBibtex(paper) {
  const id = paper.paper_id?.replace(/[^a-zA-Z0-9]/g, '') || 'unknown';
  const year = paper.year || 'n.d.';
  const authors = paper.authors || 'Unknown';
  const title = paper.title || 'Untitled';
  return `@article{${id},
  title={${title}},
  author={${authors}},
  year={${year}},
  journal={arXiv preprint arXiv:${paper.paper_id}}
}`;
}

const App = () => {
  // Search state
  const [searchTerm, setSearchTerm] = useState('');
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState([]);
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  
  // Selection & hover
  const [selectedNodeId, setSelectedNodeId] = useState(null);
  const [hoveredNode, setHoveredNode] = useState(null);

  // Graph controls
  const [showEdges, setShowEdges] = useState(true);
  const [clusterView, setClusterView] = useState(false);
  
  // ─── NEW: Sort by Year ─────────────────────────────────────────
  const [sortMode, setSortMode] = useState('relevance'); // 'relevance' | 'newest' | 'oldest'
  
  // ─── NEW: Search History ───────────────────────────────────────
  const [searchHistory, setSearchHistory] = useState(getSearchHistory());
  
  // ─── NEW: Autocomplete ─────────────────────────────────────────
  const [suggestions, setSuggestions] = useState([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const suggestDebounceRef = useRef(null);
  const searchBarRef = useRef(null);
  
  // ─── NEW: "Did You Mean?" ──────────────────────────────────────
  const [didYouMean, setDidYouMean] = useState(null);
  
  // ─── NEW: Similar Papers ───────────────────────────────────────
  const [similarPapers, setSimilarPapers] = useState([]);
  const [similarLoading, setSimilarLoading] = useState(false);
  const [similarSource, setSimilarSource] = useState('');
  const [showSimilarModal, setShowSimilarModal] = useState(false);
  
  // ─── NEW: Copy toast ──────────────────────────────────────────
  const [copiedId, setCopiedId] = useState(null);
  
  // ─── NEW: Author filter ────────────────────────────────────────
  const [authorFilter, setAuthorFilter] = useState(null);
  
  const fgRef = useRef();
  const listRefs = useRef({});

  // ─── Autocomplete fetch ────────────────────────────────────────
  const fetchSuggestions = useCallback(async (q) => {
    if (!q || q.length < 2) {
      setSuggestions([]);
      setShowSuggestions(false);
      return;
    }
    try {
      const res = await axios.get(`http://localhost:8001/suggest?q=${encodeURIComponent(q)}`);
      setSuggestions(res.data.suggestions || []);
      setShowSuggestions(true);
    } catch {
      setSuggestions([]);
    }
  }, []);

  const handleInputChange = (e) => {
    const val = e.target.value;
    setSearchTerm(val);
    // Debounced autocomplete
    if (suggestDebounceRef.current) clearTimeout(suggestDebounceRef.current);
    suggestDebounceRef.current = setTimeout(() => fetchSuggestions(val), 300);
  };

  // Close suggestions on outside click
  useEffect(() => {
    const handle = (e) => {
      if (searchBarRef.current && !searchBarRef.current.contains(e.target)) {
        setShowSuggestions(false);
      }
    };
    document.addEventListener('mousedown', handle);
    return () => document.removeEventListener('mousedown', handle);
  }, []);

  // ─── Search ────────────────────────────────────────────────────
  const performSearch = async (q) => {
    setLoading(true);
    setResults([]);
    setGraphData({ nodes: [], links: [] });
    setSelectedNodeId(null);
    setDidYouMean(null);
    setAuthorFilter(null);
    setSortMode('relevance');
    setShowSuggestions(false);
    try {
      const response = await axios.post('http://localhost:8001/search', { query: q, top_k: 50 });
      setResults(response.data.results || []);
      setGraphData(response.data.graph || { nodes: [], links: [] });
      // "Did You Mean?"
      if (response.data.suggestion) {
        setDidYouMean(response.data.suggestion);
      }
      // Save to history
      addToSearchHistory(q);
      setSearchHistory(getSearchHistory());
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

  const handleSuggestionClick = (text) => {
    setSearchTerm(text);
    setQuery(text);
    setShowSuggestions(false);
    performSearch(text);
  };

  const handleHistoryClick = (q) => {
    setSearchTerm(q);
    setQuery(q);
    performSearch(q);
  };

  const handleHistoryRemove = (e, q) => {
    e.stopPropagation();
    removeFromHistory(q);
    setSearchHistory(getSearchHistory());
  };

  // ─── Similar Papers ────────────────────────────────────────────
  const fetchSimilar = async (paperId) => {
    setSimilarLoading(true);
    setSimilarPapers([]);
    setShowSimilarModal(true);
    try {
      const res = await axios.post('http://localhost:8001/similar', { paper_id: paperId, top_k: 5 });
      setSimilarPapers(res.data.similar || []);
      setSimilarSource(res.data.source || '');
    } catch (err) {
      console.error('Similar papers error:', err);
    } finally {
      setSimilarLoading(false);
    }
  };

  // ─── Copy citation ─────────────────────────────────────────────
  const handleCopyCitation = (paper) => {
    const bibtex = generateBibtex(paper);
    navigator.clipboard.writeText(bibtex).then(() => {
      setCopiedId(paper.paper_id);
      setTimeout(() => setCopiedId(null), 2000);
    });
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
    const el = listRefs.current[node.id];
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  };

  // ─── Sorted + Filtered results ─────────────────────────────────
  const displayedResults = React.useMemo(() => {
    let items = [...results]; // Start with all 50 results
    
    // Author filter (client-side)
    if (authorFilter) {
      items = items.filter(p => p.authors && p.authors.toLowerCase().includes(authorFilter.toLowerCase()));
    }
    
    // Global Sort (do this BEFORE slicing)
    if (sortMode === 'newest') {
      items.sort((a, b) => (b.year || 0) - (a.year || 0));
    } else if (sortMode === 'oldest') {
      items.sort((a, b) => (a.year || 9999) - (b.year || 9999));
    } else {
      // Relevance (default)
      items.sort((a, b) => (b.final_score || b.ltr_score) - (a.final_score || a.ltr_score));
    }
    
    return items.slice(0, 15); // Finally take the top 15
  }, [results, sortMode, authorFilter]);
  
  const filteredGraphData = React.useMemo(() => {
    const topPaperIds = new Set(displayedResults.map(p => p.paper_id));
    const validNodes = graphData.nodes.filter(n => n.is_bridge || topPaperIds.has(n.id));
    const validNodeIds = new Set(validNodes.map(n => n.id));
    const validLinks = graphData.links.filter(l => 
      validNodeIds.has(typeof l.source === 'object' ? l.source.id : l.source) &&
      validNodeIds.has(typeof l.target === 'object' ? l.target.id : l.target)
    );
    return { nodes: validNodes, links: validLinks };
  }, [graphData, displayedResults]);

  // Truncate author string for display
  const formatAuthors = (authorsStr) => {
    if (!authorsStr) return null;
    const parts = authorsStr.split(',').map(a => a.trim()).filter(Boolean);
    if (parts.length <= 3) return parts.join(', ');
    return `${parts[0]}, ${parts[1]}, ... +${parts.length - 2} more`;
  };

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

          <div className="search-section" ref={searchBarRef}>
            <form onSubmit={handleManualSearch} className="search-bar-wrapper">
               <Search size={20} className="search-icon-inline" />
               <input
                 type="text"
                 value={searchTerm}
                 onChange={handleInputChange}
                 onFocus={() => { if (suggestions.length > 0) setShowSuggestions(true); }}
                 placeholder="Enter research query (e.g. contrastive learning)..."
                 className="search-input"
               />
               <button type="submit" className="search-submit-btn" disabled={loading}>
                  {loading ? <Loader2 size={16} className="animate-spin" /> : <MousePointer2 size={16} />}
                  Search
               </button>
            </form>

            {/* ─── Autocomplete Dropdown ─── */}
            <AnimatePresence>
              {showSuggestions && suggestions.length > 0 && (
                <motion.div
                  className="autocomplete-dropdown"
                  initial={{ opacity: 0, y: -4 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -4 }}
                >
                  {suggestions.map((s, i) => (
                    <div key={i} className="autocomplete-item" onClick={() => handleSuggestionClick(s.text)}>
                      {s.type === 'query' ? <Search size={14} /> : <Layers size={14} />}
                      <span>{s.text}</span>
                      <span className="autocomplete-type">{s.type === 'query' ? 'Topic' : 'Paper'}</span>
                    </div>
                  ))}
                </motion.div>
              )}
            </AnimatePresence>

            {/* ─── Search History Chips ─── */}
            {searchHistory.length > 0 && (
              <div className="history-row">
                <History size={13} color={THEME.textDim} style={{ flexShrink: 0, marginTop: '1px' }} />
                {searchHistory.map((h, i) => (
                  <div key={i} className="history-chip" onClick={() => handleHistoryClick(h)}>
                    {h}
                    <X size={12} className="history-chip-x" onClick={(e) => handleHistoryRemove(e, h)} />
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </header>

      <main className="main-grid">
        <div className="results-column">
          {/* ─── "Did You Mean?" Banner ─── */}
          <AnimatePresence>
            {didYouMean && (
              <motion.div
                className="did-you-mean"
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
              >
                <Sparkles size={16} color={THEME.accent} />
                <span>Did you mean: </span>
                <button className="dym-link" onClick={() => { setSearchTerm(didYouMean); setQuery(didYouMean); performSearch(didYouMean); }}>
                  {didYouMean}
                </button>
                <span>?</span>
                <X size={14} className="dym-close" onClick={() => setDidYouMean(null)} />
              </motion.div>
            )}
          </AnimatePresence>

          <div className="results-header">
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <span style={{ fontSize: '0.85rem', color: THEME.textDim, letterSpacing: '1px', fontWeight: 600 }}>
                RELEVANCE RANKED
              </span>
              {displayedResults.length > 0 && (
                <span style={{ fontSize: '0.8rem', color: THEME.primary, fontWeight: 600 }}>Top {displayedResults.length} Results</span>
              )}
            </div>
            
            {/* ─── Sort by Year Toggle ─── */}
            {results.length > 0 && (
              <div className="sort-pills">
                <button className={`sort-pill ${sortMode === 'relevance' ? 'active' : ''}`} onClick={() => setSortMode('relevance')}>
                  Relevance
                </button>
                <button className={`sort-pill ${sortMode === 'newest' ? 'active' : ''}`} onClick={() => setSortMode('newest')}>
                  Newest
                </button>
                <button className={`sort-pill ${sortMode === 'oldest' ? 'active' : ''}`} onClick={() => setSortMode('oldest')}>
                  Oldest
                </button>
              </div>
            )}
          </div>

          {/* ─── Active Author Filter ─── */}
          {authorFilter && (
            <div className="author-filter-bar">
              <User size={14} />
              <span>Filtered by: <strong>{authorFilter}</strong></span>
              <button className="author-filter-clear" onClick={() => setAuthorFilter(null)}>
                <X size={14} /> Clear
              </button>
            </div>
          )}

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
                const authorList = parseAuthors(paper.authors);
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
                    
                    {/* ─── Authors as clickable chips ─── */}
                    <AuthorChips
                      authors={authorList}
                      onFilter={setAuthorFilter}
                    />

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

                          {/* ─── Action Buttons ─── */}
                          <div className="paper-actions">
                            <a
                              href={`https://arxiv.org/abs/${paper.paper_id}`}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="action-btn"
                              onClick={(e) => e.stopPropagation()}
                            >
                              <ExternalLink size={14} /> View on arXiv
                            </a>
                            <button
                              className="action-btn"
                              onClick={(e) => { e.stopPropagation(); handleCopyCitation(paper); }}
                            >
                              {copiedId === paper.paper_id ? <><Check size={14} /> Copied!</> : <><Copy size={14} /> Copy Citation</>}
                            </button>
                            <button
                              className="action-btn action-btn-accent"
                              onClick={(e) => { e.stopPropagation(); fetchSimilar(paper.paper_id); }}
                            >
                              <Sparkles size={14} /> Find Similar
                            </button>
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
                    linkColor={() => showEdges ? THEME.primary + '30' : 'transparent'}
                    linkWidth={showEdges ? 1.5 : 0}
                    linkDirectionalArrowLength={showEdges ? 4 : 0}
                    linkDirectionalArrowRelPos={1}
                    linkCurvature={0.2}
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

                      const baseRadius = isBridge ? 4 : 6 + (node.pagerank ? node.pagerank * 10 : 0);
                      const radius = Math.min(Math.max(baseRadius, 3), 12);

                      let nodeColor = 'rgba(255,255,255,0.7)';
                      if (clusterView) {
                         const clusterSeed = node.year || (isBridge ? 'bridge' : 'recent');
                         nodeColor = isBridge ? THEME.textDim : stringToColor(String(clusterSeed));
                      } else {
                         nodeColor = isBridge ? THEME.textDim : 'rgba(255,255,255,0.85)';
                      }

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

      {/* ─── Similar Papers Modal ─── */}
      <AnimatePresence>
        {showSimilarModal && (
          <motion.div
            className="modal-overlay"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setShowSimilarModal(false)}
          >
            <motion.div
              className="similar-modal"
              initial={{ opacity: 0, scale: 0.95, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: 20 }}
              onClick={(e) => e.stopPropagation()}
            >
              <div className="similar-modal-header">
                <div>
                  <h3 style={{ margin: 0, fontSize: '1.1rem', fontWeight: 600 }}>
                    <Sparkles size={18} color={THEME.accent} style={{ display: 'inline', marginRight: '8px', verticalAlign: 'middle' }} />
                    Similar Papers
                  </h3>
                  {similarSource && (
                    <p style={{ margin: '0.3rem 0 0', fontSize: '0.8rem', color: THEME.textDim }}>
                      Based on: <em>{similarSource.length > 60 ? similarSource.substring(0, 57) + '...' : similarSource}</em>
                    </p>
                  )}
                </div>
                <button className="modal-close" onClick={() => setShowSimilarModal(false)}>
                  <X size={18} />
                </button>
              </div>
              <div className="similar-modal-body">
                {similarLoading ? (
                  [1,2,3].map(k => (
                    <div key={k} className="similar-card">
                      <div className="skeleton" style={{ width: '90%', height: '18px', marginBottom: '10px' }} />
                      <div className="skeleton" style={{ width: '50%', height: '14px' }} />
                    </div>
                  ))
                ) : similarPapers.length === 0 ? (
                  <p style={{ textAlign: 'center', color: THEME.textDim, padding: '2rem' }}>No similar papers found.</p>
                ) : (
                  similarPapers.map((sp, i) => (
                    <div key={sp.paper_id || i} className="similar-card">
                      <h4 className="similar-title">{sp.title}</h4>
                      {sp.authors && (
                        <p className="similar-authors"><User size={12} /> {formatAuthors(sp.authors)}</p>
                      )}
                      <div className="similar-meta">
                        <Badge icon={Clock} label={sp.year || 'N/A'} />
                        <Badge icon={Star} label={`${(sp.similarity * 100).toFixed(1)}% match`} color={THEME.primary} />
                        <a
                          href={`https://arxiv.org/abs/${sp.paper_id}`}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="action-btn action-btn-sm"
                          onClick={(e) => e.stopPropagation()}
                        >
                          <ExternalLink size={12} /> arXiv
                        </a>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

const Badge = ({ icon: Icon, label, color = "#fff" }) => (
  <div className="badge" style={{ color: color }}>
    <Icon size={12} />
    {label}
  </div>
);

// ─── AuthorChips: Clickable per-author chips with expand/collapse ─────
const AuthorChips = ({ authors, onFilter }) => {
  const [expanded, setExpanded] = React.useState(false);
  const SHOW_LIMIT = 3;

  if (!authors || authors.length === 0) {
    return (
      <div className="author-chips-row">
        <User size={12} style={{ color: 'var(--text-muted)', flexShrink: 0 }} />
        <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>
          Authors not found — re-index required
        </span>
      </div>
    );
  }

  const visible = expanded ? authors : authors.slice(0, SHOW_LIMIT);
  const hidden = authors.length - SHOW_LIMIT;

  return (
    <div className="author-chips-row" onClick={e => e.stopPropagation()}>
      <User size={12} style={{ color: 'var(--text-dim)', flexShrink: 0, marginTop: '2px' }} />
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem', flex: 1 }}>
        {visible.map((author, i) => (
          <button
            key={i}
            className="author-chip"
            onClick={(e) => { e.stopPropagation(); onFilter(author); }}
            title={`Filter by ${author}`}
          >
            {author}
          </button>
        ))}
        {!expanded && hidden > 0 && (
          <button
            className="author-chip author-chip-more"
            onClick={(e) => { e.stopPropagation(); setExpanded(true); }}
          >
            +{hidden} more
          </button>
        )}
        {expanded && hidden > 0 && (
          <button
            className="author-chip author-chip-more"
            onClick={(e) => { e.stopPropagation(); setExpanded(false); }}
          >
            show less
          </button>
        )}
      </div>
    </div>
  );
};

export default App;
