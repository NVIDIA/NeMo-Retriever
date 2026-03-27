/* ===== Pipeline Designer View ===== */

/* ---------- helpers ---------- */
function _uid() { return 'n' + Math.random().toString(36).slice(2, 10); }

const _FALLBACK_PALETTE = ['#ff9f43', 'var(--nv-green)', '#64b4ff', '#e06cff', '#42d6a4', '#ff6b6b', '#a0a0a0', '#22d3ee', '#f472b6'];
const _catColorCache = {};
let _palIdx = 0;

function _categoryColor(cat, explicitColor) {
  if (explicitColor) { _catColorCache[cat] = explicitColor; return explicitColor; }
  if (_catColorCache[cat]) return _catColorCache[cat];
  const c = _FALLBACK_PALETTE[_palIdx % _FALLBACK_PALETTE.length];
  _palIdx++;
  _catColorCache[cat] = c;
  return c;
}

function _resolvedCategoryColor(cat, operators) {
  const sample = operators.find(o => o.category === cat && o.category_color);
  return _categoryColor(cat, sample?.category_color);
}

function _toPythonValue(val) {
  if (val === true || val === 'true' || val === 'True') return 'True';
  if (val === false || val === 'false' || val === 'False') return 'False';
  if (val === null || val === 'null' || val === 'None' || val === 'none' || val === undefined) return 'None';
  if (typeof val === 'number') return String(val);
  if (typeof val === 'string') {
    if (/^-?\d+$/.test(val)) return val;
    if (/^-?\d+\.\d+$/.test(val)) return val;
    return JSON.stringify(val);
  }
  return JSON.stringify(String(val));
}

function _buildKwargsEntries(opParams, nodeCfg) {
  const entries = [];
  opParams.forEach(p => {
    if (p.pydantic && p.fields) {
      const subCfg = nodeCfg[p.name] || {};
      const subEntries = Object.entries(subCfg).filter(([, v]) => v !== '' && v !== undefined);
      if (subEntries.length > 0) {
        const args = subEntries.map(([k, v]) => `${k}=${_toPythonValue(v)}`).join(', ');
        entries.push(`${JSON.stringify(p.name)}: ${p.pydantic_class}(${args})`);
      } else {
        entries.push(`${JSON.stringify(p.name)}: ${p.pydantic_class}()`);
      }
    } else {
      const val = nodeCfg[p.name];
      if (val !== '' && val !== undefined) {
        entries.push(`${JSON.stringify(p.name)}: ${_toPythonValue(val)}`);
      }
    }
  });
  return entries;
}

function _generateCode(nodes, edges) {
  if (nodes.length === 0) return '# Empty graph — add operators to the canvas\n';

  const hasSource = nodes.some(n => n.operator?.type === 'ray_data_source');
  if (hasSource) return _generateRayDataCode(nodes, edges);
  return _generateLegacyCode(nodes, edges);
}

function _generateRayDataCode(nodes, edges) {
  const nodeMap = {};
  nodes.forEach(n => { nodeMap[n.id] = n; });

  const childrenOf = {};
  edges.forEach(e => {
    if (!childrenOf[e.from]) childrenOf[e.from] = [];
    childrenOf[e.from].push(e);
  });

  const roots = new Set(nodes.map(n => n.id));
  edges.forEach(e => roots.delete(e.to));

  if (roots.size !== 1) return '# Error: Ray Data pipeline requires exactly one root node (data source)\n';

  const rootId = Array.from(roots)[0];
  const rootNode = nodeMap[rootId];

  if (rootNode.operator?.type !== 'ray_data_source') {
    return '# Warning: Pipeline should start with a Ray Data source (e.g. ReadBinaryFiles)\n' +
           _generateLegacyCode(nodes, edges);
  }

  if (!_isLinear(rootId, childrenOf)) {
    return '# Error: Ray Data pipeline must be linear (no branching)\n';
  }

  const chain = [];
  let cur = rootId;
  while (cur) {
    chain.push(nodeMap[cur]);
    const children = childrenOf[cur];
    cur = children && children.length === 1 ? children[0].to : null;
  }

  const imports = new Set();
  imports.add('import ray');
  imports.add('import ray.data');
  chain.forEach(n => {
    if (n.operator && n.operator.type !== 'ray_data_source') {
      imports.add(n.operator.import_path);
      (n.operator.params || []).forEach(p => {
        if (p.pydantic && p.pydantic_import) imports.add(p.pydantic_import);
      });
    }
  });

  const lines = [];
  lines.push(Array.from(imports).sort().join('\n'));
  lines.push('');
  lines.push('ray.init(ignore_reinit_error=True)');
  lines.push('');

  const src = chain[0];
  const srcOp = src.operator;
  const srcCfg = src.config || {};
  const srcParams = srcOp.params || [];
  const pathsVal = srcCfg.paths || '';

  const srcArgParts = [JSON.stringify(pathsVal)];
  srcParams.forEach(p => {
    if (p.name === 'paths') return;
    const val = srcCfg[p.name];
    const effectiveVal = (val !== undefined && val !== '') ? val : p.default;
    if (effectiveVal !== undefined && effectiveVal !== '') {
      srcArgParts.push(`${p.name}=${_toPythonValue(effectiveVal)}`);
    }
  });
  lines.push(`# Data source: ${srcOp.class_name}`);
  lines.push(`ds = ${srcOp.ray_fn}(${srcArgParts.join(', ')})`);
  lines.push('');

  const sinkNodes = [];
  const evalNodes = [];

  for (let i = 1; i < chain.length; i++) {
    const n = chain[i];
    const op = n.operator;
    if (!op) continue;

    if (op.type === 'pipeline_sink') { sinkNodes.push(n); continue; }
    if (op.type === 'pipeline_evaluator') { evalNodes.push(n); continue; }

    const isGpu = op.compute === 'gpu';

    const kwargsEntries = _buildKwargsEntries(op.params || [], n.config || {});

    let kwargsStr = '{}';
    if (kwargsEntries.length > 0) {
      kwargsStr = '{' + kwargsEntries.join(', ') + '}';
    }

    lines.push(`# Stage: ${op.display_name || op.class_name}`);
    lines.push(`ds = ds.map_batches(`);
    lines.push(`    ${op.class_name},`);
    lines.push(`    fn_constructor_kwargs=${kwargsStr},`);
    lines.push(`    batch_size=1,`);
    lines.push(`    batch_format="pandas",`);
    if (isGpu) {
      lines.push(`    num_cpus=0,`);
      lines.push(`    num_gpus=1,`);
    }
    lines.push(`)`);
    lines.push('');
  }

  sinkNodes.forEach(sn => {
    const op = sn.operator;
    if (!op) return;
    imports.add(op.import_path);

    const kwargsEntries = _buildKwargsEntries(op.params || [], sn.config || {});
    let kwargsStr = '{}';
    if (kwargsEntries.length > 0) {
      kwargsStr = '{' + kwargsEntries.join(', ') + '}';
    }

    lines.push(`# Sink: ${op.display_name || op.class_name}`);
    lines.push(`ds = ds.map_batches(`);
    lines.push(`    ${op.class_name},`);
    lines.push(`    fn_constructor_kwargs=${kwargsStr},`);
    lines.push(`    batch_size=1,`);
    lines.push(`    batch_format="pandas",`);
    lines.push(`)`);
    lines.push('');
  });

  lines.push('result = ds.materialize()');
  lines.push('');

  evalNodes.forEach(en => {
    const op = en.operator;
    if (!op) return;
    imports.add(op.import_path);

    const kwargsEntries = _buildKwargsEntries(op.params || [], en.config || {});
    let kwargsStr = kwargsEntries.length > 0 ? '{' + kwargsEntries.join(', ') + '}' : '{}';

    const varName = en.varName || '_evaluator';
    lines.push(`# Evaluation: ${op.display_name || op.class_name}`);
    lines.push(`${varName} = ${op.class_name}(**${kwargsStr})`);
    lines.push(`_eval_summary = ${varName}.evaluate()`);
    lines.push('');
  });

  const importBlock = Array.from(imports).sort().join('\n');
  lines[0] = importBlock;

  return lines.join('\n');
}

function _generateLegacyCode(nodes, edges) {
  const imports = new Set();
  imports.add('from nemo_retriever.graph import Graph, Node');

  const nodeMap = {};
  nodes.forEach(n => { nodeMap[n.id] = n; });

  nodes.forEach(n => {
    if (n.operator) {
      imports.add(n.operator.import_path);
      (n.operator.params || []).forEach(p => {
        if (p.pydantic && p.pydantic_import) imports.add(p.pydantic_import);
      });
    }
  });

  const lines = [];
  lines.push(Array.from(imports).sort().join('\n'));
  lines.push('');

  nodes.forEach(n => {
    const op = n.operator;
    if (!op) return;
    const opParams = op.params || [];
    const hasPydantic = opParams.some(p => p.pydantic);

    if (hasPydantic) {
      const argParts = [];
      opParams.forEach(p => {
        if (p.pydantic && p.fields) {
          const subCfg = (n.config || {})[p.name] || {};
          const subEntries = Object.entries(subCfg).filter(([, v]) => v !== '' && v !== undefined);
          const constructorArgs = subEntries.map(([k, v]) => `${k}=${_toPythonValue(v)}`).join(', ');
          argParts.push(`${p.name}=${p.pydantic_class}(${constructorArgs})`);
        } else {
          const val = (n.config || {})[p.name];
          if (val !== '' && val !== undefined) {
            argParts.push(`${p.name}=${_toPythonValue(val)}`);
          }
        }
      });
      lines.push(`${n.varName} = ${op.class_name}(${argParts.join(', ')})`);
    } else {
      const cfgEntries = Object.entries(n.config || {}).filter(([, v]) => v !== '' && v !== undefined);
      let argStr = '';
      if (cfgEntries.length > 0) {
        argStr = cfgEntries.map(([k, v]) => {
          const strVal = typeof v === 'string' ? JSON.stringify(v) : String(v);
          return `${k}=${strVal}`;
        }).join(', ');
      }
      lines.push(`${n.varName} = ${op.class_name}(${argStr})`);
    }
  });

  lines.push('');

  const roots = new Set(nodes.map(n => n.id));
  edges.forEach(e => roots.delete(e.to));

  const childrenOf = {};
  edges.forEach(e => {
    if (!childrenOf[e.from]) childrenOf[e.from] = [];
    childrenOf[e.from].push(e);
  });

  if (roots.size === 1 && _isLinear(Array.from(roots)[0], childrenOf)) {
    const chain = [];
    let cur = Array.from(roots)[0];
    while (cur) {
      chain.push(nodeMap[cur].varName);
      const children = childrenOf[cur];
      cur = children && children.length === 1 ? children[0].to : null;
    }
    lines.push('graph = (');
    lines.push('    ' + chain.join('\n    >> '));
    lines.push(')');
  } else {
    lines.push('graph = Graph()');
    roots.forEach(rootId => {
      lines.push(`graph.add_root(${nodeMap[rootId].varName})`);
    });
    edges.forEach(e => {
      lines.push(`${nodeMap[e.from].varName} >> ${nodeMap[e.to].varName}`);
    });
  }

  const hasEdgeConfig = edges.some(e => Object.keys(e.config || {}).length > 0);
  if (hasEdgeConfig) {
    lines.push('');
    lines.push('# Connection arguments (for reference):');
    edges.forEach(e => {
      const cfg = e.config || {};
      const entries = Object.entries(cfg).filter(([,v]) => v !== '');
      if (entries.length > 0) {
        lines.push(`# ${nodeMap[e.from]?.varName} -> ${nodeMap[e.to]?.varName}: ${JSON.stringify(Object.fromEntries(entries))}`);
      }
    });
  }

  lines.push('');
  return lines.join('\n');
}

function _isLinear(rootId, childrenOf) {
  let cur = rootId;
  const visited = new Set();
  while (cur) {
    if (visited.has(cur)) return false;
    visited.add(cur);
    const ch = childrenOf[cur];
    if (!ch || ch.length === 0) return true;
    if (ch.length > 1) return false;
    cur = ch[0].to;
  }
  return true;
}

function _toVarName(className) {
  return className.replace(/([A-Z])/g, '_$1').replace(/^_/, '').toLowerCase();
}

/* ---------- Operator Palette ---------- */
function OperatorPalette({ operators, onDragStart, filter, setFilter }) {
  const filtered = operators.filter(op => {
    if (op.hidden) return false;
    if (!filter) return true;
    const q = filter.toLowerCase();
    return (op.display_name || op.class_name).toLowerCase().includes(q) || op.class_name.toLowerCase().includes(q);
  });

  const cats = useMemo(() => {
    const priority = ['Data Sources', 'Document Processing', 'Text & Content', 'Detection & OCR', 'Embeddings & Ranking', 'Audio', 'Data Sinks', 'Evaluation', 'Graph Utilities'];
    const seen = new Set();
    operators.forEach(op => seen.add(op.category));
    const ordered = priority.filter(c => seen.has(c));
    seen.forEach(c => { if (!ordered.includes(c)) ordered.push(c); });
    return ordered;
  }, [operators]);

  return (
    <div style={{width:'220px',borderRight:'1px solid var(--nv-border)',display:'flex',flexDirection:'column',overflow:'hidden',flexShrink:0}}>
      <div style={{padding:'12px',borderBottom:'1px solid var(--nv-border)'}}>
        <div style={{fontSize:'12px',fontWeight:700,color:'#fff',marginBottom:'8px'}}>Components</div>
        <input className="input" placeholder="Filter…" value={filter} onChange={e=>setFilter(e.target.value)}
          style={{width:'100%',fontSize:'11px',padding:'5px 8px'}} />
      </div>
      <div style={{flex:1,overflow:'auto',padding:'8px'}}>
        {cats.map(cat => {
          const ops = filtered.filter(o => o.category === cat);
          if (ops.length === 0) return null;
          const catColor = _resolvedCategoryColor(cat, operators);
          return (
            <div key={cat} style={{marginBottom:'14px'}}>
              <div style={{fontSize:'10px',fontWeight:700,textTransform:'uppercase',color:catColor,marginBottom:'6px',letterSpacing:'0.05em'}}>
                {cat}
              </div>
              {ops.map(op => {
                const displayLabel = op.display_name || op.class_name;
                const isSpecial = op.type === 'ray_data_source' || op.type === 'pipeline_sink' || op.type === 'pipeline_evaluator';
                return (
                  <div key={op.class_name + op.module} draggable
                    onDragStart={e => { e.dataTransfer.setData('application/json', JSON.stringify(op)); onDragStart && onDragStart(op); }}
                    style={{
                      padding:'6px 8px',marginBottom:'3px',borderRadius:'6px',cursor:'grab',
                      background:'rgba(255,255,255,0.03)',border:'1px solid var(--nv-border)',
                      fontSize:'11px',color:'#fff',fontWeight:500,
                      display:'flex',alignItems:'center',gap:'6px',
                    }}
                    title={op.description || (op.type === 'ray_data_source' ? `ray.data · ${op.ray_fn}` : `${op.module}.${op.class_name}`)}>
                    <span style={{width:'6px',height:'6px',borderRadius: isSpecial ? '2px' : '50%',background:catColor,flexShrink:0}}></span>
                    {displayLabel}
                  </div>
                );
              })}
            </div>
          );
        })}
        {filtered.length === 0 && (
          <div style={{textAlign:'center',color:'var(--nv-text-dim)',fontSize:'12px',padding:'20px 0'}}>No components found</div>
        )}
      </div>
    </div>
  );
}

/* ---------- Canvas Node ---------- */
function CanvasNode({ node, selected, onMouseDown, onSelect, onDelete }) {
  const color = node.operator ? _categoryColor(node.operator.category, node.operator.category_color) : 'var(--nv-text-dim)';
  const isSource = node.operator?.type === 'ray_data_source';
  const isSink = node.operator?.type === 'pipeline_sink';
  const isEval = node.operator?.type === 'pipeline_evaluator';
  const isSpecial = isSource || isSink || isEval;
  const nodeHeight = isSpecial ? 66 : 56;
  const typeLabel = isSource ? 'SOURCE' : isSink ? 'SINK' : isEval ? 'EVAL' : null;
  const displayName = node.operator?.display_name || node.operator?.class_name || 'Unknown';
  return (
    <g transform={`translate(${node.x},${node.y})`}
      onMouseDown={e => { e.stopPropagation(); onMouseDown(e, node.id); }}
      onClick={e => { e.stopPropagation(); onSelect(node.id); }}
      style={{cursor:'grab'}}>
      <rect x="0" y="0" width="180" height={nodeHeight} rx="8"
        fill={selected ? 'rgba(255,255,255,0.08)' : 'rgba(255,255,255,0.04)'}
        stroke={selected ? color : 'var(--nv-border)'} strokeWidth={selected ? 2 : 1}
        strokeDasharray={isSpecial ? '4 2' : undefined} />
      {!isSource && <circle cx="0" cy={nodeHeight/2} r="7" fill="var(--nv-bg)" stroke={color} strokeWidth="2" pointerEvents="none" />}
      <circle cx="180" cy={nodeHeight/2} r="7" fill="var(--nv-bg)" stroke={color} strokeWidth="2" pointerEvents="none" />
      <rect x="6" y="6" width="4" height={nodeHeight - 12} rx="2" fill={color} opacity="0.6" />
      {typeLabel && <text x="170" y="12" fill={color} fontSize="8" fontWeight="700" textAnchor="end" fontFamily="inherit">{typeLabel}</text>}
      <text x="18" y={isSpecial ? 24 : 22} fill="#fff" fontSize="12" fontWeight="600" fontFamily="inherit">{displayName}</text>
      <text x="18" y={isSource ? 40 : 38} fill="var(--nv-text-dim)" fontSize="10" fontFamily="inherit">{node.varName}</text>
      {isSource && node.config?.paths && (
        <text x="18" y="54" fill="var(--nv-text-dim)" fontSize="8" fontFamily="inherit" opacity="0.6">
          {node.config.paths.length > 24 ? '\u2026' + node.config.paths.slice(-24) : node.config.paths}
        </text>
      )}
      <text x="170" y={isSource ? 24 : 14} fill="var(--nv-text-dim)" fontSize="10" textAnchor="end" style={{cursor:'pointer'}}
        onClick={e => { e.stopPropagation(); onDelete(node.id); }}>✕</text>
    </g>
  );
}

/* ---------- Edge Arrow ---------- */
function _nodeHeight(n) {
  const t = n.operator?.type;
  return (t === 'ray_data_source' || t === 'pipeline_sink' || t === 'pipeline_evaluator') ? 66 : 56;
}

function EdgeArrow({ edge, nodes, selected, onSelect }) {
  const fromNode = nodes.find(n => n.id === edge.from);
  const toNode = nodes.find(n => n.id === edge.to);
  if (!fromNode || !toNode) return null;

  const x1 = fromNode.x + 180;
  const y1 = fromNode.y + _nodeHeight(fromNode) / 2;
  const x2 = toNode.x;
  const y2 = toNode.y + _nodeHeight(toNode) / 2;
  const mx = (x1 + x2) / 2;

  const hasConfig = Object.keys(edge.config || {}).some(k => edge.config[k] !== '');
  const strokeColor = selected ? '#fff' : hasConfig ? '#64b4ff' : 'var(--nv-text-dim)';

  return (
    <g onClick={e => { e.stopPropagation(); onSelect(edge.id); }} style={{cursor:'pointer'}}>
      <path d={`M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`}
        stroke={strokeColor} strokeWidth={selected ? 2.5 : 1.5} fill="none" markerEnd="url(#arrowhead)" />
      {hasConfig && (
        <circle cx={mx} cy={(y1+y2)/2} r="5" fill="#64b4ff" stroke="var(--nv-bg)" strokeWidth="1" />
      )}
    </g>
  );
}

/* ---------- Node Config Panel ---------- */
function _defaultLabel(field) {
  if (field.required) return null;
  if (field.default === undefined) return null;
  if (field.default === null) return 'None';
  return String(field.default);
}

function _PydanticFieldInput({ paramName, field, value, onChange }) {
  const hasDefault = field.default !== undefined;
  const defaultStr = hasDefault ? (field.default === null ? '' : String(field.default)) : '';
  const displayVal = value !== undefined ? value : defaultStr;
  const isModified = value !== undefined && value !== defaultStr;
  const defaultHint = _defaultLabel(field);

  return (
    <div style={{marginBottom:'8px'}}>
      <label style={{display:'flex',alignItems:'baseline',gap:'4px',fontSize:'11px',color:'#fff',fontWeight:500,marginBottom:'3px'}}>
        <span>{field.name}</span>
        {field.required && <span style={{color:'#ff9f43',fontSize:'9px',fontWeight:700}}>REQ</span>}
        {field.type && <span style={{color:'var(--nv-text-dim)',fontWeight:400,fontSize:'9px',marginLeft:'auto'}}>{field.type}</span>}
      </label>
      <input className="input" value={displayVal}
        onChange={e => onChange(paramName, field.name, e.target.value)}
        placeholder={field.default === null ? 'None' : (field.required ? 'required' : '')}
        style={{width:'100%',fontSize:'11px',fontFamily:'var(--font-mono)',padding:'4px 8px',
          borderColor: field.required && !value ? 'rgba(255,159,67,0.4)' : undefined,
          color: isModified ? '#fff' : 'var(--nv-text-muted)'}} />
      {defaultHint !== null && (
        <div style={{fontSize:'9px',color:'var(--nv-text-dim)',marginTop:'2px',fontFamily:'var(--font-mono)'}}>
          default: {defaultHint}
        </div>
      )}
    </div>
  );
}

function NodeConfigPanel({ node, onUpdate, onClose }) {
  const params = node.operator?.params || [];
  const [cfg, setCfg] = useState({...(node.config || {})});
  const [varName, setVarName] = useState(node.varName);

  function handleSave() {
    onUpdate(node.id, { config: cfg, varName });
    onClose();
  }

  function handlePydanticField(paramName, fieldName, value) {
    const subCfg = { ...(cfg[paramName] || {}) };
    if (value === '' || value === undefined) {
      delete subCfg[fieldName];
    } else {
      subCfg[fieldName] = value;
    }
    setCfg({ ...cfg, [paramName]: subCfg });
  }

  const scalarParams = params.filter(p => !p.pydantic);
  const pydanticParams = params.filter(p => p.pydantic && p.fields);

  return (
    <div style={{width:'300px',borderLeft:'1px solid var(--nv-border)',display:'flex',flexDirection:'column',overflow:'hidden',flexShrink:0,background:'var(--nv-surface)'}}>
      <div style={{padding:'12px',borderBottom:'1px solid var(--nv-border)',display:'flex',justifyContent:'space-between',alignItems:'center'}}>
        <div>
          <div style={{fontSize:'12px',fontWeight:700,color:'#fff'}}>{node.operator?.display_name || node.operator?.class_name}</div>
          <div style={{fontSize:'10px',color:'var(--nv-text-dim)',marginTop:'2px'}}>{node.operator?.class_name} · {node.operator?.module}</div>
        </div>
        <button className="btn btn-ghost btn-icon" onClick={onClose} style={{borderRadius:'50%',padding:'2px'}}><IconX /></button>
      </div>
      <div style={{flex:1,overflow:'auto',padding:'12px'}}>
        <div style={{marginBottom:'14px'}}>
          <label style={{display:'block',fontSize:'10px',fontWeight:600,color:'var(--nv-text-dim)',textTransform:'uppercase',marginBottom:'4px'}}>Variable Name</label>
          <input className="input" value={varName} onChange={e => setVarName(e.target.value)}
            style={{width:'100%',fontSize:'12px',fontFamily:'var(--font-mono)',padding:'5px 8px'}} />
        </div>

        {scalarParams.length > 0 && (
          <>
            <div style={{fontSize:'10px',fontWeight:700,color:'var(--nv-text-dim)',textTransform:'uppercase',marginBottom:'8px'}}>Parameters</div>
            {scalarParams.filter(p => !p.hidden).map(p => {
              const hasDefault = p.default !== undefined;
              const defaultStr = hasDefault ? (p.default === null ? '' : String(p.default)) : '';
              const curVal = cfg[p.name] !== undefined ? cfg[p.name] : defaultStr;
              const displayLabel = p.label || p.name;
              const hasChoices = p.choices && p.choices.length > 0;
              const isBool = (p.type && /bool/i.test(p.type)) || p.default === true || p.default === false;
              return (
                <div key={p.name} style={{marginBottom:'10px'}}>
                  <label style={{display:'block',fontSize:'11px',color:'#fff',fontWeight:500,marginBottom:'3px'}}>
                    {displayLabel}
                    {p.type && <span style={{color:'var(--nv-text-dim)',fontWeight:400,marginLeft:'6px',fontSize:'10px'}}>{p.type}</span>}
                  </label>
                  {p.description && (
                    <div style={{fontSize:'9px',color:'var(--nv-text-dim)',marginBottom:'3px'}}>{p.description}</div>
                  )}
                  {hasChoices ? (
                    <select className="input" value={curVal}
                      onChange={e => setCfg({...cfg, [p.name]: e.target.value})}
                      style={{width:'100%',fontSize:'11px',padding:'4px 8px'}}>
                      {!curVal && <option value="">-- select --</option>}
                      {p.choices.map(c => <option key={c} value={String(c)}>{String(c)}</option>)}
                    </select>
                  ) : isBool ? (
                    <select className="input" value={String(curVal === true || curVal === 'true' || curVal === 'True' ? 'True' : 'False')}
                      onChange={e => setCfg({...cfg, [p.name]: e.target.value})}
                      style={{width:'100%',fontSize:'11px',padding:'4px 8px'}}>
                      <option value="True">True</option>
                      <option value="False">False</option>
                    </select>
                  ) : (
                    <input className="input" value={curVal}
                      onChange={e => setCfg({...cfg, [p.name]: e.target.value})}
                      placeholder={p.placeholder || (p.default === null ? 'None' : (hasDefault ? defaultStr : ''))}
                      style={{width:'100%',fontSize:'11px',fontFamily:'var(--font-mono)',padding:'4px 8px'}} />
                  )}
                  {hasDefault && (
                    <div style={{fontSize:'9px',color:'var(--nv-text-dim)',marginTop:'2px',fontFamily:'var(--font-mono)'}}>
                      default: {p.default === null ? 'None' : String(p.default)}
                    </div>
                  )}
                </div>
              );
            })}
          </>
        )}

        {pydanticParams.map(p => {
          const subCfg = cfg[p.name] || {};
          return (
            <div key={p.name} style={{marginBottom:'16px'}}>
              <div style={{
                fontSize:'10px',fontWeight:700,textTransform:'uppercase',letterSpacing:'0.04em',
                marginBottom:'10px',paddingBottom:'6px',borderBottom:'1px solid var(--nv-border)',
                display:'flex',alignItems:'center',gap:'6px',
              }}>
                <span style={{color:'var(--nv-green)'}}>{p.pydantic_class}</span>
                <span style={{color:'var(--nv-text-dim)',fontWeight:400,fontSize:'9px'}}>({p.name})</span>
              </div>
              {p.fields.map(f => (
                <_PydanticFieldInput key={f.name} paramName={p.name} field={f}
                  value={subCfg[f.name]} onChange={handlePydanticField} />
              ))}
            </div>
          );
        })}

        {params.length === 0 && (
          <div style={{color:'var(--nv-text-dim)',fontSize:'12px',fontStyle:'italic',padding:'12px 0'}}>No configurable parameters</div>
        )}
      </div>
      <div style={{padding:'12px',borderTop:'1px solid var(--nv-border)'}}>
        <button className="btn btn-primary" onClick={handleSave} style={{width:'100%',justifyContent:'center',fontSize:'12px'}}>Apply</button>
      </div>
    </div>
  );
}

/* ---------- Edge Config Panel ---------- */
function EdgeConfigPanel({ edge, nodes, onUpdate, onClose }) {
  const fromNode = nodes.find(n => n.id === edge.from);
  const toNode = nodes.find(n => n.id === edge.to);
  const [pairs, setPairs] = useState(() => {
    const entries = Object.entries(edge.config || {});
    return entries.length > 0 ? entries.map(([k, v]) => ({ k, v })) : [{ k: '', v: '' }];
  });

  function addRow() { setPairs([...pairs, { k: '', v: '' }]); }
  function updateRow(i, field, val) { const np = [...pairs]; np[i] = {...np[i], [field]: val}; setPairs(np); }
  function removeRow(i) { setPairs(pairs.filter((_, idx) => idx !== i)); }

  function handleSave() {
    const cfg = {};
    pairs.forEach(({ k, v }) => { if (k.trim()) cfg[k.trim()] = v; });
    onUpdate(edge.id, { config: cfg });
    onClose();
  }

  return (
    <div style={{width:'280px',borderLeft:'1px solid var(--nv-border)',display:'flex',flexDirection:'column',overflow:'hidden',flexShrink:0,background:'var(--nv-surface)'}}>
      <div style={{padding:'12px',borderBottom:'1px solid var(--nv-border)',display:'flex',justifyContent:'space-between',alignItems:'center'}}>
        <div>
          <div style={{fontSize:'12px',fontWeight:700,color:'#fff'}}>Connection</div>
          <div style={{fontSize:'10px',color:'var(--nv-text-dim)',marginTop:'2px'}}>
            {fromNode?.operator?.class_name || '?'} → {toNode?.operator?.class_name || '?'}
          </div>
        </div>
        <button className="btn btn-ghost btn-icon" onClick={onClose} style={{borderRadius:'50%',padding:'2px'}}><IconX /></button>
      </div>
      <div style={{flex:1,overflow:'auto',padding:'12px'}}>
        <div style={{fontSize:'10px',fontWeight:700,color:'var(--nv-text-dim)',textTransform:'uppercase',marginBottom:'8px'}}>Key / Value Arguments</div>
        {pairs.map((row, i) => (
          <div key={i} style={{display:'flex',gap:'4px',marginBottom:'6px',alignItems:'center'}}>
            <input className="input" value={row.k} onChange={e => updateRow(i, 'k', e.target.value)}
              placeholder="key" style={{flex:1,fontSize:'11px',fontFamily:'var(--font-mono)',padding:'4px 6px'}} />
            <input className="input" value={row.v} onChange={e => updateRow(i, 'v', e.target.value)}
              placeholder="value" style={{flex:1,fontSize:'11px',fontFamily:'var(--font-mono)',padding:'4px 6px'}} />
            <button className="btn btn-ghost btn-icon" onClick={() => removeRow(i)} style={{padding:'2px',fontSize:'10px',color:'var(--nv-text-dim)'}}>✕</button>
          </div>
        ))}
        <button className="btn btn-secondary" onClick={addRow} style={{fontSize:'11px',padding:'3px 10px',marginTop:'4px'}}>+ Add</button>
      </div>
      <div style={{padding:'12px',borderTop:'1px solid var(--nv-border)'}}>
        <button className="btn btn-primary" onClick={handleSave} style={{width:'100%',justifyContent:'center',fontSize:'12px'}}>Apply</button>
      </div>
    </div>
  );
}

/* ---------- Code Preview Modal ---------- */
function CodePreviewModal({ code, onClose, onSave, saving }) {
  const [copied, setCopied] = useState(false);
  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" style={{maxWidth:'720px',maxHeight:'85vh',overflow:'hidden',display:'flex',flexDirection:'column'}} onClick={e=>e.stopPropagation()}>
        <div className="modal-head">
          <h2 style={{fontSize:'16px',fontWeight:700,color:'#fff'}}>Generated Graph Code</h2>
          <button className="btn btn-ghost btn-icon" onClick={onClose} style={{borderRadius:'50%'}}><IconX /></button>
        </div>
        <div style={{flex:1,overflow:'auto',padding:'0'}}>
          <pre className="mono" style={{
            fontSize:'12px',padding:'20px',margin:0,lineHeight:'1.7',
            background:'var(--nv-bg)',color:'var(--nv-text-muted)',whiteSpace:'pre-wrap',wordBreak:'break-all',
          }}>{code}</pre>
        </div>
        <div className="modal-foot" style={{display:'flex',gap:'8px'}}>
          <button className="btn btn-secondary" onClick={() => { navigator.clipboard.writeText(code); setCopied(true); setTimeout(()=>setCopied(false),2000); }}>
            {copied ? 'Copied!' : 'Copy to Clipboard'}
          </button>
          {onSave && (
            <button className="btn btn-primary" onClick={onSave} disabled={saving} style={{flex:1,justifyContent:'center'}}>
              {saving ? <><span className="spinner" style={{marginRight:'6px'}}></span>Saving…</> : 'Save Graph'}
            </button>
          )}
          <button className="btn btn-secondary" onClick={onClose}>Close</button>
        </div>
      </div>
    </div>
  );
}

/* ---------- Save / Name Modal ---------- */
function SaveGraphModal({ name, setName, description, setDescription, onSave, onClose, saving, isUpdate }) {
  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" style={{maxWidth:'420px'}} onClick={e=>e.stopPropagation()}>
        <div className="modal-head">
          <h2 style={{fontSize:'16px',fontWeight:700,color:'#fff'}}>{isUpdate ? 'Update Graph' : 'Save Graph'}</h2>
          <button className="btn btn-ghost btn-icon" onClick={onClose} style={{borderRadius:'50%'}}><IconX /></button>
        </div>
        <div style={{padding:'20px'}}>
          <div style={{marginBottom:'14px'}}>
            <label style={{display:'block',fontSize:'11px',fontWeight:600,color:'var(--nv-text-dim)',textTransform:'uppercase',marginBottom:'5px'}}>Name</label>
            <input className="input" value={name} onChange={e=>setName(e.target.value)} style={{width:'100%'}} placeholder="My Pipeline" />
          </div>
          <div>
            <label style={{display:'block',fontSize:'11px',fontWeight:600,color:'var(--nv-text-dim)',textTransform:'uppercase',marginBottom:'5px'}}>Description</label>
            <textarea className="input" value={description} onChange={e=>setDescription(e.target.value)} rows={3}
              style={{width:'100%',resize:'vertical',fontFamily:'inherit'}} placeholder="Optional description…" />
          </div>
        </div>
        <div className="modal-foot">
          <button className="btn btn-secondary" onClick={onClose}>Cancel</button>
          <button className="btn btn-primary" onClick={onSave} disabled={saving || !name.trim()} style={{flex:1,justifyContent:'center'}}>
            {saving ? <><span className="spinner" style={{marginRight:'6px'}}></span>Saving…</> : isUpdate ? 'Update' : 'Save'}
          </button>
        </div>
      </div>
    </div>
  );
}

/* ---------- Run Graph Modal ---------- */
function RunGraphModal({ graphId, graphName, onClose }) {
  const [runners, setRunners] = useState([]);
  const [runnerId, setRunnerId] = useState('');
  const [rayAddress, setRayAddress] = useState('');
  const [gitRef, setGitRef] = useState('');
  const [gitCommit, setGitCommit] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch('/api/runners').then(r => r.json()).then(list => {
      const online = (list || []).filter(r => r.status === 'online' || r.status === 'idle');
      setRunners(online.length > 0 ? online : list || []);
    }).catch(() => {});
  }, []);

  async function handleRun() {
    setSubmitting(true);
    setError(null);
    setResult(null);
    try {
      const body = {};
      if (runnerId) body.runner_id = parseInt(runnerId, 10);
      if (rayAddress.trim()) body.ray_address = rayAddress.trim();
      if (gitRef.trim()) body.git_ref = gitRef.trim();
      if (gitCommit.trim()) body.git_commit = gitCommit.trim();

      const res = await fetch(`/api/graphs/${graphId}/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setSubmitting(false);
    }
  }

  const fieldLabel = { display: 'block', fontSize: '11px', fontWeight: 600, color: 'var(--nv-text-dim)', textTransform: 'uppercase', marginBottom: '5px' };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" style={{ maxWidth: '520px' }} onClick={e => e.stopPropagation()}>
        <div className="modal-head">
          <span>Run Graph: {graphName}</span>
          <button className="modal-close" onClick={onClose}>&times;</button>
        </div>
        <div className="modal-body" style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
          {result ? (
            <div style={{ textAlign: 'center', padding: '16px 0' }}>
              <div style={{ fontSize: '14px', color: 'var(--nv-green)', fontWeight: 600, marginBottom: '8px' }}>Job submitted successfully</div>
              <div style={{ fontSize: '12px', color: 'var(--nv-text-dim)', marginBottom: '12px' }}>
                Job ID: <code style={{ color: '#fff', background: 'rgba(255,255,255,0.06)', padding: '2px 6px', borderRadius: '4px' }}>{result.job_id}</code>
              </div>
              <div style={{ marginTop: '12px' }}>
                <a href="#runs" onClick={() => { window.location.hash = 'runs'; onClose(); }}
                  style={{ fontSize: '13px', color: 'var(--nv-green)', textDecoration: 'none', fontWeight: 600,
                    display: 'inline-flex', alignItems: 'center', gap: '6px', padding: '8px 16px',
                    borderRadius: '6px', background: 'rgba(118,185,0,0.08)', border: '1px solid rgba(118,185,0,0.2)',
                    cursor: 'pointer' }}>
                  View in Runs &rarr;
                </a>
              </div>
            </div>
          ) : (
            <>
              <div style={{ padding: '10px 12px', borderRadius: '6px', background: 'rgba(255,159,67,0.08)', border: '1px solid rgba(255,159,67,0.2)', fontSize: '12px', color: '#ff9f43' }}>
                Data source, sinks (LanceDB Writer), and evaluation (Recall / BEIR) are configured as components in the graph itself.
              </div>
              <div style={{ display: 'flex', gap: '10px' }}>
                <div style={{ flex: 1 }}>
                  <label style={fieldLabel}>Runner (optional)</label>
                  <select className="input" value={runnerId} onChange={e => setRunnerId(e.target.value)} style={{ width: '100%' }}>
                    <option value="">Any available</option>
                    {runners.map(r => (
                      <option key={r.id} value={r.id}>{r.name || `Runner ${r.id}`} ({r.status})</option>
                    ))}
                  </select>
                </div>
                <div style={{ flex: 1 }}>
                  <label style={fieldLabel}>Ray Address (optional)</label>
                  <input className="input" value={rayAddress} onChange={e => setRayAddress(e.target.value)}
                    style={{ width: '100%' }} placeholder="auto" />
                </div>
              </div>
              <div style={{ display: 'flex', gap: '10px' }}>
                <div style={{ flex: 1 }}>
                  <label style={fieldLabel}>Git Ref (optional)</label>
                  <input className="input" value={gitRef} onChange={e => setGitRef(e.target.value)}
                    style={{ width: '100%' }} placeholder="origin/main" />
                </div>
                <div style={{ flex: 1 }}>
                  <label style={fieldLabel}>Git Commit SHA (optional)</label>
                  <input className="input" value={gitCommit} onChange={e => setGitCommit(e.target.value)}
                    style={{ width: '100%' }} placeholder="abc123…" />
                </div>
              </div>
              {error && (
                <div style={{ padding: '8px 12px', borderRadius: '6px', background: 'rgba(255,80,80,0.1)', border: '1px solid rgba(255,80,80,0.2)', color: '#ff5050', fontSize: '12px' }}>
                  {error}
                </div>
              )}
            </>
          )}
        </div>
        <div className="modal-foot">
          <button className="btn btn-secondary" onClick={onClose}>{result ? 'Close' : 'Cancel'}</button>
          {!result && (
            <button className="btn btn-primary" onClick={handleRun} disabled={submitting} style={{ flex: 1, justifyContent: 'center' }}>
              {submitting ? <><span className="spinner" style={{ marginRight: '6px' }}></span>Submitting…</> : 'Run on Runner'}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

/* ========== Main Designer View ========== */
function DesignerView() {
  const [operators, setOperators] = useState([]);
  const [graphs, setGraphs] = useState([]);
  const [graphsLoading, setGraphsLoading] = useState(true);
  const [paletteFilter, setPaletteFilter] = useState('');

  const [nodes, setNodes] = useState([]);
  const [edges, setEdges] = useState([]);
  const [selectedNodeId, setSelectedNodeId] = useState(null);
  const [selectedEdgeId, setSelectedEdgeId] = useState(null);
  const [activeGraphId, setActiveGraphId] = useState(null);
  const [graphName, setGraphName] = useState('');
  const [graphDescription, setGraphDescription] = useState('');

  const [showCode, setShowCode] = useState(false);
  const [showSaveModal, setShowSaveModal] = useState(false);
  const [showRunModal, setShowRunModal] = useState(false);
  const [saving, setSaving] = useState(false);
  const [showTable, setShowTable] = useState(true);

  const [connecting, setConnecting] = useState(null);
  const [dragNode, setDragNode] = useState(null);
  const [dragOffset, setDragOffset] = useState({x:0,y:0});
  const dragStartPos = useRef(null);
  const isDragging = useRef(false);
  const nodeInteractionRef = useRef(false);

  const svgRef = useRef(null);
  const [viewBox, setViewBox] = useState({x:0,y:0,w:1200,h:700});

  useEffect(() => {
    fetch('/api/operators').then(r=>r.json()).then(setOperators).catch(()=>{});
    fetchGraphs();
  }, []);

  function fetchGraphs() {
    setGraphsLoading(true);
    fetch('/api/graphs').then(r=>r.json()).then(setGraphs).catch(()=>{}).finally(()=>setGraphsLoading(false));
  }

  function handleCanvasDrop(e) {
    e.preventDefault();
    const raw = e.dataTransfer.getData('application/json');
    if (!raw) return;
    const op = JSON.parse(raw);
    const rect = svgRef.current.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * viewBox.w + viewBox.x;
    const y = ((e.clientY - rect.top) / rect.height) * viewBox.h + viewBox.y;

    const existing = nodes.filter(n => n.operator?.class_name === op.class_name);
    const suffix = existing.length > 0 ? `_${existing.length + 1}` : '';

    const newNode = {
      id: _uid(),
      x: Math.round(x - 90),
      y: Math.round(y - 28),
      operator: op,
      config: {},
      varName: _toVarName(op.class_name) + suffix,
    };
    setNodes(prev => [...prev, newNode]);
  }

  function handleNodeMouseDown(e, nodeId) {
    if (connecting) return;
    const rect = svgRef.current.getBoundingClientRect();
    const node = nodes.find(n => n.id === nodeId);
    if (!node) return;
    const mx = ((e.clientX - rect.left) / rect.width) * viewBox.w + viewBox.x;
    const my = ((e.clientY - rect.top) / rect.height) * viewBox.h + viewBox.y;
    setDragNode(nodeId);
    setDragOffset({ x: mx - node.x, y: my - node.y });
    dragStartPos.current = { x: e.clientX, y: e.clientY };
    isDragging.current = false;
    nodeInteractionRef.current = true;
    setSelectedNodeId(nodeId);
    setSelectedEdgeId(null);
  }

  function handleCanvasMouseMove(e) {
    if (!dragNode) return;
    if (!isDragging.current && dragStartPos.current) {
      const dx = e.clientX - dragStartPos.current.x;
      const dy = e.clientY - dragStartPos.current.y;
      if (Math.abs(dx) < 4 && Math.abs(dy) < 4) return;
      isDragging.current = true;
    }
    const rect = svgRef.current.getBoundingClientRect();
    const mx = ((e.clientX - rect.left) / rect.width) * viewBox.w + viewBox.x;
    const my = ((e.clientY - rect.top) / rect.height) * viewBox.h + viewBox.y;
    setNodes(prev => prev.map(n => n.id === dragNode ? { ...n, x: mx - dragOffset.x, y: my - dragOffset.y } : n));
  }

  function handleCanvasMouseUp() {
    setDragNode(null);
    dragStartPos.current = null;
    isDragging.current = false;
  }

  function handlePortClick(nodeId, port) {
    if (!connecting) {
      if (port === 'out') setConnecting({ from: nodeId });
    } else {
      if (port === 'in' && connecting.from !== nodeId) {
        const exists = edges.some(e => e.from === connecting.from && e.to === nodeId);
        if (!exists) {
          setEdges(prev => [...prev, { id: _uid(), from: connecting.from, to: nodeId, config: {} }]);
        }
      }
      setConnecting(null);
    }
  }

  function deleteNode(nodeId) {
    setNodes(prev => prev.filter(n => n.id !== nodeId));
    setEdges(prev => prev.filter(e => e.from !== nodeId && e.to !== nodeId));
    if (selectedNodeId === nodeId) setSelectedNodeId(null);
  }

  function updateNode(nodeId, updates) {
    setNodes(prev => prev.map(n => n.id === nodeId ? { ...n, ...updates } : n));
  }

  function updateEdge(edgeId, updates) {
    setEdges(prev => prev.map(e => e.id === edgeId ? { ...e, ...updates } : e));
  }

  function deleteEdge(edgeId) {
    setEdges(prev => prev.filter(e => e.id !== edgeId));
    if (selectedEdgeId === edgeId) setSelectedEdgeId(null);
  }

  function clearCanvas() {
    if (nodes.length === 0 || confirm('Clear the canvas? All unsaved work will be lost.')) {
      setNodes([]); setEdges([]);
      setSelectedNodeId(null); setSelectedEdgeId(null);
      setActiveGraphId(null); setGraphName(''); setGraphDescription('');
    }
  }

  function loadGraph(graph) {
    try {
      const data = typeof graph.graph_json === 'string' ? JSON.parse(graph.graph_json) : graph.graph_json;
      setNodes(data.nodes || []);
      setEdges(data.edges || []);
      setActiveGraphId(graph.id);
      setGraphName(graph.name || '');
      setGraphDescription(graph.description || '');
      setSelectedNodeId(null);
      setSelectedEdgeId(null);
      setShowTable(false);
    } catch {
      console.error('Failed to parse graph JSON');
    }
  }

  async function handleSave() {
    setSaving(true);
    const code = _generateCode(nodes, edges);
    const payload = {
      name: graphName,
      description: graphDescription,
      graph_json: { nodes, edges },
      generated_code: code,
    };
    try {
      if (activeGraphId) {
        await fetch(`/api/graphs/${activeGraphId}`, { method:'PUT', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) });
      } else {
        const res = await fetch('/api/graphs', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) });
        const data = await res.json();
        setActiveGraphId(data.id);
      }
      fetchGraphs();
      setShowSaveModal(false);
    } catch (err) {
      console.error('Save failed:', err);
    } finally {
      setSaving(false);
    }
  }

  async function handleDeleteGraph(id) {
    if (!confirm('Delete this saved graph?')) return;
    try {
      await fetch(`/api/graphs/${id}`, { method:'DELETE' });
      if (activeGraphId === id) { setActiveGraphId(null); setGraphName(''); setGraphDescription(''); }
      fetchGraphs();
    } catch {}
  }

  const generatedCode = useMemo(() => _generateCode(nodes, edges), [nodes, edges]);
  const selectedNode = nodes.find(n => n.id === selectedNodeId);
  const selectedEdge = edges.find(e => e.id === selectedEdgeId);

  return (
    <div style={{display:'flex',flexDirection:'column',height:'100%',overflow:'hidden'}}>
      {/* Toolbar */}
      <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',padding:'10px 16px',borderBottom:'1px solid var(--nv-border)',background:'var(--nv-surface)',flexShrink:0}}>
        <div style={{display:'flex',alignItems:'center',gap:'8px'}}>
          <button className={`btn btn-sm ${showTable ? 'btn-primary' : 'btn-secondary'}`}
            onClick={() => setShowTable(!showTable)} style={{fontSize:'11px',padding:'4px 10px'}}>
            <IconDatabase /> Saved Graphs
          </button>
          {activeGraphId && (
            <span style={{fontSize:'12px',color:'var(--nv-green)',fontWeight:600}}>
              Editing: {graphName || `Graph #${activeGraphId}`}
            </span>
          )}
        </div>
        <div style={{display:'flex',alignItems:'center',gap:'8px'}}>
          {connecting && (
            <span style={{fontSize:'11px',color:'#64b4ff',fontWeight:600,padding:'3px 10px',background:'rgba(100,180,255,0.1)',borderRadius:'6px',border:'1px solid rgba(100,180,255,0.2)'}}>
              Click a target input port to connect ·
              <span style={{cursor:'pointer',marginLeft:'6px',color:'var(--nv-text-dim)'}} onClick={()=>setConnecting(null)}>Cancel</span>
            </span>
          )}
          <button className="btn btn-secondary btn-sm" onClick={clearCanvas} style={{fontSize:'11px',padding:'4px 10px'}}>
            <IconTrash /> Clear
          </button>
          <button className="btn btn-secondary btn-sm" onClick={() => setShowCode(true)} disabled={nodes.length===0}
            style={{fontSize:'11px',padding:'4px 10px'}}>
            <IconFileText /> View Code
          </button>
          <button className="btn btn-primary btn-sm" onClick={() => { setShowSaveModal(true); }} disabled={nodes.length===0}
            style={{fontSize:'11px',padding:'4px 10px'}}>
            <IconDownload /> Save Graph
          </button>
          <button className="btn btn-sm" onClick={() => setShowRunModal(true)} disabled={!activeGraphId || nodes.length===0}
            style={{fontSize:'11px',padding:'4px 10px',background:'rgba(118,185,0,0.15)',color:'var(--nv-green)',border:'1px solid rgba(118,185,0,0.3)'}}>
            <IconPlay /> Run Graph
          </button>
        </div>
      </div>

      {/* Saved Graphs Table (collapsible) */}
      {showTable && (
        <div style={{borderBottom:'1px solid var(--nv-border)',maxHeight:'220px',overflow:'auto',flexShrink:0}}>
          {graphsLoading ? (
            <div style={{padding:'20px',textAlign:'center',color:'var(--nv-text-dim)'}}><span className="spinner"></span> Loading…</div>
          ) : graphs.length === 0 ? (
            <div style={{padding:'20px',textAlign:'center',color:'var(--nv-text-dim)',fontSize:'13px'}}>No saved graphs yet. Design a pipeline and save it.</div>
          ) : (
            <table className="runs-table" style={{width:'100%',fontSize:'12px'}}>
              <thead><tr>
                <th style={{padding:'8px 12px'}}>Name</th>
                <th style={{padding:'8px 12px'}}>Description</th>
                <th style={{padding:'8px 12px'}}>Created</th>
                <th style={{padding:'8px 12px'}}>Updated</th>
                <th style={{padding:'8px 12px',width:'100px'}}>Actions</th>
              </tr></thead>
              <tbody>
                {graphs.map(g => (
                  <tr key={g.id} style={{cursor:'pointer',background: activeGraphId === g.id ? 'rgba(118,185,0,0.06)' : undefined}}
                    onClick={() => loadGraph(g)}>
                    <td style={{padding:'8px 12px',fontWeight:600,color:'#fff'}}>{g.name}</td>
                    <td style={{padding:'8px 12px',color:'var(--nv-text-muted)',maxWidth:'300px',overflow:'hidden',textOverflow:'ellipsis',whiteSpace:'nowrap'}}>{g.description || '—'}</td>
                    <td style={{padding:'8px 12px',color:'var(--nv-text-dim)'}}>{g.created_at?.substring(0,16).replace('T',' ')}</td>
                    <td style={{padding:'8px 12px',color:'var(--nv-text-dim)'}}>{g.updated_at?.substring(0,16).replace('T',' ')}</td>
                    <td style={{padding:'8px 12px'}} onClick={e=>e.stopPropagation()}>
                      <button className="btn btn-ghost btn-sm" onClick={() => handleDeleteGraph(g.id)}
                        style={{fontSize:'10px',color:'#ff5050',padding:'2px 6px'}}>
                        <IconTrash />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}

      {/* Main area: palette + canvas + config panel */}
      <div style={{display:'flex',flex:1,overflow:'hidden'}}>
        <OperatorPalette operators={operators} filter={paletteFilter} setFilter={setPaletteFilter} />

        {/* SVG Canvas */}
        <div style={{flex:1,position:'relative',overflow:'hidden',background:'var(--nv-bg)'}}
          onDragOver={e=>e.preventDefault()} onDrop={handleCanvasDrop}>
          <svg ref={svgRef} viewBox={`${viewBox.x} ${viewBox.y} ${viewBox.w} ${viewBox.h}`}
            style={{width:'100%',height:'100%',display:'block'}}
            onMouseMove={handleCanvasMouseMove} onMouseUp={handleCanvasMouseUp}
            onClick={e => {
              if (nodeInteractionRef.current) { nodeInteractionRef.current = false; return; }
              if (e.target === svgRef.current || e.target.getAttribute('fill') === 'url(#grid)') { setSelectedNodeId(null); setSelectedEdgeId(null); setConnecting(null); }
            }}>
            <defs>
              <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="var(--nv-text-dim)" />
              </marker>
              <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
                <path d="M 20 0 L 0 0 0 20" fill="none" stroke="rgba(255,255,255,0.03)" strokeWidth="0.5" />
              </pattern>
            </defs>
            <rect x={viewBox.x} y={viewBox.y} width={viewBox.w} height={viewBox.h} fill="url(#grid)" />

            {edges.map(e => (
              <EdgeArrow key={e.id} edge={e} nodes={nodes} selected={selectedEdgeId===e.id}
                onSelect={id => { setSelectedEdgeId(id); setSelectedNodeId(null); }} />
            ))}

            {nodes.map(n => {
              const isSource = n.operator?.type === 'ray_data_source';
              const nh = isSource ? 66 : 56;
              return (
                <g key={n.id}>
                  <CanvasNode node={n} selected={selectedNodeId===n.id}
                    onMouseDown={handleNodeMouseDown} onSelect={id => { nodeInteractionRef.current = true; setSelectedNodeId(id); setSelectedEdgeId(null); }}
                    onDelete={deleteNode} />
                  {!isSource && (
                    <circle cx={n.x} cy={n.y+nh/2} r="12"
                      fill="rgba(0,0,0,0.001)" pointerEvents="all" style={{cursor:'crosshair'}}
                      onMouseDown={e => e.stopPropagation()}
                      onClick={e => { e.stopPropagation(); handlePortClick(n.id, 'in'); }} />
                  )}
                  <circle cx={n.x+180} cy={n.y+nh/2} r="12"
                    fill="rgba(0,0,0,0.001)" pointerEvents="all" style={{cursor:'crosshair'}}
                    onMouseDown={e => e.stopPropagation()}
                    onClick={e => { e.stopPropagation(); handlePortClick(n.id, 'out'); }} />
                </g>
              );
            })}

            {nodes.length === 0 && (
              <text x={viewBox.x + viewBox.w/2} y={viewBox.y + viewBox.h/2} textAnchor="middle" fill="var(--nv-text-dim)" fontSize="14" fontFamily="inherit">
                Drag operators from the palette onto this canvas
              </text>
            )}
          </svg>

          {/* Connection instructions overlay */}
          {connecting && (
            <div style={{position:'absolute',bottom:'12px',left:'50%',transform:'translateX(-50%)',
              padding:'6px 14px',borderRadius:'8px',background:'rgba(100,180,255,0.15)',border:'1px solid rgba(100,180,255,0.3)',
              color:'#64b4ff',fontSize:'12px',fontWeight:500,pointerEvents:'none'}}>
              Click the input port (left circle) of another operator to create a connection
            </div>
          )}
        </div>

        {/* Right config panel */}
        {selectedNode && (
          <NodeConfigPanel node={selectedNode} onUpdate={updateNode}
            onClose={() => setSelectedNodeId(null)} />
        )}
        {selectedEdge && !selectedNode && (
          <EdgeConfigPanel edge={selectedEdge} nodes={nodes} onUpdate={updateEdge}
            onClose={() => setSelectedEdgeId(null)} />
        )}
        {selectedEdge && !selectedNode && (
          <div style={{position:'absolute',bottom:'12px',right:'300px'}}>
            <button className="btn btn-sm" style={{fontSize:'10px',padding:'3px 8px',background:'rgba(255,80,80,0.12)',color:'#ff5050',border:'1px solid rgba(255,80,80,0.2)'}}
              onClick={() => deleteEdge(selectedEdge.id)}>
              <IconTrash /> Delete Connection
            </button>
          </div>
        )}
      </div>

      {/* Modals */}
      {showCode && (
        <CodePreviewModal code={generatedCode} onClose={() => setShowCode(false)}
          onSave={() => { setShowCode(false); setShowSaveModal(true); }} />
      )}
      {showSaveModal && (
        <SaveGraphModal name={graphName} setName={setGraphName} description={graphDescription} setDescription={setGraphDescription}
          onSave={handleSave} onClose={() => setShowSaveModal(false)} saving={saving} isUpdate={!!activeGraphId} />
      )}
      {showRunModal && activeGraphId && (
        <RunGraphModal graphId={activeGraphId} graphName={graphName || `Graph #${activeGraphId}`}
          onClose={() => setShowRunModal(false)} />
      )}
    </div>
  );
}
