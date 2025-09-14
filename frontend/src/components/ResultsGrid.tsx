import type { SearchResult } from '../types'

type Props = {
    apiBase?: string
    userId: string
    results: SearchResult[]
    selected: string[]
    onToggleSelect: (id: string) => void
}

export default function ResultsGrid({ apiBase = '/api', userId, results, selected, onToggleSelect }: Props) {
    const sendFeedback = async (image_id: string, action: 'click' | 'save', style?: string) => {
        try {
            await fetch(`${apiBase}/feedback`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ user_id: userId, image_id, action, style }),
            })
        } catch {}
    }

    return (
        <div className="grid gap-3" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))' }}>
            {results.map((r) => {
                const isSel = selected.includes(r.id)
                const style = r.payload?.style_cluster
                const tags = r.payload?.tags_all?.slice(0, 5) || []
                return (
                    <div key={r.id} className={`border rounded overflow-hidden ${isSel ? 'outline outline-2 outline-blue-500' : ''}`} style={{ background: '#0b0b10', borderColor: '#2a2a31' }}>
                        <div className="relative">
                            <img
                                src={`${apiBase}/image/${r.id}?w=512`}
                                alt={r.id}
                                loading="lazy"
                                style={{ width: '100%', display: 'block', aspectRatio: '1/1', objectFit: 'cover' }}
                                onClick={() => { onToggleSelect(r.id); sendFeedback(r.id, 'click', style) }}
                            />
                            <button
                                className={`btn ${isSel ? 'primary' : 'ghost'}`}
                                style={{ position:'absolute', top:8, right:8, fontSize:12, background: isSel ? '#f38ba8' : 'transparent', color: isSel ? '#11111b' : '#f38ba8', border: `1px solid #f38ba8`, padding: '4px 8px', borderRadius: 6 }}
                                onClick={(e) => { e.stopPropagation(); onToggleSelect(r.id) }}
                            >{isSel ? 'Selected' : 'Select'}</button>
                        </div>
                        <div style={{ padding: 8, fontSize: 12, color: '#ddd' }}>
                            <div className="badge" title="score" style={{ display:'inline-block', border: '1px solid #444', borderRadius: 6, padding: '2px 6px' }}>
                                {(r.score || 0).toFixed(3)}
                            </div>
                            {style && <div className="badge" style={{ display:'inline-block', marginLeft: 6, border: '1px solid #444', borderRadius: 6, padding: '2px 6px' }}>{style}</div>}
                            <div style={{ display:'flex', flexWrap:'wrap', gap:4, marginTop:6 }}>
                                {tags.map(t => <span key={t} className="badge" style={{ border:'1px solid #333', borderRadius: 6, padding:'2px 6px' }}>{t}</span>)}
                            </div>
                            <div style={{ display:'flex', gap:8, marginTop:8 }}>
                                <button className="btn" onClick={() => sendFeedback(r.id, 'save', style)} style={{ border:'1px solid #f38ba8', color:'#f38ba8', borderRadius:6, padding:'6px 10px', background:'transparent' }}>
                                    Save
                                </button>
                                {r.payload?.source_post_url && (
                                    <a className="btn" href={r.payload.source_post_url} target="_blank" rel="noreferrer" style={{ border:'1px solid #444', color:'#ddd', borderRadius:6, padding:'6px 10px', background:'transparent' }}>
                                        Source
                                    </a>
                                )}
                            </div>
                        </div>
                    </div>
                )
            })}
        </div>
    )
}
