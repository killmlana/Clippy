type SearchParams = {
    queryText: string
    wImg: number
    wEdge: number
    wTxt: number
    tagFilters: string
    personalize: boolean
}


type Props = {
    params: SearchParams
    onChange: (p: SearchParams) => void
    onSearch: () => void
}


export default function SearchControls({ params, onChange, onSearch }: Props) {
    const { queryText, wImg, wEdge, wTxt, tagFilters, personalize } = params
    return (
        <div className="toolbar border rounded" style={{ padding: 8 }}>
            <div style={{ flex: 1, minWidth: 240 }}>
                <label className="block text-xs">Text / Tags (optional)</label>
                <input
                    className="border rounded"
                    style={{ width: '100%', padding: '6px 8px' }}
                    value={queryText}
                    onChange={(e) => onChange({ ...params, queryText: e.target.value })}
                    placeholder="e.g. apple, still life, pastel"
                    onKeyDown={(e)=>{ if(e.key==='Enter') onSearch(); }}
                />
            </div>
            <div>
                <label className="block text-xs">w_img: {wImg.toFixed(2)}</label>
                <input type="range" min={0} max={1} step={0.05} value={wImg} style={{ accentColor: '#f38ba8' }}
                       onChange={(e)=> onChange({ ...params, wImg: parseFloat(e.target.value) })} />
            </div>
            <div>
                <label className="block text-xs">w_edge: {wEdge.toFixed(2)}</label>
                <input type="range" min={0} max={1} step={0.05} value={wEdge} style={{ accentColor: '#f38ba8' }}
                       onChange={(e)=> onChange({ ...params, wEdge: parseFloat(e.target.value) })} />
            </div>
            <div>
                <label className="block text-xs">w_txt: {wTxt.toFixed(2)}</label>
                <input type="range" min={0} max={1} step={0.05} value={wTxt} style={{ accentColor: '#f38ba8' }}
                       onChange={(e)=> onChange({ ...params, wTxt: parseFloat(e.target.value) })} />
            </div>
            <div style={{ minWidth: 220 }}>
                <label className="block text-xs">Tag filters (CSV)</label>
                <input
                    className="border rounded"
                    style={{ width: '100%', padding: '6px 8px' }}
                    value={tagFilters}
                    onChange={(e) => onChange({ ...params, tagFilters: e.target.value })}
                    placeholder="e.g. apple, fruit, still_life"
                />
            </div>
            <label className="flex items-center gap-2 text-xs">
                <input type="checkbox" checked={personalize}
                       onChange={(e)=> onChange({ ...params, personalize: e.target.checked })} />
                Personalize
            </label>
            <button className="btn primary" onClick={onSearch}>Search</button>
        </div>
    )
}