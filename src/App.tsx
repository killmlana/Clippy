import React from "react"
import FabricSketchPad from "./components/FabricSketchPad"
import ResultsGrid from "./components/ResultsGrid"
import SearchControls from "./components/SearchControl"
import type { SearchResult } from "./types"

// shadcn button for consistent theming with the pad
import { Button } from "@/components/ui/button"

const BG = "#11111b"
const ACCENT = "#f38ba8"

export default function App() {
    const apiBase = "/api"

    const [results, setResults] = React.useState<SearchResult[]>([])
    const [selected, setSelected] = React.useState<string[]>([])
    const [lastLatency, setLastLatency] = React.useState<number | null>(null)
    const [params, setParams] = React.useState({
        queryText: "",
        wImg: 0.7,
        wEdge: 0.3,
        wTxt: 0.2,
        tagFilters: "",
        personalize: true,
    })

    const padRef = React.useRef<any>(null)

    const onSearched = (r: any) => {
        // FabricSketchPad sets _latency_ms on the response
        setResults(r.results || r.hits || [])
        setLastLatency(r._latency_ms ?? null)
        setSelected([])
    }

    const onSearchClick = () => padRef.current?.searchNow?.()
    const onEditWithSelected = () => padRef.current?.editWithRefs?.(selected)
    const toggleSelect = (id: string) =>
        setSelected((s) => (s.includes(id) ? s.filter((x) => x !== id) : [...s, id]))

    return (
        <div
            className="min-h-screen w-full"
            style={{ backgroundColor: BG, color: "#e5e7eb" }}
        >
            <div className="mx-auto max-w-6xl px-4 py-6">
                <header className="mb-4 flex items-end justify-between gap-3">
                    <h1
                        className="text-2xl font-semibold tracking-tight"
                        style={{ color: ACCENT, margin: 0 }}
                    >
                        Clippy → Style-Aware Reference and Generation
                    </h1>

                    {/* Optional external search trigger to complement the pad’s own button */}
                    <Button
                        onClick={onSearchClick}
                        style={{ backgroundColor: ACCENT, color: BG }}
                    >
                        Run Search
                    </Button>
                </header>

                <section className="mb-4 rounded-lg border p-3" style={{ borderColor: ACCENT }}>
                    <SearchControls params={params} onChange={setParams} onSearch={onSearchClick} />
                </section>

                <section className="mb-4">
                    <FabricSketchPad
                        ref={padRef}
                        userId="demo-user"
                        apiBase={apiBase}
                        searchParams={params}
                        onSearched={onSearched}
                    />
                </section>

                <section className="mb-2 flex items-center gap-2">
                    <Button
                        onClick={onEditWithSelected}
                        disabled={!selected.length}
                        style={{
                            backgroundColor: selected.length ? ACCENT : "#2a2a31",
                            color: selected.length ? BG : "#9aa0a6",
                        }}
                    >
                        Use Selected References → AI Edit
                    </Button>
                    <span style={{ fontSize: 12, color: "#9aa0a6" }}>
            Selected: {selected.length}
                        {lastLatency ? ` · Last query: ${lastLatency} ms` : ""}
          </span>
                </section>

                <section className="rounded-lg border" style={{ borderColor: ACCENT }}>
                    <ResultsGrid
                        apiBase={apiBase}
                        userId="demo-user"
                        results={results}
                        selected={selected}
                        onToggleSelect={toggleSelect}
                    />
                </section>
            </div>
        </div>
    )
}
