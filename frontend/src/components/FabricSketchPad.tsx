// src/components/FabricSketchPad.tsx
import { useEffect, useRef, useState, useCallback, forwardRef, useMemo } from "react"
// --- Fabric import made robust across ESM/CJS/Vite variants ---
import * as fabricNS from "fabric"
const fabric: typeof import("fabric").fabric = (fabricNS as any).fabric ?? (fabricNS as any)

import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Slider } from "@/components/ui/slider"
import { Separator } from "@/components/ui/separator"

// icons
import { Brush, Layers, Undo2, Eraser, Search, WandSparkles } from "lucide-react"

// grid for results (no Sheet)
import ResultsGrid from "./ResultsGrid"
import type { SearchResult } from "../types"

export type FabricSketchPadProps = {
    userId: string
    apiBase?: string
    width?: number
    height?: number
    initialBrushSize?: number
    onSearched?: (resp: any) => void
    /**
     * Optional: tap into the ready-to-send package (FormData + JSON) before we POST.
     */
    // @ts-ignore
    onEdited?: (pkg: {
        formData: FormData
        json: any
        meta: {
            refs: Array<any>
            mask_bbox?: { x: number; y: number; w: number; h: number }
            sketchBytes: number
            maskBytes: number
            imageMode: "rgb-on-white"
            maskMode: "white_edit_black_keep"
            model?: string
        }
    }) => void
    searchParams?: {
        queryText?: string
        wImg?: number
        wEdge?: number
        wTxt?: number
        tagFilters?: string
        personalize?: boolean
    }
}

const BG = "#11111b"
const ACCENT = "#f38ba8"

// Allowlisted style tags (max 2 at a time)
const STYLE_TAGS_ALLOWLIST = [
    "still_life","portrait","character_design","landscape","cityscape","pop_art","surrealism","impressionism",
    "minimalism","art_nouveau","art_deco","vaporwave","cyberpunk","manga_style","vector_art","pixel_art","line_art",
    "cel_shading","soft_shading","dutch_angle","negative_space","symmetry","close_up","isometric","monochrome",
    "neon_lights","pastel_palettes","pastel_colors","bokeh","vignetting","anatomy","abstract","abstract_background"
] as const

function dataURLToBlob(dataURL: string): Blob {
    const [meta, content] = dataURL.split(",")
    const isBase64 = /;base64$/.test(meta) || /;base64;/.test(meta)
    const byteString = isBase64 ? atob(content) : decodeURIComponent(content)
    const mimeMatch = meta.match(/data:([^;]+)/)
    const mime = mimeMatch ? mimeMatch[1] : "image/png"
    const ia = new Uint8Array(byteString.length)
    for (let i = 0; i < byteString.length; i++) ia[i] = byteString.charCodeAt(i)
    return new Blob([ia], { type: mime })
}

type Hit = {
    id: string | number
    score?: number
    image_url?: string
    preview_url?: string
    path?: string
    payload?: Record<string, any>
    [k: string]: any
}

const MAX_STYLE_TAGS = 2

const FabricSketchPadImpl = (
    {
        userId,
        apiBase = "/api",
        width = 900,
        height = 600,
        initialBrushSize = 4,
        onSearched,
        // @ts-ignore
        onEdited,
        searchParams,
    }: FabricSketchPadProps,
    _ref: any
) => {
    const lineRef = useRef<HTMLCanvasElement>(null)
    const maskRef = useRef<HTMLCanvasElement>(null)

    const [lineCanvas, setLineCanvas] = useState<fabric.Canvas | null>(null)
    const [maskCanvas, setMaskCanvas] = useState<fabric.Canvas | null>(null)
    const [activeLayer, setActiveLayer] = useState<"line" | "mask">("line")
    const [brushSize, setBrushSize] = useState<number>(initialBrushSize)
    const [lineHistory, setLineHistory] = useState<Array<fabric.Object>>([])
    const [maskHistory, setMaskHistory] = useState<Array<fabric.Object>>([])
    const [maskDrawCount, setMaskDrawCount] = useState(0)
    // @ts-ignore
    const maskHasPaint = maskDrawCount > 0

    // results state
    const [hits, setHits] = useState<Hit[]>([])
    const [selected, setSelected] = useState<string[]>([])
    const [lastLatencyMs, setLastLatencyMs] = useState<number | null>(null)

    // style tag selection for promptless customization
    const [styleTags, setStyleTags] = useState<string[]>([])

    // generating overlay state
    const [isGenerating, setIsGenerating] = useState(false)

    // keep wrapper refs so we can control stacking
    const lineWrapRef = useRef<HTMLElement | null>(null)
    const maskWrapRef = useRef<HTMLElement | null>(null)

    // ---- helpers
    function applyFabricSizing(c: fabric.Canvas, cssW: number, cssH: number, dpr: number) {
        c.setDimensions({ width: Math.floor(cssW * dpr), height: Math.floor(cssH * dpr) }, { backstoreOnly: true })
        c.setDimensions({ width: cssW, height: cssH }, { cssOnly: true })
    }

    function ensureBrush(c: fabric.Canvas, color: string, px: number) {
        // PencilBrush exists under fabric namespace in v5+
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        c.freeDrawingBrush = new (fabric as any).PencilBrush(c)
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        ;(c.freeDrawingBrush as any).color = color
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        ;(c.freeDrawingBrush as any).width = px
    }

    function getHitEl(c: fabric.Canvas): HTMLElement | null {
        const anyC = c as any
        if (typeof anyC.getSelectionElement === "function") return anyC.getSelectionElement()
        return (anyC.upperCanvasEl as HTMLElement) || (anyC.lowerCanvasEl as HTMLElement) || null
    }

    function captureWrapper(c: fabric.Canvas): HTMLElement | null {
        const anyC = c as any
        const upper = anyC.upperCanvasEl as HTMLElement | undefined
        const wrap = upper?.parentElement ?? (anyC.lowerCanvasEl as HTMLElement | undefined)?.parentElement ?? null
        return wrap ?? null
    }

    function styleWrapper(el: HTMLElement | null, cssW: number, cssH: number, z: number) {
        if (!el) return
        Object.assign(el.style, {
            position: "absolute",
            justifySelf: "center",
            inset: "0",
            width: `${cssW}px`,
            height: `${cssH}px`,
            zIndex: String(z),
            pointerEvents: "auto",
        } as Partial<CSSStyleDeclaration>)
    }

    // ---- URL & inlining helpers ----

    function normalizeApiBase(base?: string) {
        const b = (base || "").trim()
        if (!b) return ""
        const noLead = b.replace(/^\/+/, "")
        return "/" + noLead.replace(/\/+$/, "")
    }

    function resolveToApi(u: string | undefined, apiBase: string) {
        if (!u) return u
        if (/^(data:|https?:\/\/)/i.test(u)) return u
        const api = normalizeApiBase(apiBase)
        if (u.startsWith("/")) {
            return window.location.origin + (u.startsWith(api + "/") ? u : api + u)
        }
        return window.location.origin + api + "/" + u.replace(/^\/+/, "")
    }

    async function fetchImageAsDataURL(u: string) {
        const res = await fetch(u, {
            credentials: "same-origin",
            headers: { Accept: "image/*" },
        })
        const ct = (res.headers.get("content-type") || "").toLowerCase()
        if (!ct.startsWith("image/")) {
            throw new Error(`Non-image content-type: ${ct}`)
        }
        const blob = await res.blob()
        return await blobToDataURL(blob)
    }

    function blobToDataURL(b: Blob): Promise<string> {
        return new Promise((resolve) => {
            const fr = new FileReader()
            fr.onloadend = () => resolve(String(fr.result))
            fr.readAsDataURL(b)
        })
    }

    async function inlineRefs(refs: any[], apiBase: string): Promise<any[]> {
        const out: any[] = []
        for (const r of refs) {
            const current = r?.image_url as string | undefined
            if (!current) { out.push(r); continue }
            const resolved = resolveToApi(current, apiBase)
            if (!resolved) { out.push(r); continue }
            if (resolved.startsWith("data:")) {
                out.push({ ...r, image_url: resolved }); continue
            }
            try {
                const data = await fetchImageAsDataURL(resolved)
                out.push({ ...r, image_url: data })
            } catch (e) {
                console.warn("inlineRefs: failed to inline", resolved, e)
                out.push(r)
            }
        }
        return out
    }

    // -----------------------------

    useEffect(() => {
        if (!lineRef.current || !maskRef.current) return
        const dpr = window.devicePixelRatio || 1
        const cssW = width
        const cssH = height

        const line = new fabric.Canvas(lineRef.current, {
            isDrawingMode: true,
            selection: false,
            enableRetinaScaling: false,
            backgroundColor: "rgba(0,0,0,0)",
        })
        const mask = new fabric.Canvas(maskRef.current, {
            isDrawingMode: false,
            selection: false,
            backgroundColor: "rgba(0,0,0,0)",
            enableRetinaScaling: false,
        })

        applyFabricSizing(line, cssW, cssH, dpr)
        applyFabricSizing(mask, cssW, cssH, dpr)

        ensureBrush(line, ACCENT, initialBrushSize * dpr)
        ensureBrush(mask, "#ffffff", initialBrushSize * dpr)

        lineWrapRef.current = captureWrapper(line)
        maskWrapRef.current = captureWrapper(mask)
        styleWrapper(lineWrapRef.current, cssW, cssH, 2)
        styleWrapper(maskWrapRef.current, cssW, cssH, 3)

        const addHandler = (layer: "line" | "mask", obj: fabric.Object) => {
            if (layer === "line") setLineHistory((h) => [...h, obj])
            else setMaskHistory((h) => [...h, obj])
        }
        const onLinePath = (e: any) => { if (e?.path) addHandler("line", e.path) }
        const onMaskPath = (e: any) => { if (e?.path) { addHandler("mask", e.path); setMaskDrawCount((n) => n + 1) } }

        line.on("path:created", onLinePath)
        mask.on("path:created", onMaskPath)

        setLineCanvas(line)
        setMaskCanvas(mask)

        return () => {
            line.off("path:created", onLinePath)
            mask.off("path:created", onMaskPath)
            line.dispose()
            mask.dispose()
            setLineCanvas(null)
            setMaskCanvas(null)
            lineWrapRef.current = null
            maskWrapRef.current = null
        }
    }, [width, height, initialBrushSize])

    useEffect(() => {
        const dpr = window.devicePixelRatio || 1
        if (lineCanvas?.freeDrawingBrush) (lineCanvas.freeDrawingBrush as any).width = brushSize * dpr
        if (maskCanvas?.freeDrawingBrush) (maskCanvas.freeDrawingBrush as any).width = brushSize * dpr
    }, [brushSize, lineCanvas, maskCanvas])

    useEffect(() => {
        if (!lineCanvas || !maskCanvas) return
        const lineOn = activeLayer === "line"
        const maskOn = activeLayer === "mask"

        if (lineWrapRef.current && maskWrapRef.current) {
            if (lineOn) {
                lineWrapRef.current.style.zIndex = "3"
                maskWrapRef.current.style.zIndex = "2"
            } else {
                lineWrapRef.current.style.zIndex = "2"
                maskWrapRef.current.style.zIndex = "3"   // <- fixed typo
            }
        }

        lineCanvas.isDrawingMode = lineOn
        maskCanvas.isDrawingMode = maskOn

        const lineHit = getHitEl(lineCanvas)
        const maskHit = getHitEl(maskCanvas)
        if (lineHit) lineHit.style.pointerEvents = lineOn ? "auto" : "none"
        if (maskHit) maskHit.style.pointerEvents = maskOn ? "auto" : "none"

        lineCanvas.selection = lineOn
        maskCanvas.selection = maskOn
        lineCanvas.forEachObject((o) => (o.evented = lineOn))
        maskCanvas.forEachObject((o) => (o.evented = maskOn))
    }, [activeLayer, lineCanvas, maskCanvas])

    const undo = useCallback(() => {
        if (activeLayer === "line" && lineCanvas) {
            const obj = lineHistory[lineHistory.length - 1]
            if (obj) {
                lineCanvas.remove(obj)
                setLineHistory((h) => h.slice(0, -1))
                lineCanvas.requestRenderAll()
            }
        }
        if (activeLayer === "mask" && maskCanvas) {
            const obj = maskHistory[maskHistory.length - 1]
            if (obj) {
                maskCanvas.remove(obj)
                setMaskHistory((h) => h.slice(0, -1))
                maskCanvas.requestRenderAll()
            }
        }
    }, [activeLayer, lineCanvas, maskCanvas, lineHistory, maskHistory])

    const clearLayer = useCallback(() => {
        const dpr = window.devicePixelRatio || 1
        if (activeLayer === "line" && lineCanvas) {
            lineCanvas.clear()
            lineCanvas.setBackgroundColor("rgba(0,0,0,0)" as any, () => {})
            setLineHistory([])
            ensureBrush(lineCanvas, ACCENT, dpr * brushSize)
        }
        if (activeLayer === "mask" && maskCanvas) {
            maskCanvas.clear()
            maskCanvas.setBackgroundColor("rgba(0,0,0,0)" as any, () => {})
            setMaskHistory([])
            setMaskDrawCount(0)
            ensureBrush(maskCanvas, "#ffffff", dpr * brushSize)
        }
    }, [activeLayer, lineCanvas, maskCanvas, brushSize])

    const clearSketch = useCallback(() => {
        const dpr = window.devicePixelRatio || 1
        if (lineCanvas) {
            lineCanvas.clear()
            lineCanvas.setBackgroundColor("rgba(0,0,0,0)" as any, () => {})
            setLineHistory([])
            ensureBrush(lineCanvas, ACCENT, dpr * brushSize)
        }
    }, [lineCanvas, brushSize])

    const exportSketchRGB = useCallback((): Blob | null => {
        if (!lineCanvas) return null
        lineCanvas.discardActiveObject()
        lineCanvas.requestRenderAll()
        const src = lineCanvas.toCanvasElement(1)
        const w = src.width, h = src.height
        const out = document.createElement("canvas")
        out.width = w
        out.height = h
        const ctx = out.getContext("2d")
        if (!ctx) return null
        ctx.fillStyle = "#fff"
        ctx.fillRect(0, 0, w, h)
        ctx.drawImage(src, 0, 0)
        return dataURLToBlob(out.toDataURL("image/png"))
    }, [lineCanvas])

    // --- Results helpers
    function normalizeHits(json: any): Hit[] {
        const arr = json?.hits || json?.results || json || []
        return Array.isArray(arr) ? arr : []
    }
    function hitId(h: Hit): string { return String(h.id ?? (h.point?.id ?? h.payload?.id)) }
    function hitScore(h: Hit): number | undefined { return (h.score ?? h.payload?.score) as number | undefined }
    function hitImageUrl(h: Hit): string | undefined {
        const fromPayload = h.payload?.image_url || h.payload?.preview_url
        const direct = h.image_url || h.preview_url || fromPayload
        if (direct) return resolveToApi(direct as string, apiBase)
        const id = hitId(h)
        return `${normalizeApiBase(apiBase)}/image/${encodeURIComponent(id)}`
    }

    const postSearch = useCallback(async () => {
        if (!lineCanvas || lineCanvas.getObjects().length === 0) return
        const sketchRGB = exportSketchRGB()
        if (!sketchRGB) return

        const fd = new FormData()
        fd.append("user_id", userId)
        fd.append("sketch", sketchRGB, "sketch.png")
        fd.append("image", sketchRGB, "image.png")

        if (searchParams?.queryText) fd.append("query_text", searchParams.queryText)
        if (typeof searchParams?.wImg === "number") { fd.append("w_img", String(searchParams.wImg)); fd.append("w_image", String(searchParams.wImg)) }
        if (typeof searchParams?.wEdge === "number") { fd.append("w_edge", String(searchParams.wEdge)); fd.append("w_edges", String(searchParams.wEdge)) }
        if (typeof searchParams?.wTxt === "number")  { fd.append("w_txt", String(searchParams.wTxt)); fd.append("w_text", String(searchParams.wTxt)) }
        if (searchParams?.tagFilters) fd.append("tag_filters", searchParams.tagFilters)
        if (typeof searchParams?.personalize === "boolean") fd.append("personalize", String(searchParams.personalize))

        fd.append("meta_sketch_bytes", String(sketchRGB.size))

        const t0 = performance.now()
        const resp = await fetch(`${apiBase}/search/hybrid`, { method: "POST", body: fd })
        const json = await resp.json()
        const latency = Math.round(performance.now() - t0)
        ;(json as any)._latency_ms = latency

        onSearched?.(json)
        const h = normalizeHits(json)
        setHits(h)
        setSelected([])
        setStyleTags([]) // reset style tags on new search
        setLastLatencyMs(latency)
    }, [apiBase, userId, exportSketchRGB, onSearched, searchParams, lineCanvas])

    // ---------- helpers for applying result ----------
    function setLineCanvasBackgroundFromDataURL(url: string) {
        if (!lineCanvas) return
            ;(fabric.Image as any).fromURL(url, (img: fabric.Image) => {
            img.set({ left: 0, top: 0, originX: "left", originY: "top" })
            img.scaleToWidth(lineCanvas.getWidth())
            img.scaleToHeight(lineCanvas.getHeight())
            lineCanvas.setBackgroundImage(img, lineCanvas.renderAll.bind(lineCanvas))
        }, { crossOrigin: "anonymous" })
    }

    // ---- Style tags UI — strict allowlist + nicer active state
    const availableStyleTags: string[] = useMemo(() => {
        return STYLE_TAGS_ALLOWLIST as unknown as string[]
    }, [])

    const toggleStyleTag = (t: string) => {
        setStyleTags((prev) => {
            const allowed = (STYLE_TAGS_ALLOWLIST as readonly string[]).includes(t)
            if (!allowed) return prev
            if (prev.includes(t)) return prev.filter((x) => x !== t)
            const next = [...prev, t]
            if (next.length > MAX_STYLE_TAGS) next.splice(0, next.length - MAX_STYLE_TAGS)
            return next
        })
    }

    // ---- GENERATE: Style + Sketch (STYLE ref + CONTROL/SCRIBBLE + sketch analysis prompt)
    const emitCustomizeFromSketch = useCallback(async (refIds: string[] = [], n = 4) => {
        const sketchRGB = exportSketchRGB()
        if (!sketchRGB) return

        const chosen = refIds.length ? [refIds[0]] : []
        const refsRaw = buildRefsFromSelection(hits, chosen)
        const refs = await inlineRefs(refsRaw.slice(0, 1), apiBase)

        const image_url = await blobToDataURL(sketchRGB)
        clearSketch() // clear immediately after capture
        const filteredTags = styleTags.filter(t => (STYLE_TAGS_ALLOWLIST as readonly string[]).includes(t)).slice(0, 2)

        const json = {
            refs,                // first ref = STYLE
            image_url,           // sketch (scribble control will be created server-side)
            style_tags: filteredTags,
            n,
            model: "auto"
        }

        setIsGenerating(true)
        try {
            const resp = await fetch(`${apiBase}/generate_image`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(json),
            })
            const out = await resp.json()
            const b64 = out?.data?.[0]?.b64_json
            if (!b64) return
            const url = `data:image/png;base64,${b64}`
            setLineCanvasBackgroundFromDataURL(url)
        } finally {
            setIsGenerating(false)
        }
    }, [apiBase, exportSketchRGB, hits, styleTags, clearSketch])

    // ---- GENERATE: Subject + Sketch (SUBJECT ref + CONTROL/SCRIBBLE + sketch analysis prompt)
    const emitSubjectFromSketch = useCallback(async (refIds: string[] = [], n = 1) => {
        const sketchRGB = exportSketchRGB()
        if (!sketchRGB) return

        const chosen = refIds.length ? [refIds[0]] : []
        const [subjectRef] = await inlineRefs(buildRefsFromSelection(hits, chosen).slice(0,1), apiBase)
        if (!subjectRef) return

        const filteredTags = styleTags.filter(t => (STYLE_TAGS_ALLOWLIST as readonly string[]).includes(t)).slice(0, 2)
        const image_url = await blobToDataURL(sketchRGB)
        clearSketch() // clear immediately after capture
        const json = {
            subject_ref: subjectRef,
            image_url,
            style_tags: filteredTags,
            subject_type_default: "SUBJECT_TYPE_PERSON",
            n,
            model: "auto"
        }

        setIsGenerating(true)
        try {
            const resp = await fetch(`${apiBase}/generate_image`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(json),
            })
            const out = await resp.json()
            const b64 = out?.data?.[0]?.b64_json
            if (!b64) return
            const url = `data:image/png;base64,${b64}`
            setLineCanvasBackgroundFromDataURL(url)
        } finally {
            setIsGenerating(false)
        }
    }, [apiBase, exportSketchRGB, hits, styleTags, clearSketch])

    // --- selection helpers
    // @ts-ignore
    const toggleSelect = (id: string) =>
        setSelected((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]))

    function buildRefsFromSelection(hs: Hit[], ids: string[]) {
        const selectedHits = hs.filter((h) => ids.includes(String(h.id ?? h.payload?.id)))
        return selectedHits.slice(0, 6).map((h) => {
            const id = String(h.id ?? h.payload?.id)
            const img = hitImageUrl(h)
            const tagsAll = (h.payload?.tags_all as string[] | undefined) || (h.payload?.tags as string[] | undefined) || []
            return {
                id,
                tags: tagsAll,
                palette: (h.payload?.palette as string[] | undefined) || (h.payload?.palette_hex as string[] | undefined) || [],
                caption: (h.payload?.caption as string | undefined) || (h.payload?.title as string | undefined) || "",
                image_url: img,
                score: hitScore(h),
            }
        })
    }

    // Map Hits -> SearchResult for the grid
    const gridResults: SearchResult[] = hits.map((h) => ({
        id: String(h.id ?? h.payload?.id),
        score: hitScore(h) ?? 0,
        payload: {
            ...h.payload,
            style_cluster: h.payload?.style_cluster,
            tags_all: h.payload?.tags_all || h.payload?.tags || [],
            source_post_url: h.payload?.source_post_url,
        },
    }))

    const styleButtonsEnabled = selected.length > 0

    return (
        <div className="w-full">
            {/* Toolbar */}
            <div
                className="mb-2 flex flex-wrap items-center gap-2 rounded-lg border p-2"
                style={{ backgroundColor: BG, borderColor: ACCENT, color: "#eee" }}
            >
                <div className="flex items-center gap-2">
                    <Badge
                        variant="secondary"
                        className="shrink-0 border"
                        style={{ backgroundColor: "transparent", borderColor: ACCENT, color: ACCENT }}
                    >
                        <Layers className="mr-1 h-3 w-3" style={{ color: ACCENT }} />
                        Layer
                    </Badge>
                    <Button
                        size="sm"
                        variant={activeLayer === "line" ? "default" : "outline"}
                        className={activeLayer === "line" ? "text-[#11111b]" : ""}
                        style={activeLayer === "line"
                            ? { backgroundColor: ACCENT }
                            : { borderColor: ACCENT, color: ACCENT, backgroundColor: "transparent" }}
                        onClick={() => setActiveLayer("line")}
                        disabled={isGenerating}
                    >
                        Line
                    </Button>
                    <Button
                        size="sm"
                        variant={activeLayer === "mask" ? "default" : "outline"}
                        className={activeLayer === "mask" ? "text-[#11111b]" : ""}
                        style={activeLayer === "mask"
                            ? { backgroundColor: ACCENT }
                            : { borderColor: ACCENT, color: ACCENT, backgroundColor: "transparent" }}
                        onClick={() => setActiveLayer("mask")}
                        disabled={isGenerating}
                    >
                        Mask
                    </Button>
                </div>

                <Separator orientation="vertical" className="mx-1 hidden h-6 md:block" style={{ backgroundColor: ACCENT }} />

                <div className="flex items-center gap-3">
          <span className="inline-flex items-center text-sm" style={{ color: "#ddd" }}>
            <Brush className="mr-1 h-4 w-4" style={{ color: ACCENT }} />
            Brush
          </span>
                    <div className="flex items-center gap-3">
                        <Slider
                            className="w-40"
                            min={1}
                            max={24}
                            step={1}
                            value={[brushSize]}
                            onValueChange={(v) => setBrushSize(v?.[0] ?? 1)}
                            disabled={isGenerating}
                        />
                        <span className="text-xs tabular-nums" style={{ color: "#ccc" }}>
              {brushSize}px
            </span>
                    </div>
                </div>

                <div className="ml-auto flex items-center gap-2">
                    <Button
                        size="sm"
                        variant="outline"
                        style={{ borderColor: ACCENT, color: ACCENT, backgroundColor: "transparent" }}
                        onClick={undo}
                        disabled={isGenerating}
                    >
                        <Undo2 className="mr-1 h-4 w-4" style={{ color: ACCENT }} />
                        Undo
                    </Button>
                    <Button
                        size="sm"
                        variant="outline"
                        style={{ borderColor: ACCENT, color: ACCENT, backgroundColor: "transparent" }}
                        onClick={clearLayer}
                        disabled={isGenerating}
                    >
                        <Eraser className="mr-1 h-4 w-4" style={{ color: ACCENT }} />
                        Clear Layer
                    </Button>
                    <Button size="sm" style={{ backgroundColor: ACCENT, color: BG }} onClick={postSearch} disabled={isGenerating}>
                        <Search className="mr-1 h-4 w-4" />
                        Use Sketch → Search
                    </Button>

                    {/* Promptless customization: no mask required */}
                    <Button
                        size="sm"
                        variant="secondary"
                        style={{ backgroundColor: "transparent", borderColor: ACCENT, color: ACCENT }}
                        onClick={() => emitCustomizeFromSketch(selected, 4)}
                        disabled={!styleButtonsEnabled || isGenerating}
                        title={styleButtonsEnabled ? "Generate with Style + Sketch (no mask)" : "Select a style reference first"}
                    >
                        <WandSparkles className="mr-1 h-4 w-4" />
                        Generate (Style+Sketch)
                    </Button>

                    {/* Subject + Sketch */}
                    <Button
                        size="sm"
                        variant="secondary"
                        style={{ backgroundColor: "transparent", borderColor: ACCENT, color: ACCENT }}
                        onClick={() => emitSubjectFromSketch(selected, 1)}
                        disabled={!selected.length || isGenerating}
                        title={selected.length ? "Generate with Subject + Sketch" : "Select a subject reference first"}
                    >
                        <WandSparkles className="mr-1 h-4 w-4" />
                        Generate (Subject+Sketch)
                    </Button>
                </div>
            </div>

            {/* Style tag picker (up to 2) */}
            <div className="mb-2 rounded-lg border p-2" style={{ borderColor: ACCENT, background: '#0b0b10', color:'#ddd' }}>
                <div className="flex items-center gap-2 flex-wrap">
                    <strong>Style tags:</strong>
                    <span className="text-xs opacity-70">choose up to {MAX_STYLE_TAGS}</span>
                    <div className="flex flex-wrap gap-2 ml-2">
                        {availableStyleTags.map((t) => {
                            const sel = styleTags.includes(t)
                            return (
                                <button
                                    key={t}
                                    onClick={() => !isGenerating && toggleStyleTag(t)}
                                    aria-pressed={sel}
                                    disabled={isGenerating}
                                    title={t.replace(/_/g, " ")}
                                    className={[
                                        "px-2 py-1 rounded-full border text-xs",
                                        "transition-all duration-200 ease-out",
                                        isGenerating ? "opacity-60 cursor-not-allowed" : "hover:opacity-90 active:scale-95",
                                    ].join(" ")}
                                    style={{
                                        borderColor: ACCENT,
                                        backgroundColor: sel ? ACCENT : "transparent",
                                        color: sel ? "#11111b" : ACCENT,
                                        boxShadow: sel
                                            ? "0 0 0 1px rgba(243,139,168,0.25), 0 6px 12px rgba(243,139,168,0.15)"
                                            : "none",
                                        transition: "background-color 180ms ease, color 180ms ease, box-shadow 180ms ease, transform 100ms ease",
                                    }}
                                >
                                    {t.replace(/_/g," ")}
                                </button>
                            )
                        })}
                    </div>
                    <span className="ml-auto text-xs opacity-70">{styleTags.length}/{MAX_STYLE_TAGS} selected</span>
                </div>
            </div>

            {/* Canvas stage */}
            <div
                className="relative overflow-hidden rounded-lg border"
                style={{ width, height, backgroundColor: BG, borderColor: ACCENT }}
            >
                <canvas ref={maskRef} className="absolute inset-0" />
                <canvas ref={lineRef} className="absolute inset-0" />

                {/* Low-opacity #f38ba8 skeleton overlay while generating */}
                {isGenerating && (
                    <div
                        className="absolute inset-0 pointer-events-none flex items-center justify-center animate-pulse"
                        style={{ backgroundColor: ACCENT, opacity: 0.18, zIndex: 5 }}
                    >
                        <div
                            className="rounded-md px-3 py-1 text-xs font-medium"
                            style={{ backgroundColor: "#ffffff", color: BG, opacity: 0.85 }}
                        >
                            Generating…
                        </div>
                    </div>
                )}
            </div>

            {/* Results + selection actions (grid) */}
            <div className="mt-4 rounded-lg border p-3" style={{ borderColor: ACCENT, background: '#0b0b10' }}>
                <div className="mb-3 flex flex-wrap items-center gap-2" style={{ color:'#ddd' }}>
                    <strong>Results:</strong> {hits.length}
                    {typeof lastLatencyMs === "number" ? <span>• {lastLatencyMs} ms</span> : null}
                    <div className="ml-auto flex gap-2">
                        <Button
                            size="sm"
                            variant="outline"
                            style={{ borderColor: ACCENT, color: ACCENT, backgroundColor: "transparent" }}
                            onClick={() => setSelected(hits.map((h) => String(h.id ?? h.payload?.id)))}
                            disabled={!hits.length || isGenerating}
                        >
                            Select all
                        </Button>
                        <Button
                            size="sm"
                            variant="outline"
                            style={{ borderColor: ACCENT, color: ACCENT, backgroundColor: "transparent" }}
                            onClick={() => setSelected([])}
                            disabled={!selected.length || isGenerating}
                        >
                            Clear selection
                        </Button>

                        {/* Convenience duplicates */}
                        <Button
                            size="sm"
                            variant="outline"
                            style={{ borderColor: ACCENT, color: ACCENT, backgroundColor: "transparent" }}
                            onClick={() => emitCustomizeFromSketch(selected, 4)}
                            disabled={!styleButtonsEnabled || isGenerating}
                        >
                            <WandSparkles className="mr-1 h-4 w-4" />
                            Generate (Style+Sketch)
                        </Button>

                        <Button
                            size="sm"
                            variant="outline"
                            style={{ borderColor: ACCENT, color: ACCENT, backgroundColor: "transparent" }}
                            onClick={() => emitSubjectFromSketch(selected, 1)}
                            disabled={!selected.length || isGenerating}
                        >
                            <WandSparkles className="mr-1 h-4 w-4" />
                            Generate (Subject+Sketch)
                        </Button>
                    </div>
                </div>

                <ResultsGrid
                    apiBase={apiBase}
                    userId={userId}
                    results={gridResults}
                    selected={selected}
                    onToggleSelect={(id) => !isGenerating && setSelected((prev) => prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id])}
                />
            </div>

            <p className="mt-2 text-xs" style={{ color: "#bbb" }}>
                Tip: Draw on <span style={{ color: ACCENT, fontWeight: 600 }}>Line</span>. Paint regions on{" "}
                <span style={{ color: ACCENT, fontWeight: 600 }}>Mask</span> for edits. For promptless generation, select a result, choose up to{" "}
                {MAX_STYLE_TAGS} style tags, and click{" "}
                <span style={{ color: ACCENT, fontWeight: 600 }}>Generate</span>.
            </p>
        </div>
    )
}

const FabricSketchPad = forwardRef(FabricSketchPadImpl)
export default FabricSketchPad
