export type SearchResult = {
    id: string
    score: number
    payload: {
        path?: string
        tags_all?: string[]
        style_cluster?: string
        source_post_url?: string
    }
}