
# How It Works

This document provides a detailed explanation of the inner workings of the Clippy application, from its core search functionality to image generation and personalization.

## Hybrid Search

The application's search functionality is powered by a hybrid system that combines multiple data modalities to deliver highly relevant results. This system leverages text, sketches, and edge detection to create a comprehensive search query that understands user intent from multiple angles.

### Vector Embeddings and Qdrant

At the heart of the search system are vector embeddings. Every image and text query is converted into a high-dimensional vector representation using a sophisticated AI model (OpenCLIP). These vectors capture the semantic meaning of the content, allowing for nuanced and context-aware searches.

These vectors are stored and indexed in a **Qdrant** vector database, which is optimized for high-speed similarity searches. When a user performs a search, the application queries the Qdrant database to find the vectors that are most similar to the user's query vector.

### Multi-Modal Search

The search is "hybrid" because it combines three different types of vectors:

*   **Image Vectors**: Generated from the user's sketch, capturing the overall composition and content.
*   **Edge Vectors**: Also generated from the sketch, but focusing on the shapes and outlines.
*   **Text Vectors**: Generated from the user's text query, capturing the semantic meaning of the words.

These three vectors are combined, with user-defined weights, to create a single, powerful query vector. This allows users to fine-tune their searches by emphasizing the importance of the sketch, the text, or the outlines.

## Image Generation

The application integrates with Google's **Imagen 3 (capability-001)** model through Vertex AI. This model enables advanced generative workflows, with strong support for style-transfer, subject conditioning, and mask-based editing. Compared to prompt-only systems, this approach provides predictable, high-quality outputs aligned with user intent.

### Key Capabilities of Imagen 3 (capability-001)

- **Subject + Style Customization**: Users can pass reference images for both the subject (e.g., a character, product, or object) and artistic style. The model adapts to these references, maintaining identity consistency while shifting the look and feel.
- **Sketch + Subject or Style**: Sketches can be used in combination with subject/style references. The sketch drives layout and composition, while the references provide grounding and aesthetics.
- **Mask-Based Editing**: Regions of an image can be selected (masked) for inpainting or outpainting. This enables targeted edits, object insertion/removal, or extending an artwork beyond its original boundaries.
- **Negative Prompting**: Prompts can explicitly state what *not* to include, reducing artifacts and guiding outputs away from undesired elements.
- **Supported Resolutions**: Multiple aspect ratios are supported, including 1024×1024, 1280×896, 896×1280, 1408×768, and 768×1408.

### Integration in Clippy

Clippy’s hybrid retrieval (sketch + edge + text) provides structured input to Imagen 3:

1. **Prompt Construction**: User’s text query, tags, and knowledge-graph preferences are combined into a descriptive prompt.
2. **Reference Conditioning**: Retrieved images (subject or style) can be passed as conditioning examples for Imagen 3.
3. **Sketch as Guidance**: Sketches are processed into edge maps or masks that guide composition during generation.
4. **Editing Workflows**: For edits, masks are generated either by the user or automatically from the sketch/selection, then applied in the Imagen request.
5. **Feedback Loop**: Generated images are re-embedded, tagged, and fed back into Clippy’s knowledge graph so that user interactions refine future outputs.

By combining hybrid search with Imagen 3’s generation and editing capabilities, Clippy provides artists with a system that delivers reference-aware, style-consistent, and controllable outputs.