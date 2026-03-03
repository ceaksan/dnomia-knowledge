# dnomia-knowledge: Unified Knowledge Management System

## Product Requirements Document

### Problem Statement

Claude Code oturumlarinda bilgi erisimi fragmentli:

- `mcp-code-search`: Sadece kod chunks, LanceDB, all-MiniLM-L6-v2 (384d), AST-aware
- `ceaksan-knowledge`: Sadece ceaksan icerigi, LanceDB, multilingual-e5-base (768d), GraphRAG
- `web-scraper`: Sadece scraped URL'ler, LanceDB
- Her biri ayri embedding modeli, ayri storage, ayri MCP server

Projeler arasi cross-reference yok. Bir projede ogrenilen pattern diger projelere tasinmiyor. Arama kalitesi hakkinda feedback loop yok.

### Solution

Tek bir MCP server, tek bir SQLite veritabani, tek bir embedding modeli. Her proje kendi `.knowledge.toml` config'iyle kayitli. Icerik ve kod ayni DB'de ayri domain'lerle. Graph edge'leri her zaman schema'da mevcut, populate edilmesi opsiyonel. Cross-project arama explicit link'ler uzerinden.

### Success Criteria

- Tek `search` tool'u ile hem ceaksan blog icerigi hem dnomia kodu aranabilir
- Cross-project arama: ceaksan'dan dnomia'ya link varsa, ceaksan'da arama yaparken dnomia sonuclari da gelir
- Incremental reindex: sadece degisen dosyalar yeniden indekslenir (MD5 content hash)
- Arama kalitesi zamanla iyilesir (interaction-based relevance boost)
- Mevcut `mcp-code-search` ve `ceaksan-knowledge` retire edilebilir

### Non-Goals

- Context guard (buyuk tool ciktilarini intercept/ozetleme) — ayri proje
- Real-time file watching (fsevents/inotify) — CLI/hook ile tetikleme yeterli
- Multi-user/multi-machine sync — tek makine, tek kullanici
- Web scraping — mevcut `web-scraper` MCP server devam eder

### Target User

Solo entrepreneur, 3-10 proje yonetiyor. Her projede farkli profil: icerik agirlikli (blog/course), SaaS (kod agirlikli), static site. Tek makinede calisiyor.

---

## Technical Architecture

### Technology Stack

| Katman | Teknoloji | Neden |
|--------|-----------|-------|
| Dil | Python 3.11+ | Mevcut kodla uyumlu, sentence-transformers/tree-sitter ekosistemi Python-first |
| MCP Framework | FastMCP (mcp[cli]) | ceaksan-knowledge ve mcp-code-search'te kanitlanmis |
| Storage | SQLite + sqlite-vec + FTS5 | Tek dosya, cross-project native SQL, backup = dosya kopyala |
| Embedding | multilingual-e5-base (768d) | 100+ dil, hem kod hem icerik, ceaksan'da kanitlanmis |
| AST Parsing | tree-sitter + tree-sitter-language-pack | 18 dil AST-aware chunking, mcp-code-search'ten reuse |
| CLI | Typer | Basit, tip-safe, auto-completion |
| Data Models | Pydantic v2 | Validation, serialization |
| Config | TOML (tomllib) | Python 3.11+ stdlib, human-readable |

### Why SQLite (Not LanceDB)

| Kriter | SQLite + sqlite-vec | LanceDB |
|--------|---------------------|---------|
| Olcek | <50K chunk icin ideal | Milyon+ chunk'ta avantajli |
| Cross-project | `WHERE project_id IN (...)` — native SQL | Ayri DB instance, birlestirme ek kod |
| FTS + Vector | Ayni DB'de FTS5 + sqlite-vec, tek transaction | FTS ayri cozum gerektirir |
| Backup | Tek dosya kopyala | Dizin kopyala (Arrow/Lance format) |
| Migration | SQL ALTER TABLE | Schema evolution sinirli |
| Dependency | sqlite3 stdlib + sqlite-vec (tek ek) | PyArrow + Lance format (agir) |
| Operasyonel | Tek .db dosyasi, SQLite CLI ile inspect | Ozel tooling gerektirir |

Olcek varsayimi: proje basi 2-8K chunk, toplam <50K. Bu olcekte LanceDB'nin columnar format overhead, avantaj degil.

### Decision Gate

| Criterion | Assessment |
|-----------|------------|
| **Benefit** | Fragmentli 3 sistemi birlestirme, cross-project arama, feedback loop |
| **Necessity** | Mevcut sistemler birbirinden habersiz, her proje icin ayri setup gerekiyor |
| **Burden** | Yeni proje ama mevcut kodun %60-70'i reuse. 8 tablo, 4 MCP tool |
| **Conflict** | mcp-code-search ve ceaksan-knowledge retire edilecek, gecis donemi gerekli |
| **Performance** | SQLite tek dosya I/O, embedding model 768d (mevcut 384d'den buyuk ama ceaksan'da sorunsuz) |
| **Security** | Lokal dosya, ag erisimi yok, kullanici verisi yok |
| **Bottleneck** | Embedding hesaplama (batch processing ile yonetilebilir) |
| **Currency** | sqlite-vec aktif gelistirme altinda, sentence-transformers stable |

### Alternatives Considered

**Track A — Monorepo, tek MCP server:** Config, graph, vector, code parsing tek lifecycle altinda kilitlenir. Buyudukce kirilgan.

**Track C — mcp-code-search'u genislet:** Code search icin optimize edilmis yapiya content graph ve cross-project orchestration eklemek cohesion bozar. Kavramsal olarak yanlis abstraction.

**LanceDB vs SQLite:** LanceDB milyon+ chunk olceginde avantajli. Bizim olcek proje basi birkac bin chunk, toplam <50K. Bu olcekte LanceDB'nin columnar format'i overhead. SQLite'da tek DB, cross-project query native SQL join.

**Hybrid TS+Python vs Pure Python:** TS wrapper ek complexity: iki build pipeline, subprocess bridge latency, iki debug story. Python FastMCP zaten MCP protocol'u tam destekliyor.

---

## Directory Structure

```
dnomia-knowledge/
├── src/dnomia_knowledge/
│   ├── __init__.py
│   ├── server.py              # FastMCP server — 4 tool definition
│   ├── chunker/
│   │   ├── __init__.py
│   │   ├── base.py            # Chunker protocol/interface
│   │   ├── ast_chunker.py     # tree-sitter (18 dil)
│   │   └── md_chunker.py      # heading-based + configurable overlap
│   ├── embedder.py            # multilingual-e5-base, query:/passage: prefix
│   ├── store.py               # SQLite + sqlite-vec + FTS5, tum CRUD
│   ├── search.py              # Hybrid: FTS5 BM25 + vec cosine + RRF + interaction bonus
│   ├── graph.py               # Edge builder + Louvain community detection + BFS traversal
│   ├── indexer.py             # Scan -> chunk -> embed -> store pipeline
│   ├── registry.py            # .knowledge.toml discovery + validation
│   ├── presets.py             # Extension preset definitions
│   └── models.py              # Pydantic data models
├── cli.py                     # Typer CLI: index, search, stats, gc, doctor
├── pyproject.toml
├── docs/
│   └── plans/
│       └── 2026-03-02-dnomia-knowledge-design.md  # bu dosya
└── tests/
    ├── test_store.py
    ├── test_chunker.py
    ├── test_search.py
    ├── test_embedder.py
    ├── test_indexer.py
    └── test_graph.py
```

---

## Database Schema

Tek SQLite veritabani: `~/.local/share/dnomia-knowledge/knowledge.db`

9 tablo: 6 fiziksel tablo (projects, file_index, chunks, edges, search_log, system_metadata) + 1 junction tablo (chunk_interactions) + 2 virtual tablo (chunks_fts, chunks_vec).

```sql
-- ============================================================
-- Performance PRAGMA'lari
-- ONEMLI: WAL haric hicbiri persistent degildir.
-- store.py her connect() cagrisinda bunlari yeniden set etmeli.
-- ============================================================
PRAGMA journal_mode = WAL;          -- Write-Ahead Logging: concurrent read + write (persistent)
PRAGMA synchronous = NORMAL;        -- WAL ile guvenli, fsync azaltir (per-connection)
PRAGMA temp_store = MEMORY;         -- Temp tablolari RAM'de tut (per-connection)
PRAGMA mmap_size = 67108864;        -- 64MB memory-mapped I/O (8GB M2 icin guvenli)
PRAGMA cache_size = -64000;         -- 64MB page cache (per-connection)

-- ============================================================
-- Proje kayitlari
-- ============================================================
CREATE TABLE projects (
    id TEXT PRIMARY KEY,                -- "ceaksan", "dnomia-app"
    path TEXT UNIQUE NOT NULL,          -- /Users/ceair/Desktop/ceaksan-v4.0
    type TEXT NOT NULL,                 -- "content" | "saas" | "static"
    graph_enabled INTEGER DEFAULT 0,
    last_indexed TEXT,
    config_hash TEXT                    -- .knowledge.toml MD5, stale detection
);

-- ============================================================
-- Dosya index (incremental reindex icin dedicated lookup)
-- chunks tablosunu taramadan "bu dosya degisti mi" sorusuna
-- O(1) cevap verir.
-- ============================================================
CREATE TABLE file_index (
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    file_hash TEXT NOT NULL,            -- MD5 content hash
    chunk_count INTEGER DEFAULT 0,
    last_indexed TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (project_id, file_path)
);

-- ============================================================
-- Chunk'lar (content + code ayni tabloda, chunk_domain ile ayrim)
-- ============================================================
CREATE TABLE chunks (
    id INTEGER PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    chunk_domain TEXT NOT NULL,          -- "content" | "code"
    chunk_type TEXT NOT NULL,            -- "heading" | "function" | "class" | "method"
                                        -- | "module" | "block" | "struct" | "interface"
    name TEXT,                          -- function/class adi veya heading text
    language TEXT,                      -- "python" | "typescript" | "mdx" | ...
    start_line INTEGER,
    end_line INTEGER,
    content TEXT NOT NULL,              -- ham chunk text
    metadata TEXT,                      -- JSON: tags, categories, imports, frontmatter, etc.
    created_at TEXT DEFAULT (datetime('now'))
);

-- ============================================================
-- FTS5 virtual table (content kolonu uzerinde full-text search)
-- Porter stemming + unicode61 tokenizer
-- ============================================================
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    content,
    content=chunks,
    content_rowid=id,
    tokenize='porter unicode61'
);

-- FTS5 sync trigger'lari
CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, content) VALUES (new.id, new.content);
END;
CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content) VALUES ('delete', old.id, old.content);
END;
CREATE TRIGGER chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content) VALUES ('delete', old.id, old.content);
    INSERT INTO chunks_fts(rowid, content) VALUES (new.id, new.content);
END;

-- ============================================================
-- Vector table (sqlite-vec, 768-dim float for multilingual-e5-base)
-- ============================================================
CREATE VIRTUAL TABLE chunks_vec USING vec0(
    id INTEGER PRIMARY KEY,
    embedding float[768]
);

-- vec0 sync trigger
-- NOT: vec0 INSERT trigger'i yok — embedding verisi gerektiginden
-- insert indexer.py pipeline'inda explicit yapilir (embed_passage sonrasi).
-- Ama DELETE trigger zorunlu: chunk silindiginde orphan embedding
-- kalir ve arama sonuclarinda hayalet sonuclar doner.
CREATE TRIGGER chunks_vec_ad AFTER DELETE ON chunks BEGIN
    DELETE FROM chunks_vec WHERE id = old.id;
END;

-- ============================================================
-- Edge'ler (schema'da her zaman mevcut, populate opsiyonel)
-- Graph disabled projelerde bu tablo bos kalir.
-- ============================================================
CREATE TABLE edges (
    source_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    target_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    edge_type TEXT NOT NULL,            -- "link" | "tag" | "category" | "semantic" | "import"
    weight REAL DEFAULT 1.0,
    metadata TEXT,                      -- JSON: shared tags, similarity score, etc.
    PRIMARY KEY (source_id, target_id, edge_type)
);

-- ============================================================
-- Arama kalitesi feedback loop
-- search_log: her arama kaydedilir
-- chunk_interactions: dosya erisim korelasyonu
--
-- Trade-off: result_chunk_ids JSON array olarak TEXT'te tutuluyor.
-- Aggregate query'ler (en cok aranan chunk'lar) icin JSON parse
-- gerekir. Bu bilincli bir karar — basitlik icin JSON ile basliyoruz.
-- Darboğaz olursa search_results junction tablosuna normalize edilir:
--   CREATE TABLE search_results (
--       search_id INTEGER REFERENCES search_log(id),
--       chunk_id INTEGER REFERENCES chunks(id),
--       rank INTEGER,
--       PRIMARY KEY (search_id, chunk_id)
--   );
-- ============================================================
CREATE TABLE search_log (
    id INTEGER PRIMARY KEY,
    query TEXT NOT NULL,
    project_id TEXT,                    -- NULL = cross-project search
    domain TEXT,                        -- "all" | "code" | "content"
    result_chunk_ids TEXT,              -- JSON array: [12, 45, 78]
    result_count INTEGER DEFAULT 0,
    timestamp TEXT DEFAULT (datetime('now'))
);

CREATE TABLE chunk_interactions (
    id INTEGER PRIMARY KEY,
    chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    interaction TEXT NOT NULL,          -- "read" | "edit" | "search_hit"
    source_tool TEXT,                   -- "Read" | "Edit" | "search"
    timestamp TEXT DEFAULT (datetime('now'))
);

-- ============================================================
-- Indexes
-- ============================================================
CREATE INDEX idx_chunks_project ON chunks(project_id);
CREATE INDEX idx_chunks_file ON chunks(file_path);
CREATE INDEX idx_chunks_domain ON chunks(chunk_domain);
CREATE INDEX idx_chunks_type ON chunks(chunk_type);
CREATE INDEX idx_edges_source ON edges(source_id);
CREATE INDEX idx_edges_target ON edges(target_id);
CREATE INDEX idx_edges_type ON edges(edge_type);
CREATE INDEX idx_search_log_project ON search_log(project_id);
CREATE INDEX idx_search_log_ts ON search_log(timestamp);
CREATE INDEX idx_interactions_chunk ON chunk_interactions(chunk_id);
CREATE INDEX idx_interactions_ts ON chunk_interactions(timestamp);

-- ============================================================
-- System metadata (global key-value store)
-- Embedding model takibi, schema version, vb.
-- Proje bazli degil, tum DB icin gecerli.
-- ============================================================
CREATE TABLE system_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
-- Ilk indexlemede:
--   INSERT INTO system_metadata VALUES ('embedding_model', 'intfloat/multilingual-e5-base');
--   INSERT INTO system_metadata VALUES ('embedding_dim', '768');
--   INSERT INTO system_metadata VALUES ('schema_version', '1');
-- doctor komutu:
--   SELECT value FROM system_metadata WHERE key = 'embedding_model'
--   → mevcut config'teki modelle karsilastir, uyumsuzluk varsa uyar
```

---

## Embedding Strategy

### Model: multilingual-e5-base

- 768 dimensions, 278M parameters
- 100+ language support (Turkce, Ingilizce, Fince, vb.)
- ceaksan-knowledge'da kanitlanmis
- Lokal model: `~/.cache/huggingface/` veya proje icinde

### Query/Passage Prefix (Kritik)

multilingual-e5-base training convention'i geregi:

```python
# Indeksleme sirasinda (chunk'lar icin)
def embed_passage(self, text: str) -> list[float]:
    return self._encode(f"passage: {text}")

# Arama sirasinda (query icin)
def embed_query(self, text: str) -> list[float]:
    return self._encode(f"query: {text}")
```

Prefix olmadan cosine similarity %10-15 duser. Bu ayrimi embedder.py'de enforce etmek zorunlu.

### Memory Optimization (M2 Air 8GB)

multilingual-e5-base float32'de ~1.1GB RAM kullanir. 8GB makinede bu kritik:

| Backend | Quantization | Model RAM | Inference Hizi | Reindex Gerekli mi |
|---------|-------------|-----------|---------------|---------------------|
| torch (varsayilan) | float32 | ~1.1GB | baseline | - |
| torch | float16 | ~560MB | ~1.5x | Hayir (ayni 768d) |
| ONNX | int8 | ~300MB | ~2-3x | Hayir (ayni 768d) |

Sprint 1'de torch float32 ile basla (en basit). Sprint 2'de ONNX int8'e gec.
Quantization embedding boyutunu degistirmez (768d), sadece bellek ve hizi iyilestirir.

**Lazy model unload:** 8GB'da model her zaman bellekte kalamaz.

```python
class Embedder:
    def _ensure_loaded(self):
        if self._model is None:
            self._model = SentenceTransformer(...)
            self._last_used = time.time()

    def maybe_unload(self, idle_minutes=10):
        if self._model and time.time() - self._last_used > idle_minutes * 60:
            del self._model
            self._model = None
            gc.collect()
```

**Embedding config — kesinlikle global, per-project DEGIL:**

Tum projeler ayni model + ayni boyut kullanmak zorunda. Farkli model = farkli embedding boyutu = karisik DB. Bu config `~/.config/dnomia-knowledge/config.toml`'da:

```toml
# ~/.config/dnomia-knowledge/config.toml
[embedding]
model = "intfloat/multilingual-e5-base"   # degisirse full re-embed
backend = "onnx"              # "torch" | "onnx"
quantization = "int8"         # "float32" | "float16" | "int8"
idle_unload_minutes = 10      # model bellekten atilma suresi
batch_size = 8                # 8GB icin 8, 16GB icin 16
```

Bu config `.knowledge.toml`'a KONULMAZ. Per-project embedding config yok.

### Caching

- In-memory query cache: MD5 key, 15 dakika TTL (ceaksan-knowledge pattern)
- Passage embedding cache: indeksleme sirasinda batch processing (batch_size configurable)

---

## Chunking Strategy

### Chunker Protocol (base.py)

```python
from typing import Protocol

class Chunk(BaseModel):
    content: str
    chunk_type: str        # "heading" | "function" | "class" | ...
    name: str | None
    language: str | None
    start_line: int
    end_line: int
    metadata: dict | None  # frontmatter, imports, tags, etc.

class Chunker(Protocol):
    def chunk(self, file_path: str, content: str) -> list[Chunk]: ...
```

### Markdown/MDX Chunker (md_chunker.py)

Kaynak: ceaksan-knowledge/index_content.py'den reuse.

- Heading-based split: `##` ve `###` seviyelerinde
- Kod bloklari butun halde korunur (ortasindan bolunmez)
- Configurable overlap: varsayilan 2 satir
- Kucuk chunk'lar (<100 token) bir sonraki ile merge edilir
- Frontmatter parse: title, description, tags, categories, tldr, keyTakeaways, faq
- Frontmatter metadata olarak chunk'a eklenir

### AST Chunker (ast_chunker.py)

Kaynak: mcp-code-search/chunker.py'den reuse.

- tree-sitter ile AST parse
- Target node types: function, class, method, struct, interface, enum, impl, module
- Class/struct icindeki method'lar ayri chunk olarak da cikarilir
- Fallback: AST parse basarisizsa sliding-window (max_chunk_lines=50, overlap=2)
- 18 dil destegi: Python, JavaScript, TypeScript, TSX, Go, Rust, Java, C, C++, C#, Ruby, PHP, Swift, Kotlin, Scala, Elixir, Dart, Zig

### Chunker Selection

`.knowledge.toml` config'ine gore:

```python
def get_chunker(file_path: str, config: ProjectConfig) -> Chunker:
    ext = Path(file_path).suffix
    if ext in [".md", ".mdx"]:
        return MdChunker(overlap=config.content.chunk_overlap)
    elif ext in config.code.resolved_extensions:
        return AstChunker(max_lines=config.code.max_chunk_lines)
    else:
        return SlidingWindowChunker()  # generic fallback
```

---

## Search Architecture

### Hybrid Search: FTS5 + Vector + RRF

```
query
  |
  ├── embed_query(query)
  |     └── chunks_vec: cosine similarity search
  |           └── ranked list B (vector)
  |
  ├── sanitize_query(query)
  |     └── chunks_fts: MATCH with BM25 ranking
  |           └── ranked list A (keyword)
  |
  └── RRF merge(A, B, k=60)
        |
        ├── interaction bonus (read + edit only, project-scoped)
        |
        └── final ranked results
```

### RRF (Reciprocal Rank Fusion)

```python
def rrf_merge(
    keyword_results: list[SearchResult],
    vector_results: list[SearchResult],
    k: int = 60,
    interaction_bonus_weight: float = 0.1,
    project_id: str | None = None,
) -> list[SearchResult]:
    scores: dict[int, float] = {}

    for rank, result in enumerate(keyword_results):
        scores[result.chunk_id] = scores.get(result.chunk_id, 0) + 1.0 / (k + rank + 1)

    for rank, result in enumerate(vector_results):
        scores[result.chunk_id] = scores.get(result.chunk_id, 0) + 1.0 / (k + rank + 1)

    # Interaction-based relevance boost
    # ONEMLI: Sadece "read" ve "edit" interaction'lari sayilir.
    # "search_hit" haric tutulur — aksi halde positive feedback loop olusur:
    # sik aranan chunk'lar surekli yukari cikar, hic aranmayanlar gomulur
    # (rich-get-richer etkisi).
    # ONEMLI: project_id ile scope'lanir. Cross-project aramada
    # ceaksan'da sik erisiilen chunk dnomia aramasinda bias uretmez.
    for chunk_id in scores:
        interaction_count = get_interaction_count(
            chunk_id,
            days=30,
            interactions=["read", "edit"],  # search_hit haric
            project_id=project_id,          # project-scoped
        )
        scores[chunk_id] += interaction_bonus_weight * min(interaction_count, 10) / 10

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

### Cross-Project Search

```
search(query, project="ceaksan", cross=True)
  |
  ├── ceaksan .knowledge.toml: links.related = ["dnomia-app"]
  |
  ├── search ceaksan → results_A (interaction boost: ceaksan-scoped)
  ├── search dnomia-app → results_B (interaction boost: dnomia-scoped)
  |
  └── Per-project normalization + RRF merge → final results
        1. Her projeden top-K al (K = limit / project_count)
        2. Proje ici skor normalize et
        3. RRF ile birlestir
        (project_id her sonucta belirtilir)
```

**Cross-project balance:** ceaksan'da 1800 chunk, leetty'de 200 chunk varsa, normalize edilmeden RRF'de ceaksan domine eder. Per-project top-K limiti bu dengesizligi onler.

### FTS5 Multi-Layer Fallback

Kaynak: claude-context-mode'dan ilham.

```
Layer 1: Porter stemming ile standart FTS5 MATCH
         "configuration" → "configur" kokunuyle eslesir
         |
         sonuc yoksa ↓
         |
Layer 2: Prefix match — her terime * eklenir
         "config*" → configuration, configure, configs
         |
         sonuc yoksa ↓
         |
Layer 3: Trigram-based fuzzy — kelimeyi 3-gram'lara bol, OR ile ara
         "confg" → "con" OR "onf" OR "nfg" (yazim hatasi toleransi)
```

---

## GraphRAG

### Edge Types

| Type | Aciklama | Populate Kosulu |
|------|----------|----------------|
| `link` | Markdown internal link (href parse) | Icerik dosyalarinda otomatik |
| `tag` | 2+ ortak tag'a sahip chunk'lar | tags metadata'si varsa |
| `category` | Ayni kategorideki chunk'lar | categories metadata'si varsa |
| `semantic` | KNN top-K komsular, cosine > threshold | graph.enabled = true |
| `import` | Kod dosyalarindaki import iliskileri | AST parse sirasinda |

### Graph Build Pipeline

```
1. Link edge'leri: markdown href parse → internal link'leri edge'e cevir
2. Tag edge'leri: 2+ ortak tag'a sahip chunk ciftlerini bul
3. Category edge'leri: ayni kategori uyelerini bagla
4. Semantic edge'leri (KNN, full pairwise DEGIL):
   Her chunk icin:
     SELECT id FROM chunks_vec
     WHERE embedding MATCH knn(chunk_embedding, 20)
     → top-20 komsu
     → cosine > threshold (varsayilan 0.75) olanlari edge olarak ekle
   Complexity: O(n * k) — n=chunk sayisi, k=20
   NOT: Naive pairwise O(n^2) KULLANILMAZ.
   2000 chunk icin: pairwise = 2M karsilastirma, KNN = 40K lookup.
5. Import edge'leri: AST'den cikarilan import/export iliskileri
6. Louvain community detection (yalnizca document-level node'lar uzerinde)
7. PageRank hesaplama (NetworkX, alpha=0.85)
8. Sonuclari edges tablosuna yaz, community/pagerank'i chunk metadata'sina ekle
```

### Graph Incremental Mode

Varsayilan olarak graph build incremental calisir:

```
Incremental (varsayilan):
  1. Sadece degisen chunk'larin embedding'leri alinir
  2. Bu chunk'lar icin:
     - Eski semantic edge'leri sil (WHERE source_id = chunk_id AND edge_type = 'semantic')
     - KNN top-K sorgusu yap
     - Yeni semantic edge'leri yaz
  3. Link/tag/category edge'leri sadece degisen dosyalar icin rebuild
  4. Louvain + PageRank: DEFERRED — index_project sirasinda calistirilmaz.
     Edge'ler guncellenir ama community detection ayri tetiklenir:
     - CLI: dnomia-knowledge rebuild-graph ceaksan
     - Veya cron ile gunluk
     graph_query mode=neighbors edge'ler uzerinden aninda calisir,
     sadece mode=communities eski community verisiyle doner.
     Bu trade-off kabul edilebilir: edge guncelleme anlik, community detection batch.

Full rebuild (--full veya semantic_threshold degistiginde):
  1. edges tablosunu tamamen temizle
  2. Tum chunk'lar icin pipeline'i sifirdan calistir
```

Incremental mode sayesinde tek dosya degisikliginde full graph rebuild gereksiz.

### Graph Query

BFS traversal (mode="neighbors"): verilen chunk ID'den baslayarak depth 1-3 komsulari getir.

Community listing (mode="communities"): Louvain ile tespit edilmis tematik kumeleri listele. Bu modda chunk ID gerekmez, project parametresi kullanilir.

### Graph Activation

`.knowledge.toml`'da `graph.enabled = true` olan projelerde:
- `index_project` sonrasinda graph otomatik build edilir (incremental)
- Graph disabled projelerde edges tablosu bos kalir
- Cross-project graph yok — her projenin graph'i kendi icinde

---

## MCP Tools

4 tool. Her tool'un context maliyeti minimize edilmis.

### search

```python
@server.tool()
async def search(
    query: str,                          # Arama sorgusu
    domain: str = "all",                 # "all" | "code" | "content"
    project: str | None = None,          # None = aktif proje (CLAUDE_PROJECT_DIR)
    cross: bool = False,                 # True = linked projelerde de ara
    limit: int = 10,                     # Max sonuc
) -> str:
    """Hybrid semantic + keyword search across project knowledge."""
```

**Donus formati:**

```
[1] src/auth/middleware.ts:15-42 (score: 0.87, project: dnomia-app)
    function authMiddleware — Validates JWT token and sets user context
    ```typescript
    export async function authMiddleware(req: Request) {
      // ... snippet ...
    }
    ```

[2] src/content/posts/auth-guide.mdx:## Token Yonetimi (score: 0.73, project: ceaksan)
    JWT token lifecycle: olusturma, yenileme, iptal etme sureclerini anlatir
```

### graph_query

```python
@server.tool()
async def graph_query(
    id: int | None = None,               # Chunk ID (neighbors icin zorunlu)
    project: str | None = None,           # Project ID (communities icin zorunlu)
    mode: str = "neighbors",              # "neighbors" | "communities"
    depth: int = 1,                       # BFS depth (1-3), sadece neighbors icin
) -> str:
    """Traverse knowledge graph edges from a chunk or list communities.

    mode="neighbors": id zorunlu, project opsiyonel.
    mode="communities": project zorunlu, id opsiyonel (verilirse o chunk'in community'si).
    """
```

### index_project

```python
@server.tool()
async def index_project(
    path: str,                           # Proje root path
    incremental: bool = True,            # False = full reindex
) -> str:
    """Index or reindex a project's content and code."""
```

**Donus formati:**

```
Indexed ceaksan: 245 files, 1,823 chunks (1,204 content + 619 code)
  Changed: 12 files, 89 chunks updated
  Graph: 3,456 edges (234 link, 567 tag, 890 category, 1,234 semantic, 531 import)
  Duration: 14.2s
```

### project_info

```python
@server.tool()
async def project_info(
    project: str | None = None,          # None = tum projeler
) -> str:
    """List registered projects with stats."""
```

---

## Per-Project Configuration

### .knowledge.toml

Her projenin root'unda bulunur. Tek kaynak (single source of truth).

```toml
# ============================================================
# ceaksan-v4.0/.knowledge.toml
# ============================================================

[project]
name = "ceaksan"
type = "content"                    # "content" | "saas" | "static"

[content]
paths = ["src/content/posts/", "src/content/courses/"]
extensions = [".mdx", ".md"]
chunking = "heading"                # "heading" | "sliding"
chunk_overlap = 2                   # satir overlap

[code]
preset = "web"                      # "web" | "python" | "django" | "mixed"
paths = ["src/"]
max_chunk_lines = 50
# extensions = [".custom"]          # override varsa preset ignore edilir

[graph]
enabled = true
edge_types = ["link", "tag", "category", "semantic"]
semantic_threshold = 0.75

[links]
related = ["dnomia-app"]            # cross-project search targets

[indexing]
ignore_patterns = ["node_modules", "dist", ".next", "__pycache__"]
max_file_size_kb = 500
batch_size = 16                     # embedding batch size
```

```toml
# ============================================================
# dnomia_app/.knowledge.toml
# ============================================================

[project]
name = "dnomia-app"
type = "saas"

[content]
paths = ["docs/", "tasks/"]
extensions = [".md"]
chunking = "heading"

[code]
preset = "web"
paths = ["apps/", "packages/"]
max_chunk_lines = 50

[graph]
enabled = false

[links]
related = ["ceaksan"]
```

### Extension Presets

Core'da hardcoded, override mumkun:

```python
PRESETS = {
    "web": [".ts", ".tsx", ".js", ".jsx", ".astro", ".vue", ".svelte",
            ".css", ".scss", ".html"],
    "python": [".py", ".pyi"],
    "django": [".py", ".pyi", ".html", ".txt"],
    "mixed": [],  # web + python birlesimi, runtime'da resolve edilir
}
```

`extensions` explicit verilirse preset ignore edilir. Verilmezse preset'ten gelir.

### Project Discovery

registry.py iki modda calisir:

1. **Explicit:** `~/.config/dnomia-knowledge/projects.toml` — sadece path listesi

```toml
[[projects]]
path = "/Users/ceair/Desktop/ceaksan-v4.0"

[[projects]]
path = "/Users/ceair/Desktop/dnomia/dnomia_app"
```

2. **Auto-discovery:** Configurable glob pattern ile `.knowledge.toml` taramasi

```toml
[discovery]
roots = ["/Users/ceair/Desktop"]
depth = 2                           # max directory depth
```

Her iki modda da `.knowledge.toml` tek kaynak. projects.toml sadece path pointer.

---

## Indexing Pipeline

### Akis

```
index_project(path, incremental=True)
  |
  ├── 1. registry.load_config(path)     → ProjectConfig
  |
  ├── 2. scan_files(config)             → list[FilePath]
  |     ├── os.walk + ignore_patterns
  |     ├── .gitignore respect (pathspec)
  |     ├── binary extension skip
  |     └── max_file_size_kb filter
  |
  ├── 3. detect_changes(files, file_index)
  |     ├── MD5 content hash her dosya icin
  |     ├── file_index tablosundan mevcut hash'leri cek
  |     ├── changed = hash farki olanlar + yeni dosyalar
  |     └── deleted = file_index'te olup scan'da olmayan
  |
  ├── 4. cleanup(deleted)
  |     ├── DELETE FROM chunks WHERE file_path IN (deleted)
  |     |     → trigger: chunks_fts DELETE (FTS5 sync)
  |     |     → trigger: chunks_vec DELETE (vec0 sync)
  |     |     → CASCADE: edges, chunk_interactions
  |     └── DELETE FROM file_index WHERE file_path IN (deleted)
  |
  ├── 5. process(changed)
  |     ├── for each file:
  |     |     ├── DELETE existing chunks for this file (trigger'lar fts+vec temizler)
  |     |     ├── chunker = get_chunker(file_path, config)
  |     |     ├── chunks = chunker.chunk(file_path, content)
  |     |     ├── embeddings = embedder.embed_passage(chunks)  # batch
  |     |     ├── INSERT INTO chunks (trigger: chunks_fts INSERT)
  |     |     ├── INSERT INTO chunks_vec (explicit, embedding ile)
  |     |     └── UPSERT file_index
  |     └── commit transaction
  |
  ├── 6. build_graph(config, changed_chunk_ids)  # sadece graph.enabled = true
  |     ├── incremental=True (varsayilan):
  |     |     ├── sadece degisen chunk'lar icin edge rebuild
  |     |     ├── KNN semantic edge (top-20, threshold filter)
  |     |     └── Louvain + PageRank: tum graph uzerinde yeniden
  |     ├── incremental=False:
  |     |     ├── DELETE FROM edges WHERE project chunk'lari
  |     |     └── tum pipeline sifirdan
  |     ├── link edges: markdown href parse
  |     ├── tag edges: ortak tag tespit
  |     ├── category edges: ayni kategori
  |     ├── semantic edges: KNN top-K + threshold
  |     ├── import edges: AST import/export
  |     ├── Louvain community detection
  |     └── PageRank
  |
  └── 7. update_project_metadata()
        ├── UPDATE projects SET last_indexed, config_hash
        └── return IndexResult
```

### Incremental Reindex

- `file_index` tablosundan `file_hash` cek: O(1) lookup per dosya
- MD5 content hash: dosya icerigini oku, hashle, karsilastir
- Sadece degisen dosyalar reprocess edilir
- Full reindex: `incremental=False` — tum chunks/edges/file_index temizlenip sifirdan

### Embedding Model Degisikligi

- `projects.embedding_model` kolonu yok (tek model enforce edilir)
- Eger model degisirse: tum chunks_vec tablosu temizlenip yeniden embed edilir
- Bu durum `cli.py doctor` komutuyla tespit edilir

---

## Feedback Loop

### Interaction Tracking

PostToolUse hook'ta:

```python
# Read tool kullanildiginda
if tool == "Read" and file_path:
    chunk_ids = store.get_chunks_for_file(project_id, file_path)
    for chunk_id in chunk_ids:
        store.log_interaction(chunk_id, "read", "Read")

# Edit tool kullanildiginda
if tool == "Edit" and file_path:
    chunk_ids = store.get_chunks_for_file(project_id, file_path)
    for chunk_id in chunk_ids:
        store.log_interaction(chunk_id, "edit", "Edit")
```

### Relevance Boost

Arama sirasinda, son 30 gundeki interaction count'u RRF score'a bonus olarak eklenir.

**Kritik kurallar:**

1. **Sadece `read` ve `edit` sayilir.** `search_hit` haric tutulur. Aksi halde positive feedback loop olusur: sik aranan chunk'lar surekli yukari cikar, hic aranmayanlar gomulur (rich-get-richer etkisi).

2. **Project-scoped.** Cross-project aramada her projenin interaction'i ayri hesaplanir. ceaksan'da sik erisiilen bir chunk, dnomia aramasinda bias uretmez.

```python
interaction_count = get_interaction_count(
    chunk_id,
    days=30,
    interactions=["read", "edit"],  # search_hit haric
    project_id=project_id,          # project-scoped
)
bonus = interaction_bonus_weight * min(interaction_count, 10) / 10
# interaction_bonus_weight = 0.1 (varsayilan)
# max 10 interaction'a kadar linear artis, sonra plateau
```

---

## CLI

```bash
# Proje indeksleme
dnomia-knowledge index /path/to/project
dnomia-knowledge index /path/to/project --full    # full reindex

# Arama
dnomia-knowledge search "auth middleware" --project ceaksan
dnomia-knowledge search "auth middleware" --domain code
dnomia-knowledge search "auth middleware" --cross   # linked projeler dahil

# Proje bilgisi
dnomia-knowledge projects                          # tum projeler
dnomia-knowledge stats ceaksan                     # detayli stats

# Bakim
dnomia-knowledge gc                                # orphan chunks, stale data temizligi
dnomia-knowledge doctor                            # config validation, embedding check
```

### doctor Checklist (Scope — Sprint 4)

`doctor` komutu asagidaki kontrolleri yapar. Bu liste sabittir, Sprint 1-3'te genisletilmez:

| # | Kontrol | Basarisizlik Durumu |
|---|---------|---------------------|
| 1 | `.knowledge.toml` parse edilebiliyor mu | Syntax hatasi raporla |
| 2 | `config_hash` guncel mi | Stale: "config degisti, reindex gerekli" |
| 3 | Embedding model yuklu / erisilebilir mi | Model path kontrol, download onerisi |
| 4 | `chunks_vec` satir sayisi = `chunks` satir sayisi mi | Orphan embedding veya eksik embedding tespit |
| 5 | `file_index`'teki dosyalar disk'te var mi | Silinmis dosyalar icin "gc calistir" onerisi |
| 6 | `sqlite-vec` extension yuklu mu | Import hatasi raporla |
| 7 | Global config (`~/.config/dnomia-knowledge/config.toml`) gecerli mi | Eksik/hatali config raporla |

---

## MCP Server Registration

### Global (tum projelerde erisim)

`~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "knowledge": {
      "command": "/Users/ceair/Desktop/dnomia-knowledge/.venv/bin/python",
      "args": ["-m", "dnomia_knowledge.server"],
      "cwd": "/Users/ceair/Desktop/dnomia-knowledge"
    }
  }
}
```

### Per-Project (projeye ozel)

`<project-root>/.mcp.json`:

```json
{
  "mcpServers": {
    "knowledge": {
      "command": "/Users/ceair/Desktop/dnomia-knowledge/.venv/bin/python",
      "args": ["-m", "dnomia_knowledge.server"],
      "env": {
        "DNOMIA_KNOWLEDGE_PROJECT": "ceaksan"
      }
    }
  }
}
```

`DNOMIA_KNOWLEDGE_PROJECT` env var'i set edilmisse, `search` tool'unda `project` parametresi varsayilan olarak bu degeri alir.

---

## Reuse Map

Mevcut koddan neler alinacak:

| Modul | Kaynak | Ne Alinacak | Adaptasyon |
|-------|--------|-------------|------------|
| AST Chunking | `mcp-code-search/chunker.py` | tree-sitter parsing, CHUNK_NODE_TYPES, language detection, method extraction | Chunk model'e uyarla, sliding-window fallback'i koru |
| Heading Chunking | `ceaksan-knowledge/index_content.py` | Heading split, code block korunmasi, frontmatter parse, kucuk chunk merge | Chunk model'e uyarla, overlap configurable yap |
| GraphRAG | `ceaksan-knowledge/build_graph.py` | Louvain, PageRank, edge type definitions, community detection | SQLite'a uyarla (JSON graph yerine edges tablosu), KNN semantic edge |
| Embedding | `ceaksan-knowledge/mcp_server.py` | multilingual-e5-base loading, query cache, passage/query prefix | sqlite-vec'e uyarla |
| File Scanning | `mcp-code-search/indexer.py` | gitignore respect (pathspec), binary skip, size limit, hidden dir rules | Aynen kullan |
| MCP Server | `ceaksan-knowledge/mcp_server.py` | FastMCP pattern, tool definitions, error handling | 4 tool'a sadele |
| Hybrid Search | `mcp-code-search/store.py` | RRF merge algorithm, score normalization | sqlite-vec + FTS5 versiyonuna uyarla |

---

## Migration Strategy

### Phase 1: dnomia-knowledge Calisiyor

- Yeni proje olustur, core moduller implement et
- ceaksan ve dnomia-app'i indeksle
- Mevcut mcp-code-search ve ceaksan-knowledge paralel calisir

### Phase 2: Gecis

- dnomia-knowledge stable, tum projeler indeksli
- `~/.claude/settings.json`'dan `mcp-code-search` kaldir
- `ceaksan-v4.0/.mcp.json`'dan `ceaksan-knowledge` kaldir
- Her iki eski projeyi archive et

### Phase 3: Retire

- mcp-code-search repo'su archive
- ceaksan-knowledge dizini sil
- Eski LanceDB data dizinleri temizle

---

## Implementation Priorities

Onerilen siralama (en kritik ve en az edge case'li olandan basla):

### Sprint 1: Calisan Iskelet

1. `store.py` — SQLite schema olusturma, PRAGMA'lar, temel CRUD, trigger'lar
2. `embedder.py` — multilingual-e5-base, query/passage prefix
3. `md_chunker.py` — heading-based chunking (basit, az edge case)
4. `indexer.py` — tek bir markdown dosyasini indeksleyebilecek pipeline
5. `search.py` — FTS5 + vec hybrid, RRF merge
6. `server.py` — `search` ve `index_project` tool'lari

Bu sprint sonunda: tek bir markdown dosyasini indeksle, ara, sonuc al.

### Sprint 2: Tam Indeksleme

7. `registry.py` — .knowledge.toml parse, project discovery
8. `presets.py` — extension preset definitions
9. `cli.py` — index, search, stats komutlari
10. Incremental reindex (file_index tablosu + MD5 hash)
11. Multi-project support

### Sprint 3: AST + Graph

12. `ast_chunker.py` — tree-sitter entegrasyonu (en cok edge case burada)
13. `graph.py` — edge builder + KNN semantic + Louvain + PageRank + incremental mode
14. Cross-project search (links.related)

### Sprint 4: Feedback + Polish

15. `chunk_interactions` + PostToolUse hook
16. Interaction-based relevance boost (read/edit only, project-scoped)
17. `search_log` kayit
18. `gc` ve `doctor` CLI komutlari
19. Migration: eski sistemleri retire et

---

## Dependencies

```toml
[project]
name = "dnomia-knowledge"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "mcp[cli]>=1.13.0",                  # FastMCP framework
    "sentence-transformers>=3.0.0",       # multilingual-e5-base
    "sqlite-vec>=0.1.6",                  # Vector search for SQLite
    "tree-sitter>=0.24.0",               # AST parsing (Sprint 3)
    "tree-sitter-language-pack>=0.7.0",   # Pre-built grammars (Sprint 3)
    "pathspec>=0.12.0",                   # .gitignore pattern matching
    "pydantic>=2.0.0",                    # Data models
    "typer>=0.12.0",                      # CLI framework
    "networkx>=3.0",                      # Graph algorithms (Sprint 3)
    "python-louvain>=0.16",              # Community detection (Sprint 3)
    "python-frontmatter>=1.0.0",          # MDX/MD frontmatter parsing
    "pyyaml>=6.0",                        # YAML support
    "rich>=13.0.0",                       # CLI output formatting
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.24.0",
    "ruff>=0.8.0",
]
```

---

## Risk Register

| Risk | Etki | Olasilik | Mitigation |
|------|------|----------|------------|
| sqlite-vec buyuk index'lerde yavas | Yuksek | Dusuk (<50K chunk) | Benchmark yap, gerekirse HNSW index parametrelerini tune et |
| tree-sitter edge case'leri (malformed code) | Orta | Yuksek | Sliding-window fallback her zaman aktif, AST fail = graceful degrade |
| Embedding model memory kullanimi (768d, 278M param) | Orta | Dusuk (M2 8GB'da test edilmis) | Batch size configurable, lazy model loading |
| .knowledge.toml config drift | Dusuk | Orta | `doctor` komutu ile validation, config_hash ile stale detection |
| FTS5 porter stemmer Turkce'de zayif | Orta | Kesin | Vector search bu boslugu kapatir (hybrid arama), ileride Turkce tokenizer eklenebilir |
| Cross-project search latency | Dusuk | Dusuk (2-3 proje) | Parallel query, sonuc limit per proje |
| Semantic edge build yavas (buyuk proje) | Orta | Dusuk | KNN top-K (O(n*k)) kullan, pairwise (O(n^2)) asla. Incremental mode varsayilan. |
| Interaction boost bias | Dusuk | Orta | search_hit haric tut, project-scoped, max 10 cap |

---

## Future Considerations (Out of Scope)

Asagidakiler bu versiyonun scope'unda degil ama ileride eklenebilir:

- **Context guard hook:** Buyuk tool ciktilarini intercept et, indeksle, ozet don. Ayri proje.
- **Reranker:** Cross-encoder ile sonuc yeniden siralama (ornegin `cross-encoder/ms-marco-MiniLM-L-6-v2`)
- **Turkce tokenizer:** FTS5 icin custom tokenizer (suan porter stemmer Ingilizce-optimize)
- **Session graph:** Oturum icinde hangi chunk'lar birlikte sorgulandiginin kaydi
- **PostToolUse auto-reindex:** Dosya edit'lendiginde otomatik incremental reindex hook'u
- **Embedding model upgrade:** `multilingual-e5-large` veya `gte-multilingual-base` gecisi
