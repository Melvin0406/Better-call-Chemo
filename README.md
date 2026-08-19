# Better Call Chemo

Asistente legal y fiscal mexicano construido sobre RAG (Retrieval-Augmented Generation). Responde preguntas sobre legislación federal mexicana citando los textos oficiales, en lugar de depender de lo que el modelo "recuerde".

**El problema:** preguntarle a un LLM genérico "¿qué deducciones personales puedo aplicar?" produce respuestas plausibles pero frecuentemente desactualizadas o inventadas — un riesgo real cuando se trata de obligaciones fiscales. Este proyecto acota el modelo a un corpus de 37 leyes y códigos federales vigentes, de modo que cada respuesta se construya sobre el texto legal recuperado.

---

## Cómo funciona

```
Pregunta del usuario
      ↓
parse_filters()      ← extrae filtros (/fiscal, /lft) y limpia la pregunta
      ↓
expand_query()       ← el LLM genera variantes de la pregunta en español
      ↓
retrieve()           ← búsqueda vectorial en FAISS (k=10 por variante)
      ↓
filter_results()     ← descarta chunks fuera del área o documento pedido
      ↓
rerank()             ← cross-encoder reordena y se queda con los 4 mejores
      ↓
Assistant.ask()      ← arma el prompt con los fragmentos + historial y llama al LLM
      ↓
Respuesta citando documento y página
```

Dos decisiones que mueven la aguja en calidad de respuesta:

- **Reranking en dos etapas.** La búsqueda vectorial recupera 10 candidatos y un cross-encoder (`ms-marco-MiniLM-L-6-v2`) los reordena para quedarse con 4. La similitud de embeddings sola trae fragmentos temáticamente cercanos pero irrelevantes; el cross-encoder compara pregunta y fragmento directamente.
- **Expansión de consulta.** Una sola redacción de la pregunta recupera poco: el lenguaje coloquial ("me corrieron del trabajo") no se parece al del texto legal ("rescisión de la relación laboral"). El LLM genera variantes antes de buscar.

El índice se construye una sola vez con `ingest.py` y se persiste en disco (FAISS + pickle), así la app arranca sin reprocesar los PDFs.

---

## Stack

| Componente | Tecnología |
|---|---|
| Índice vectorial | FAISS |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` (multilingüe, para corpus en español) |
| Reranking | CrossEncoder `ms-marco-MiniLM-L-6-v2` |
| Chunking | LangChain `RecursiveCharacterTextSplitter` (512 / overlap 64) |
| Extracción de PDF | PyMuPDF |
| LLM | Compatible con la API de OpenAI (probado con Gemini vía endpoint compatible) |
| Interfaz | Streamlit |

---

## Corpus

37 documentos federales mexicanos, clasificados automáticamente por área a partir del nombre del archivo (`CATEGORY_MAP` en `rag.py`):

| Área | Documentos |
|---|---|
| Fiscal | CFF, LISR, LIVA, LIEPS, RMF |
| Laboral | LFT, LSS, LFTSE, LISSSTE, LIFNVT |
| Civil | CCF, CNPCF, CFPC, LFDA, LFPCA |
| Penal | CPF, CNPP, LFCDO, LFRA |
| Mercantil | CCom, LGSM, LGTOC, LIC, LFPDPPP |
| Administrativo | LFPA, LAmp, LFPRH, LGRA |
| Otros | CPEUM, LFPPI, LFCE, LFPC, LGS, LGDNNA, LGAMVLV, LGEEPA, LGDFS |

Esta clasificación es la que habilita los filtros: al escribir `/laboral` al final de una pregunta, la búsqueda se restringe a esa área.

---

## Instalación

### 1. Dependencias

```bash
pip install -r requirements.txt
```

### 2. Configurar el LLM

```bash
cp .env.example .env
```

Edita `.env` con tu proveedor. Funciona con cualquier endpoint compatible con la API de OpenAI:

```
OPENAI_API_KEY=tu-api-key
OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
MODEL=gemini-2.0-flash
```

### 3. Conseguir los documentos

Los PDFs no se versionan en el repositorio (pesan demasiado). Dos opciones:

- **Corpus ya armado:** [carpeta de Drive](https://drive.google.com/drive/folders/1fyfnB8QfeAXM-SmjnQX9ENeuQRDS5Oii?usp=sharing) — descárgala en `data/sat/`.
- **Armarlo tú:** descarga las leyes desde el [portal de la Cámara de Diputados](https://www.diputados.gob.mx/LeyesBiblio/index.htm) y guárdalas en `data/sat/` o `data/leyes/`. **El nombre del archivo importa:** debe coincidir con la abreviatura de la tabla de arriba (`LFT.pdf`, `CFF.pdf`, `LISR.pdf`…) para que se clasifique en el área correcta. Un archivo con otro nombre se indexa igual, pero cae en la categoría genérica `ley_federal`.

### 4. Construir el índice

```bash
python ingest.py
```

Imprime cuántas páginas, documentos y chunks procesó, y guarda `index/leyes.faiss` e `index/leyes_chunks.pkl`. Solo hay que repetirlo si cambian los documentos.

### 5. Correr la aplicación

```bash
streamlit run app.py
```

---

## Uso

Preguntas en lenguaje natural:

```
¿Qué derechos tengo si me despiden sin causa justificada?
¿Cuándo estoy obligado a cobrar IVA?
¿Cuáles son mis deducciones personales como asalariado?
```

Filtros opcionales al final de la pregunta:

| Tipo | Valores |
|---|---|
| Por área | `/fiscal` `/laboral` `/penal` `/civil` `/mercantil` `/administrativo` `/constitucional` |
| Por documento | `/cff` `/lisr` `/liva` `/rmf` `/lft` `/cpf` |

```
¿Qué es el finiquito? /laboral
```

> Las respuestas son orientativas y no constituyen asesoría legal. Para casos concretos, consulta a un abogado o contador.

---

## Configuración avanzada

Todo se puede ajustar por variables de entorno sin tocar el código (`resolve_config` en `rag.py`):

| Variable | Default |
|---|---|
| `EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `512` / `64` |
| `TOP_K` | `4` |
| `DATA_DIRS` | `data/sat,data/leyes` |

---

## Estructura

```
├── rag.py           # Núcleo: carga, chunking, índice, retrieval, rerank, clase Assistant
├── ingest.py        # Construye y persiste el índice FAISS en disco
├── app.py           # Interfaz de chat en Streamlit
├── data/            # PDFs (no versionados)
└── index/           # Índice generado (no versionado)
```

---

## Autoría

Proyecto de equipo desarrollado para la materia de Procesamiento de Lenguaje Natural (CETYS Universidad, 2026), junto con [CheminGod](https://github.com/CheminGod), [DECastaV](https://github.com/DECastaV) y [M-Alcantar](https://github.com/M-Alcantar).

---

## Licencia

MIT — ver [LICENSE](LICENSE).
