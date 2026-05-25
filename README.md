# Better Call Chemo

Asistente fiscal mexicano basado en RAG (Retrieval-Augmented Generation). Responde preguntas sobre el SAT consultando documentos oficiales.

## Requisitos

- Python 3.10+
- Una API key de Gemini (o cualquier proveedor compatible con OpenAI)

## Instalación

```bash
pip install -r requirements.txt
```

Copia `.env.example` a `.env` y llena tu API key:

```bash
cp .env.example .env
```

## Uso

Primero, coloca los PDFs del SAT en la carpeta `data/sat/`.

Luego, indexa los documentos (solo necesitas hacerlo una vez):

```bash
python ingest.py
```

Finalmente, inicia la aplicación:

```bash
streamlit run app.py
```
