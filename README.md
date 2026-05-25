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

Construir el índice de RAG (solo una vez o cuando cambien los documentos):

```bash
python ingest.py
```
