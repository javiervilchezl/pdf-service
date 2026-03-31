# Servicio de Analisis de PDF

Microservicio FastAPI para procesamiento de documentos PDF.

## Responsabilidad

- recibir PDF por carga de archivo
- extraer y limpiar texto
- generar resumen
- inferir temas principales
- devolver salida estructurada para consumo del gateway

## Endpoint

- `POST /analyze-pdf`
- tipo: `multipart/form-data`
- campo requerido: `file`

Respuesta esperada (resumen):

```json
{
	"text": "...",
	"summary": "...",
	"topics": ["..."]
}
```

## Variables de entorno

- `PROVIDER` (`openai` o `groq`)
- `OPENAI_API_KEY`
- `OPENAI_MODEL`
- `GROQ_API_KEY`
- `GROQ_MODEL`

## Ejecucion local

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8001
```

## Docker

```bash
docker compose up --build
```

## Pruebas

```bash
pip install -r requirements-dev.txt
pytest
```

Cobertura configurada al 100%.

