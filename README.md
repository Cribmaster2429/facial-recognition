# Facial Recognition System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

A real time facial recognition service in Python. It detects faces, encodes them as 128 dimension embeddings and matches them, through a REST API or a CLI.

## Features

- **Real-time Face Detection** - Detect faces using HOG (fast, CPU) or CNN (accurate, GPU) models
- **Face Recognition** - 128 dimension face embeddings from dlib's ResNet model, matched by distance with a configurable tolerance
- **REST API** - FastAPI with typed Pydantic schemas and OpenAPI docs at /docs
- **CLI Tool** - Command line interface for every API operation
- **Webcam Support** - Real-time recognition from webcam feed
- **Persistent Storage** - SQLite database for face embeddings
- **Docker Ready** - Containerized deployment with docker-compose

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Cribmaster2429/facial-recognition.git
cd facial-recognition

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

### Basic Usage

**Register a face:**
```bash
facial-recognition register "John Doe" path/to/photo.jpg
```

**Recognize faces in an image:**
```bash
facial-recognition recognize path/to/image.jpg --output result.jpg
```

**Start the API server:**
```bash
facial-recognition serve
# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

**Real-time webcam recognition:**
```bash
facial-recognition webcam
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/faces/register` | Register a new person |
| `GET` | `/faces` | List all registered persons |
| `GET` | `/faces/{id}` | Get person by ID |
| `DELETE` | `/faces/{id}` | Delete a person |
| `POST` | `/recognize` | Recognize faces in image |
| `POST` | `/recognize/detect` | Detect faces without recognition |
| `POST` | `/recognize/verify/{id}` | Verify face against specific person |
| `GET` | `/health` | Health check |

### Example API Usage

```bash
# Register a face
curl -X POST "http://localhost:8000/faces/register" \
  -F "name=John Doe" \
  -F "image=@photo.jpg"

# Recognize faces
curl -X POST "http://localhost:8000/recognize" \
  -F "image=@test.jpg"
```

## CLI Commands

```
facial-recognition --help

Commands:
  detect     Detect faces in an image
  register   Register a person from an image
  recognize  Recognize faces in an image
  list       List all registered persons
  delete     Delete a registered person
  serve      Start the REST API server
  webcam     Run real-time recognition from webcam
```

## Docker Deployment

```bash
# Build and run with docker-compose
docker-compose up -d

# Or build manually
docker build -t facial-recognition .
docker run -p 8000:8000 facial-recognition
```

## Architecture

```
src/facial_recognition/
├── core/               # Core recognition logic
│   ├── detector.py     # Face detection (HOG/CNN)
│   ├── encoder.py      # Face encoding (128-dim embeddings)
│   ├── recognizer.py   # Face matching and identification
│   └── models.py       # Pydantic data models
├── api/                # FastAPI REST API
│   ├── app.py          # Application factory
│   ├── routes/         # API endpoints
│   └── schemas.py      # Request/response schemas
├── cli/                # Typer CLI
│   └── commands.py     # CLI commands
├── storage/            # Persistence layer
│   ├── base.py         # Abstract storage interface
│   ├── sqlite.py       # SQLite implementation
│   └── memory.py       # In-memory (testing)
└── utils/              # Utilities
    ├── image.py        # Image processing
    └── logging.py      # Logging configuration
```

## Configuration

Configuration via environment variables or `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `DETECTION_MODEL` | `hog` | Detection model: `hog` or `cnn` |
| `ENCODING_MODEL` | `large` | Encoding model: `small` or `large` |
| `TOLERANCE` | `0.6` | Face matching tolerance (lower = stricter) |
| `DATABASE_URL` | `sqlite+aiosqlite:///./data/faces.db` | Database connection |
| `API_HOST` | `0.0.0.0` | API server host |
| `API_PORT` | `8000` | API server port |
| `LOG_LEVEL` | `INFO` | Logging level |

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=src/facial_recognition --cov-report=html

# Lint and format
ruff check src/ tests/
ruff format src/ tests/

# Type checking
mypy src/
```

## Technology Stack

- **Face Detection/Recognition**: [face_recognition](https://github.com/ageitgey/face_recognition) (dlib)
- **Computer Vision**: OpenCV
- **API Framework**: FastAPI
- **CLI Framework**: Typer
- **Data Validation**: Pydantic
- **Database**: SQLite with aiosqlite
- **Testing**: pytest

## License

MIT License - see [LICENSE](LICENSE) for details.

## Author

**Danson Wachira**
- Portfolio: [dansonwachira.dev](https://dansonwachira.dev)
- GitHub: [@Cribmaster2429](https://github.com/Cribmaster2429)
