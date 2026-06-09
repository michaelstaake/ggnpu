# Docker

## Build

```bash
# Builder image (for kernel compilation)
docker build -t ggnpu-builder -f docker/Dockerfile.builder .

# Runtime image
docker build -t ggnpu:latest -f docker/Dockerfile .
```

## Run

```bash
docker run --rm \
  --device=/dev/accel/accel0 \
  --group-add render \
  --ulimit memlock=-1:-1 \
  -v /usr/lib/firmware/amdnpu:/usr/lib/firmware/amdnpu:ro \
  -v $HOME/.cache/ggnpu:/root/.cache/ggnpu \
  -v /path/to/models:/models:ro \
  ggnpu:latest -m /models/model.gguf -p "Hello"
```

## Docker Compose

```yaml
version: '3.8'
services:
  ggnpu:
    image: ggnpu:latest
    devices:
      - /dev/accel/accel0
    group_add:
      - render
    ulimits:
      memlock:
        soft: -1
        hard: -1
    volumes:
      - /usr/lib/firmware/amdnpu:/usr/lib/firmware/amdnpu:ro
      - $HOME/.cache/ggnpu:/root/.cache/ggnpu
      - ./models:/models:ro
    command: ["-m", "/models/model.gguf", "-p", "Hello", "-n", "64"]
```

## Notes

- Use native Docker (not Docker Desktop VM)
- Match XRT version to host driver
- Prebuilt xclbins shipped in runtime image
