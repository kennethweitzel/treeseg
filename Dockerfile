FROM python:3.13

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libexpat1 \
    && rm -rf /var/lib/apt/lists/*

RUN python3.13 -m venv /app/venv 

ENV PATH="/app/venv/bin:$PATH"

RUN pip install --upgrade pip \
    && pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir rasterio torchgeo

COPY . /app

CMD ["python3", "-m", "treeseg", "input.tif", "--output", "output/tree_mask.geojson", "--device", "cpu"]