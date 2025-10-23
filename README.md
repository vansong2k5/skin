# Skin Insight Platform

Skin Insight is an open-source dermatology triage assistant. The platform fuses
structured form data, a local computer-vision pipeline (EfficientNet +
segmentation heuristics/SAM) and Gemini's multimodal LLM to produce early-stage
recommendations for patients describing skin conditions.

## Key features

- **Multimodal inference pipeline** – Uploaded photos are normalised, analysed
  with an EfficientNet-B0 backbone and segmented (Segment-Anything when
  available, otherwise an OpenCV fallback). The extractor produces
  interpretable features such as colour pattern, redness and border
  irregularity.
- **Gemini integration** – The structured image descriptors and the patient's
  textual description are sent to Gemini in a schema-enforced JSON prompt that
  returns triage advice, severity level and next steps in Vietnamese.
- **Modular Flask backend** – Simple REST API with health checks, debug ping and
  an `/analyze` endpoint that orchestrates image encoding, feature extraction
  and LLM interaction.
- **Frontend ready** – A static frontend (in `/frontend`) can be served directly
  by Flask for rapid prototyping.
- **Open-source friendly** – MIT licensed with clear contribution, conduct and
  security guidelines included in this repository.

## Repository layout

```
backend/
  api/                 # Flask blueprints and request handlers
  config/              # Environment loading and secrets management
  models/              # Gemini client + vision feature extractor
  utils/               # Shared helpers (image encoding, etc.)
frontend/              # Static assets for the demo UI
```

## Getting started

### 1. Environment variables

Create `backend/config/.env` with your Gemini API key:

```
GEMINI_API_KEY=your_google_api_key
```

Optional variables that control the vision stack:

```
# Optional: pointer to a Segment Anything checkpoint to enable high-quality masks
SAM_CHECKPOINT_PATH=/path/to/sam_vit_b.pth
SAM_MODEL_TYPE=vit_b  # vit_b / vit_l / vit_h, defaults to vit_b
GEMINI_MODEL=gemini-2.5-flash  # override if you need a different Gemini release
```

### 2. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Optional: Segment Anything support
# pip install git+https://github.com/facebookresearch/segment-anything.git
```

> The first time EfficientNet runs it may download pretrained weights from the
> `torchvision` model hub. Ensure the machine has network access or manually
> provide the weights if running offline.

### 3. Run the development server

```bash
export FLASK_APP=backend.main:app
python -m backend.main
```

The backend listens on http://localhost:5001 by default. Navigate to `/` to load
the bundled frontend or use the API directly.

### 4. Call the API

```bash
curl -X POST http://localhost:5001/analyze \
  -F "description=Da bị ngứa và rát 3 ngày nay" \
  -F "image=@example.jpg"
```

The response is a JSON payload containing the model's assessment, severity,
red-flag warnings (if any) and metadata about the request (image size, extracted
features, Gemini usage stats).

## Image feature pipeline

1. **Pre-processing** – Images are resized to 1280px width, converted to JPEG and
   base64 encoded for Gemini.
2. **Feature extraction** – `ImageFeatureExtractor` normalises the image,
   segments the lesion (SAM if configured; otherwise an Otsu-based fallback) and
   computes:
   - `color_pattern`: dominant hue, hue variance, saturation/value averages and
     a short textual summary.
   - `redness`: mean normalised red channel and a redness index (red dominance
     over green/blue).
   - `texture`: Laplacian variance describing surface roughness.
   - `border`: perimeter, area and irregularity (perimeter² / 4π area) with an
     easy-to-read descriptor.
   - `area_ratio`: lesion area vs. total image area.
   - `embedding_sample`: the first 16 components of the EfficientNet feature
     vector plus its L2 norm for downstream ML usage.
3. **LLM prompting** – The extracted features are serialised to JSON and
   appended to the Gemini prompt so the language model can align clinical
   reasoning with the visual evidence.

If feature extraction fails for any reason the API continues gracefully and
includes an error note in the response metadata.

## Development guidelines

- Run `python -m compileall backend` before submitting a PR to catch syntax
  errors.
- Code follows standard Python type hints. Optional heavy dependencies (SAM,
  CUDA) are guarded to keep the project usable on commodity hardware.
- The `/__routes` endpoint is available for quick route discovery during manual
  testing.

## Contributing

We welcome contributions! Please read `CONTRIBUTING.md` for coding standards,
branching strategy and review expectations. We follow the `Contributor Covenant`
(`CODE_OF_CONDUCT.md`) and publish security reporting steps in `SECURITY.md`.

## License

This project is licensed under the MIT License (`LICENSE`). You are free to use
it in commercial and non-commercial settings as long as the copyright notice is
retained.
