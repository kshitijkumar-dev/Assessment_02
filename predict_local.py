import argparse
import json
import math
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent
VARIANTS = {
    "fever": {
        "name": "FEVER (2 class)",
        "train_data": "data/fever/fever_train-2class-mh.json",
        "graph_model": "pretrained_models/graph_rep_model_fever.pt",
        "output_dir": "outputs/causal_walk_fever_2way",
        "nclass": 2,
        "evidence_count": 5,
        "sequence_length": 128,
        "max_path_length": 4,
        "clusters": 10,
        "politihop_2way": False,
        "labels": ["SUPPORTS", "REFUTES"],
    },
    "politihop-2way": {
        "name": "PolitiHop (2 class)",
        "train_data": "data/politihop/original/politihop_train_adv-2class.json",
        "graph_model": "pretrained_models/graph_rep_model_politihop.pt",
        "output_dir": "outputs/causal_walk_politihop_2way",
        "nclass": 2,
        "evidence_count": 20,
        "sequence_length": 128,
        "max_path_length": 5,
        "clusters": 5,
        "politihop_2way": True,
        "labels": ["SUPPORTS", "REFUTES"],
    },
    "politihop-3way": {
        "name": "PolitiHop (3 class)",
        "train_data": "data/politihop/original/politihop_train_adv.json",
        "graph_model": "pretrained_models/graph_rep_model_politihop.pt",
        "output_dir": "outputs/causal_walk_politihop_3way",
        "nclass": 3,
        "evidence_count": 20,
        "sequence_length": 128,
        "max_path_length": 5,
        "clusters": 5,
        "politihop_2way": False,
        "labels": ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"],
    },
}

PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Causal Walk | Local prediction</title>
  <style>
    :root { color-scheme: light; font-family: system-ui, sans-serif; color: #172033; background: #f3f6fb; }
    body { margin: 0; }
    main { max-width: 760px; margin: 48px auto; padding: 32px; background: white; border-radius: 16px; box-shadow: 0 12px 40px #1a2b4814; }
    h1 { margin: 0 0 8px; font-size: 1.8rem; }
    p { line-height: 1.55; color: #536078; }
    label { display: block; margin: 20px 0 8px; font-weight: 650; }
    select, textarea, button { box-sizing: border-box; width: 100%; font: inherit; border-radius: 8px; }
    select, textarea { padding: 12px; border: 1px solid #cbd4e2; background: white; }
    textarea { min-height: 110px; resize: vertical; }
    #evidence { min-height: 170px; }
    button { margin-top: 20px; padding: 12px; border: 0; background: #2857c5; color: white; font-weight: 700; cursor: pointer; }
    button:disabled { opacity: .65; cursor: wait; }
    .notice { margin-top: 20px; padding: 14px; border-radius: 8px; background: #edf3ff; color: #263d6b; }
    .error { background: #fff0ed; color: #842b1d; }
    .result { margin-top: 16px; padding: 16px; border: 1px solid #dfe5ef; border-radius: 8px; }
    .score { display: flex; justify-content: space-between; gap: 12px; padding: 8px 0; }
    .bar { height: 8px; margin-top: 4px; background: #e8edf5; border-radius: 8px; overflow: hidden; }
    .bar span { display: block; height: 100%; background: #2857c5; }
    small { color: #66738a; }
  </style>
</head>
<body>
<main>
  <h1>Causal Walk</h1>
  <p>Fact verification using a trained Causal Walk checkpoint. Enter a claim and one evidence sentence per line.</p>
  <div id="status" class="notice">Checking local model filesâ€¦</div>
  <form id="form">
    <label for="variant">Model</label>
    <select id="variant">
      <option value="fever">FEVER (2 class)</option>
      <option value="politihop-2way">PolitiHop (2 class)</option>
      <option value="politihop-3way">PolitiHop (3 class)</option>
    </select>
    <label for="claim">Claim</label>
    <textarea id="claim" required placeholder="Enter a claim to verify"></textarea>
    <label for="evidence">Evidence</label>
    <textarea id="evidence" required placeholder="Paste evidence sentences, one per line"></textarea>
    <small id="limit">The selected model limits evidence to the dataset's configured maximum.</small>
    <button id="submit" type="submit">Predict</button>
  </form>
  <section id="result" aria-live="polite"></section>
</main>
<script>
const statusBox = document.querySelector("#status");
const form = document.querySelector("#form");
const button = document.querySelector("#submit");
const resultBox = document.querySelector("#result");
const variant = document.querySelector("#variant");
const limits = {fever: 5, "politihop-2way": 20, "politihop-3way": 20};
variant.addEventListener("change", () => {
  document.querySelector("#limit").textContent =
    `Up to ${limits[variant.value]} evidence sentences are used.`;
});
async function updateStatus() {
  const response = await fetch("/api/status");
  const data = await response.json();
  const selected = data.variants[variant.value];
  if (selected.ready) {
    statusBox.textContent = `${selected.name}: model files found.`;
    statusBox.className = "notice";
  } else {
    statusBox.textContent = `${selected.name} is not ready: ${selected.missing.join(", ")}. See README.md to download assets and train a checkpoint.`;
    statusBox.className = "notice error";
  }
}
variant.addEventListener("change", updateStatus);
updateStatus().catch(error => {
  statusBox.textContent = `Could not check model files: ${error.message}`;
  statusBox.className = "notice error";
});
form.addEventListener("submit", async event => {
  event.preventDefault();
  button.disabled = true;
  button.textContent = "Predictingâ€¦";
  resultBox.textContent = "";
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        variant: variant.value,
        claim: document.querySelector("#claim").value,
        evidence: document.querySelector("#evidence").value
      })
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Prediction failed.");
    const result = document.createElement("div");
    result.className = "result";
    const title = document.createElement("strong");
    title.textContent = `Prediction: ${data.prediction}`;
    result.append(title);
    for (const [label, score] of Object.entries(data.scores)) {
      const row = document.createElement("div");
      row.className = "score";
      const name = document.createElement("span");
      name.textContent = label;
      const value = document.createElement("span");
      value.textContent = `${(score * 100).toFixed(1)}%`;
      row.append(name, value);
      const bar = document.createElement("div");
      bar.className = "bar";
      const fill = document.createElement("span");
      fill.style.width = `${Math.max(0, Math.min(100, score * 100))}%`;
      bar.append(fill);
      result.append(row, bar);
    }
    resultBox.append(result);
  } catch (error) {
    const message = document.createElement("div");
    message.className = "notice error";
    message.textContent = error.message;
    resultBox.append(message);
  } finally {
    button.disabled = false;
    button.textContent = "Predict";
  }
});
</script>
</body>
</html>"""


class PredictionService:
    def __init__(self):
        self._loaded = {}
        self._lock = Lock()

    def status(self):
        variants = {}
        for key, config in VARIANTS.items():
            output_dir = ROOT / config["output_dir"] / "models"
            checkpoint = output_dir / "best.pt"
            required = [
                ROOT / "pretrained_models" / "BERT-Pair" / "vocab.txt",
                ROOT / "pretrained_models" / "BERT-Pair" / "config.json",
                ROOT / "pretrained_models" / "BERT-Pair" / "pytorch_model.bin",
                ROOT / config["graph_model"],
                ROOT / config["train_data"],
                checkpoint,
            ]
            missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
            variants[key] = {
                "name": config["name"],
                "ready": not missing,
                "missing": missing,
                "checkpoint_count": int(checkpoint.is_file()),
            }
        return {"variants": variants}

    def predict(self, variant, claim, evidence):
        if not isinstance(variant, str) or variant not in VARIANTS:
            raise ValueError("Choose one of the listed model variants.")
        if not isinstance(claim, str) or not claim.strip():
            raise ValueError("Enter a claim.")
        if not isinstance(evidence, str):
            raise ValueError("Evidence must be text with one sentence per line.")
        config = VARIANTS[variant]
        evidence_sentences = [line.strip() for line in evidence.splitlines() if line.strip()]
        if not evidence_sentences:
            raise ValueError("Enter at least one evidence sentence.")
        if len(evidence_sentences) > config["evidence_count"]:
            raise ValueError(
                f"This model accepts at most {config['evidence_count']} evidence sentences."
            )

        with self._lock:
            predictor = self._loaded.get(variant)
            if predictor is None:
                predictor = self._load(variant, config)
                self._loaded[variant] = predictor
        torch, model, centers, tokenizer, device = predictor

        from data_load_utils import InputExample, convert_examples_to_features

        examples = [
            InputExample(0, claim.strip(), None, 0, 0, True, False)
        ]
        examples.extend(
            InputExample(i + 1, sentence, claim.strip(), 0, 0, False, True)
            for i, sentence in enumerate(evidence_sentences)
        )
        for i in range(len(evidence_sentences), config["evidence_count"]):
            examples.append(
                InputExample(i + 1, "[PAD]", None, 0, 0, False, True)
            )

        features = convert_examples_to_features(
            examples, config["sequence_length"], tokenizer
        )
        input_ids = torch.tensor([feature.input_ids for feature in features], dtype=torch.long)
        input_mask = torch.tensor([feature.input_mask for feature in features], dtype=torch.long)
        segment_ids = torch.tensor([feature.input_type_ids for feature in features], dtype=torch.long)
        batch = [
            input_ids.view(1, 1 + config["evidence_count"], config["sequence_length"]).to(device),
            input_mask.view(1, 1 + config["evidence_count"], config["sequence_length"]).to(device),
            segment_ids.view(1, 1 + config["evidence_count"], config["sequence_length"]).to(device),
            torch.tensor([0], dtype=torch.long, device=device),
            torch.tensor(
                [[1] + [1] * len(evidence_sentences)
                 + [0] * (config["evidence_count"] - len(evidence_sentences))],
                dtype=torch.long,
                device=device,
            ),
            torch.tensor(
                [[1] + [1] * len(evidence_sentences)
                 + [0] * (config["evidence_count"] - len(evidence_sentences))],
                dtype=torch.long,
                device=device,
            ),
        ]
        with torch.no_grad():
            outputs = model(batch, centers, False)
            probabilities = outputs[0].detach().cpu().tolist()
        if len(probabilities) != len(config["labels"]) or not all(
            math.isfinite(value) for value in probabilities
        ):
            raise RuntimeError("The model returned invalid class scores.")
        prediction = config["labels"][max(range(len(probabilities)), key=probabilities.__getitem__)]
        scores = {
            label: float(score)
            for label, score in zip(config["labels"], probabilities)
        }
        return {"prediction": prediction, "scores": scores}

    def _load(self, variant, config):
        missing = self.status()["variants"][variant]["missing"]
        if missing:
            raise RuntimeError(
                "Required files are missing: " + ", ".join(missing)
            )

        import torch
        from torch.utils.data import DataLoader
        from tqdm import tqdm
        from data_load_utils import build_dataset
        from models import Walk_with_bert
        from pytorch_pretrained_bert.tokenization import BertTokenizer
        from utils import cal_nwgm_center_attention

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint_dir = ROOT / config["output_dir"] / "models"
        checkpoint_path = checkpoint_dir / "best.pt"
        train_data = build_dataset(
            str(ROOT / config["train_data"]),
            config["evidence_count"],
            config["sequence_length"],
        )
        train_loader = tqdm(DataLoader(train_data, batch_size=4, shuffle=False))
        centers = cal_nwgm_center_attention(
            str(ROOT / config["graph_model"]),
            train_loader,
            config["nclass"],
            device,
            768,
            config["evidence_count"],
            config["clusters"],
            politihop_2way=config["politihop_2way"],
        )
        model = Walk_with_bert(
            768,
            config["nclass"],
            config["max_path_length"],
            3,
            config["evidence_count"],
            "nwgm_attention",
        ).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        tokenizer = BertTokenizer.from_pretrained(
            str(ROOT / "pretrained_models" / "BERT-Pair"),
            do_lower_case=True,
        )
        print(f"Loaded {variant} checkpoint: {checkpoint_path.relative_to(ROOT)}", flush=True)
        return torch, model, centers, tokenizer, device


SERVICE = PredictionService()


class RequestHandler(BaseHTTPRequestHandler):
    def _send_json(self, status_code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/api/status":
            self._send_json(200, SERVICE.status())
        else:
            self._send_json(404, {"error": "Not found."})

    def do_POST(self):
        if urlparse(self.path).path != "/api/predict":
            self._send_json(404, {"error": "Not found."})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > 1_000_000:
                raise ValueError("Request body must be between 1 byte and 1 MB.")
            payload = json.loads(self.rfile.read(content_length))
            if not isinstance(payload, dict):
                raise ValueError("Request body must be a JSON object.")
            prediction = SERVICE.predict(
                payload.get("variant"),
                payload.get("claim"),
                payload.get("evidence"),
            )
        except ValueError as error:
            self._send_json(400, {"error": str(error)})
            return
        except RuntimeError as error:
            self._send_json(503, {"error": str(error)})
            return
        except ImportError as error:
            self._send_json(
                503,
                {"error": f"A required Python package is missing: {error.name}. Install requirements.txt."},
            )
            return
        self._send_json(200, prediction)

    def log_message(self, format_string, *args):
        print(f"{self.address_string()} - {format_string % args}")


def main():
    parser = argparse.ArgumentParser(description="Run the Causal Walk prediction UI locally.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: localhost only).")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port.")
    args = parser.parse_args()
    server = HTTPServer((args.host, args.port), RequestHandler)
    print(f"Causal Walk is available at http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Causal Walk server.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
