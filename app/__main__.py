import base64
import io
from functools import lru_cache

import numpy as np
from flask import Flask, jsonify, render_template_string, request
from PIL import Image

app = Flask(__name__)

HTML = """
<!doctype html>
<html lang="th">
	<head>
		<meta charset="utf-8" />
		<meta name="viewport" content="width=device-width, initial-scale=1" />
		<title>Cat Food Detector</title>
		<link rel="preconnect" href="https://fonts.googleapis.com">
		<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
		<link href="https://fonts.googleapis.com/css2?family=Mali:wght@400;500;600;700&display=swap" rel="stylesheet">
		<script src="https://cdn.tailwindcss.com"></script>
	</head>
	<body class="min-h-screen bg-gradient-to-br from-rose-50 via-pink-50 to-violet-100 px-3 py-4 text-slate-700 sm:p-8" style="font-family: 'Mali', cursive;">
		<div class="mx-auto w-full max-w-3xl rounded-2xl border border-rose-100 bg-white/90 p-4 shadow-xl shadow-rose-100/60 backdrop-blur sm:rounded-3xl sm:p-8">
			<div class="mb-5 border-b border-rose-100 pb-4">
				<h2 class="text-2xl font-bold text-rose-700 sm:text-3xl">เว็บตรวจจับอาหารแมว</h2>
				<p class="mt-2 text-sm text-slate-600 sm:text-base">เปิดกล้องเพื่อวิเคราะห์ภาพแบบเรียลไทม์ ระบบจะประเมินจากสีและพื้นผิวของภาพแต่ละเฟรม</p>
			</div>

			<div class="rounded-2xl border border-rose-100 bg-rose-50/60 p-2 sm:p-3">
				<div class="relative">
					<video id="camera" autoplay playsinline muted class="aspect-video w-full rounded-lg border border-rose-200 bg-slate-900 sm:rounded-xl"></video>
					<canvas id="overlay" class="pointer-events-none absolute inset-0 h-full w-full rounded-lg sm:rounded-xl"></canvas>
				</div>
				<div id="uploadPreview" class="relative mt-2 hidden">
					<img id="previewImg" class="w-full rounded-lg sm:rounded-xl" alt="preview" />
					<canvas id="previewOverlay" class="pointer-events-none absolute inset-0 h-full w-full rounded-lg sm:rounded-xl"></canvas>
				</div>
			</div>

			<div class="mt-4 grid grid-cols-1 gap-2 sm:grid-cols-3">
				<button id="startBtn" type="button" class="w-full rounded-xl bg-rose-500 px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-rose-600 focus:outline-none focus:ring-2 focus:ring-rose-300 disabled:cursor-not-allowed">เริ่มกล้อง</button>
				<button id="stopBtn" type="button" disabled class="w-full rounded-xl bg-slate-300 px-4 py-2.5 text-sm font-semibold text-slate-600 transition disabled:cursor-not-allowed">หยุด</button>
				<input id="uploadInput" type="file" accept="image/*" class="hidden" />
				<button id="uploadBtn" type="button" class="w-full rounded-xl bg-amber-500 px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-amber-600 focus:outline-none focus:ring-2 focus:ring-amber-300">ทดสอบจากรูป</button>
			</div>

			<div class="mt-5 space-y-3 rounded-2xl border border-violet-100 bg-violet-50/60 p-4">
				<p id="resultText" class="text-sm font-semibold text-violet-700 sm:text-base">ผลลัพธ์: ยังไม่ได้เริ่ม</p>
				<div>
					<div class="mb-1 flex items-center justify-between text-xs text-slate-500">
						<span>ระดับอาหารที่เหลือในถ้วย</span>
						<span id="fillPctLabel" class="font-semibold">-</span>
					</div>
					<div class="h-4 w-full overflow-hidden rounded-full bg-slate-200">
						<div id="fillBar" class="h-4 rounded-full bg-emerald-400 transition-all duration-500" style="width: 0%"></div>
					</div>
				</div>
				<p id="confidenceText" class="text-sm text-slate-700">ความมั่นใจ: -</p>
				<p id="reasonText" class="break-words text-xs text-slate-500 sm:text-sm">-</p>
			</div>
		</div>

		<script>
			const video = document.getElementById("camera");
			const startBtn = document.getElementById("startBtn");
			const stopBtn = document.getElementById("stopBtn");
			const uploadBtn = document.getElementById("uploadBtn");
			const uploadInput = document.getElementById("uploadInput");
			const resultText = document.getElementById("resultText");
			const fillPctLabel = document.getElementById("fillPctLabel");
			const fillBar = document.getElementById("fillBar");
			const confidenceText = document.getElementById("confidenceText");
			const reasonText = document.getElementById("reasonText");

			const overlay = document.getElementById("overlay");
			const overlayCtx = overlay.getContext("2d");
			const uploadPreview = document.getElementById("uploadPreview");
			const previewImg = document.getElementById("previewImg");
			const previewOverlay = document.getElementById("previewOverlay");
			const previewCtx = previewOverlay.getContext("2d");
			const captureCanvas = document.createElement("canvas");
			let stream = null;
			let timerId = null;

			function drawBox(ctx, canvasEl, box, label, color) {
				const dw = canvasEl.offsetWidth;
				const dh = canvasEl.offsetHeight;
				canvasEl.width = dw;
				canvasEl.height = dh;
				ctx.clearRect(0, 0, dw, dh);
				if (!box) return;
				const x = box.x1 * dw;
				const y = box.y1 * dh;
				const w = (box.x2 - box.x1) * dw;
				const h = (box.y2 - box.y1) * dh;
				ctx.strokeStyle = color;
				ctx.lineWidth = 3;
				ctx.lineJoin = "round";
				ctx.strokeRect(x, y, w, h);
				const fontSize = Math.max(12, Math.round(dh * 0.045));
				ctx.font = `bold ${fontSize}px Mali, sans-serif`;
				const textW = ctx.measureText(label).width;
				const padX = 6, padY = 4;
				const labelH = fontSize + padY * 2;
				const labelY = y > labelH + 2 ? y - labelH : y + h + 2;
				ctx.fillStyle = color;
				ctx.beginPath();
				ctx.roundRect(x, labelY, textW + padX * 2, labelH, 4);
				ctx.fill();
				ctx.fillStyle = "#fff";
				ctx.fillText(label, x + padX, labelY + fontSize + padY - 2);
			}

			const statusBaseClass = "text-sm font-semibold sm:text-base";
			const statusOkClass = "text-emerald-600";
			const statusNoClass = "text-rose-600";
			const statusIdleClass = "text-violet-700";
			const startBtnEnabledClass = "w-full rounded-xl bg-rose-500 px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-rose-600 focus:outline-none focus:ring-2 focus:ring-rose-300 disabled:cursor-not-allowed";
			const startBtnDisabledClass = "w-full rounded-xl bg-rose-300 px-4 py-2.5 text-sm font-semibold text-white transition disabled:cursor-not-allowed";
			const stopBtnEnabledClass = "w-full rounded-xl bg-violet-500 px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-violet-600";
			const stopBtnDisabledClass = "w-full rounded-xl bg-slate-300 px-4 py-2.5 text-sm font-semibold text-slate-600 transition disabled:cursor-not-allowed";

			function setStatus(kind, message) {
				resultText.className = statusBaseClass;
				if (kind === "ok") {
					resultText.classList.add(statusOkClass);
				} else if (kind === "no") {
					resultText.classList.add(statusNoClass);
				} else {
					resultText.classList.add(statusIdleClass);
				}
				resultText.textContent = message;
			}

			function setButtonsRunning(isRunning) {
				startBtn.disabled = isRunning;
				stopBtn.disabled = !isRunning;
				startBtn.className = isRunning ? startBtnDisabledClass : startBtnEnabledClass;
				stopBtn.className = isRunning ? stopBtnEnabledClass : stopBtnDisabledClass;
			}

			function renderResult(result, isUpload) {
				const pct = result.fill_percent ?? 0;
				const status = result.status ?? (result.detected ? "ok" : "empty");
				const box = result.bowl_box ?? null;
				const boxColor = status === "empty" ? "#f43f5e" : status === "low" ? "#f59e0b" : "#10b981";
				const boxLabel = box ? (status === "empty" ? "ถาดอาหาร (หมดแล้ว)" : `ถาดอาหาร ${pct}%`) : "";
				if (isUpload) {
					drawBox(previewCtx, previewOverlay, box, boxLabel, boxColor);
				} else {
					drawBox(overlayCtx, overlay, box, boxLabel, boxColor);
				}

				let statusKind, statusMsg;
				if (status === "empty") {
					statusKind = "no";
					statusMsg = "อาหารหมดแล้ว! กรุณาเติมอาหารให้น้องแมว";
				} else if (status === "low") {
					statusKind = "idle";
					statusMsg = `อาหารเกือบหมด เหลือประมาณ ${pct}%`;
				} else {
					statusKind = "ok";
					statusMsg = `มีอาหารเหลืออยู่ ${pct}%`;
				}
				setStatus(statusKind, `ผลลัพธ์: ${statusMsg}`);

				fillPctLabel.textContent = `${pct}%`;
				fillBar.style.width = `${pct}%`;
				if (status === "empty") {
					fillBar.className = "h-4 rounded-full bg-rose-400 transition-all duration-500";
				} else if (status === "low") {
					fillBar.className = "h-4 rounded-full bg-amber-400 transition-all duration-500";
				} else {
					fillBar.className = "h-4 rounded-full bg-emerald-400 transition-all duration-500";
				}

				confidenceText.textContent = `ความมั่นใจ: ${result.confidence}%`;
				reasonText.textContent = result.reason;
			}

			async function analyzeImageData(imageData, isUpload = false) {
				try {
					const response = await fetch("/analyze_frame", {
						method: "POST",
						headers: { "Content-Type": "application/json" },
						body: JSON.stringify({ image: imageData }),
					});

					if (!response.ok) {
						throw new Error("analyze failed");
					}

					const result = await response.json();
					renderResult(result, isUpload);
				} catch (_) {
					setStatus("no", "ผลลัพธ์: วิเคราะห์ไม่สำเร็จ");
					confidenceText.textContent = "ความมั่นใจ: -";
					reasonText.textContent = "โปรดลองใหม่อีกครั้ง";
				}
			}

			async function analyzeFrame() {
				if (!stream || video.videoWidth === 0 || video.videoHeight === 0) {
					return;
				}

				captureCanvas.width = video.videoWidth;
				captureCanvas.height = video.videoHeight;
				const ctx = captureCanvas.getContext("2d");
				ctx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);

				const imageData = captureCanvas.toDataURL("image/jpeg", 0.7);
				await analyzeImageData(imageData, false);
			}

			startBtn.addEventListener("click", async () => {
				if (stream) {
					return;
				}

				try {
					stream = await navigator.mediaDevices.getUserMedia({
						video: { facingMode: "environment" },
						audio: false,
					});
					video.srcObject = stream;
					setButtonsRunning(true);

					timerId = window.setInterval(analyzeFrame, 500);
					analyzeFrame();
				} catch (_) {
					setStatus("no", "ผลลัพธ์: ไม่สามารถเปิดกล้องได้");
					reasonText.textContent = "กรุณาอนุญาตการใช้งานกล้องในเบราว์เซอร์";
				}
			});

			stopBtn.addEventListener("click", () => {
				if (timerId) {
					window.clearInterval(timerId);
					timerId = null;
				}
				if (stream) {
					stream.getTracks().forEach((track) => track.stop());
					stream = null;
					video.srcObject = null;
				}
				overlay.width = overlay.offsetWidth;
				overlay.height = overlay.offsetHeight;
				overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
				setButtonsRunning(false);
				setStatus("idle", "ผลลัพธ์: หยุดการตรวจจับ");
				fillPctLabel.textContent = "-";
				fillBar.style.width = "0%";
				fillBar.className = "h-4 rounded-full bg-emerald-400 transition-all duration-500";
				confidenceText.textContent = "ความมั่นใจ: -";
				reasonText.textContent = "-";
			});

			uploadBtn.addEventListener("click", () => {
				uploadInput.click();
			});

			uploadInput.addEventListener("change", async (event) => {
				const [file] = event.target.files || [];
				if (!file) {
					return;
				}

				const reader = new FileReader();
				reader.onload = async () => {
					const dataUrl = typeof reader.result === "string" ? reader.result : "";
					if (!dataUrl) {
						setStatus("no", "ผลลัพธ์: อ่านรูปไม่สำเร็จ");
						reasonText.textContent = "กรุณาลองเลือกรูปใหม่";
						return;
					}
					previewImg.src = dataUrl;
					uploadPreview.classList.remove("hidden");
					setStatus("idle", "ผลลัพธ์: กำลังวิเคราะห์จากรูป");
					await analyzeImageData(dataUrl, true);
				};
				reader.onerror = () => {
					setStatus("no", "ผลลัพธ์: อ่านรูปไม่สำเร็จ");
					reasonText.textContent = "ไฟล์รูปอาจเสียหาย";
				};
				reader.readAsDataURL(file);
				event.target.value = "";
			});
		</script>
	</body>
</html>
"""


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
		rgb = rgb.astype(np.float32) / 255.0
		r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
		maxc = np.max(rgb, axis=-1)
		minc = np.min(rgb, axis=-1)
		delta = maxc - minc

		hue = np.zeros_like(maxc)
		mask = delta != 0

		r_mask = (maxc == r) & mask
		g_mask = (maxc == g) & mask
		b_mask = (maxc == b) & mask

		hue[r_mask] = ((g[r_mask] - b[r_mask]) / delta[r_mask]) % 6
		hue[g_mask] = ((b[g_mask] - r[g_mask]) / delta[g_mask]) + 2
		hue[b_mask] = ((r[b_mask] - g[b_mask]) / delta[b_mask]) + 4
		hue = hue / 6.0

		sat = np.zeros_like(maxc)
		nonzero = maxc != 0
		sat[nonzero] = delta[nonzero] / maxc[nonzero]

		val = maxc
		return np.stack([hue, sat, val], axis=-1)


@lru_cache(maxsize=1)
def get_yolo_model():
		try:
				from ultralytics import YOLO
		except ImportError as exc:
				return None, f"ultralytics import failed: {exc}"

		try:
				return YOLO("yolov8n.pt"), None
		except Exception as exc:
				return None, f"yolo model load failed: {exc}"


def detect_bowl_mask(arr: np.ndarray) -> tuple[np.ndarray, str, dict | None]:
		height, width = arr.shape[:2]
		model, load_error = get_yolo_model()
		if model is None:
				return np.zeros((height, width), dtype=bool), load_error or "yolo unavailable", None

		try:
				result = model.predict(arr, imgsz=320, conf=0.25, verbose=False)[0]
		except Exception as exc:
				return np.zeros((height, width), dtype=bool), f"yolo inference failed: {exc}", None

		if result.boxes is None or len(result.boxes) == 0:
				return np.zeros((height, width), dtype=bool), "yolo bowl not found", None

		classes = result.boxes.cls.detach().cpu().numpy().astype(int)
		confidences = result.boxes.conf.detach().cpu().numpy()
		xyxy = result.boxes.xyxy.detach().cpu().numpy()

		priority_classes = [45, 41, 39]
		selected_indices = np.array([], dtype=int)
		for class_id in priority_classes:
				indices = np.where(classes == class_id)[0]
				if indices.size > 0:
						selected_indices = indices
						break

		reason = "yolo container-like class found"
		if selected_indices.size == 0:
				img_area = float(height * width)
				box_areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
				valid = np.where(
						(confidences >= 0.35)
						& (box_areas >= 0.02 * img_area)
						& (box_areas <= 0.90 * img_area)
				)[0]
				if valid.size == 0:
						return np.zeros((height, width), dtype=bool), "yolo no suitable container box", None
				selected_indices = valid
				reason = "yolo fallback generic box"

		best_idx = selected_indices[np.argmax(confidences[selected_indices])]
		x1, y1, x2, y2 = xyxy[best_idx]
		pad_x = int((x2 - x1) * 0.10)
		pad_y = int((y2 - y1) * 0.10)
		left = max(0, int(x1) - pad_x)
		top = max(0, int(y1) - pad_y)
		right = min(width, int(x2) + pad_x)
		bottom = min(height, int(y2) + pad_y)

		mask = np.zeros((height, width), dtype=bool)
		mask[top:bottom, left:right] = True
		box_norm = {
				"x1": left / width,
				"y1": top / height,
				"x2": right / width,
				"y2": bottom / height,
		}
		return mask, reason, box_norm


def detect_cat_food(image: Image.Image) -> dict:
		image = image.convert("RGB").resize((320, 320))
		arr = np.array(image)

		hsv = rgb_to_hsv(arr)
		h = hsv[..., 0] * 360.0
		s = hsv[..., 1]
		v = hsv[..., 2]

		height, width = h.shape
		yy, xx = np.indices((height, width))
		nx = (xx - (width / 2.0)) / (width / 2.0)
		ny = (yy - (height / 2.0)) / (height / 2.0)
		center_ellipse = (nx * nx + ny * ny) <= 0.92

		gray = np.mean(arr.astype(np.float32), axis=2)
		dx = np.abs(np.diff(gray, axis=1))
		dy = np.abs(np.diff(gray, axis=0))
		texture = float((dx.mean() + dy.mean()) / 2.0)
		gx, gy = np.gradient(gray)
		grad_mag = np.hypot(gx, gy)

		brightness = float(v.mean())

		yolo_mask, yolo_reason, bowl_box = detect_bowl_mask(arr)
		container_mask = yolo_mask
		container_pixels = int(container_mask.sum())
		if container_pixels == 0:
				plate_like = (s <= 0.55) & (v >= np.percentile(v, 35))
				container_mask = center_ellipse & plate_like
				container_pixels = int(container_mask.sum())
				min_container_pixels = int(height * width * 0.08)
				if container_pixels < min_container_pixels:
						container_mask = center_ellipse
						container_pixels = int(container_mask.sum())
				container_source = "fallback-center"
		else:
				container_source = "yolo"

		grad_threshold = float(np.percentile(grad_mag[container_mask], 72))
		texture_mask = grad_mag >= grad_threshold
		food_warm = (h >= 8) & (h <= 75) & (s >= 0.08) & (v >= 0.08) & (v <= 0.95)
		food_dark = (v >= 0.10) & (v <= 0.78) & (s >= 0.04)
		texture_food = texture_mask & (v >= 0.07)
		food_mask = (food_warm | food_dark | texture_food) & container_mask
		food_ratio = float(np.sum(food_mask) / max(container_pixels, 1))

		raw_fill_ratio = float(np.sum(food_mask & container_mask) / max(container_pixels, 1))
		fill_ratio = min(raw_fill_ratio / 0.60, 1.0)
		score = 0.60 * min(food_ratio / 0.06, 1.0) + 0.40 * min(raw_fill_ratio / 0.12, 1.0)

		confidence = int(max(0.0, min(score, 1.0)) * 100)
		fill_percent = int(max(0.0, min(fill_ratio, 1.0)) * 100)

		if fill_percent <= 5:
				status = "empty"
		elif fill_percent <= 25:
				status = "low"
		else:
				status = "ok"
		detected = fill_percent > 5

		return {
				"detected": detected,
				"status": status,
				"confidence": confidence,
				"fill_percent": fill_percent,
				"bowl_box": bowl_box,
				"reason": (
						f"food_ratio={food_ratio:.2f}, "
						f"fill_raw={raw_fill_ratio:.2f}, texture={texture:.2f}, "
						f"brightness={brightness:.2f}, container={container_source}, yolo={yolo_reason}"
				),
		}


def decode_data_url_to_image(data_url: str) -> Image.Image:
		if "," in data_url:
				_, encoded = data_url.split(",", 1)
		else:
				encoded = data_url
		binary = base64.b64decode(encoded)
		return Image.open(io.BytesIO(binary))


@app.route("/", methods=["GET"])
def index():
		return render_template_string(HTML)


@app.route("/analyze_frame", methods=["POST"])
def analyze_frame():
		payload = request.get_json(silent=True) or {}
		image_data = payload.get("image")

		if not image_data:
				return jsonify({"error": "missing image"}), 400

		try:
				image = decode_data_url_to_image(image_data)
		except Exception:
				return jsonify({"error": "invalid image"}), 400

		return jsonify(detect_cat_food(image))


if __name__ == "__main__":
		app.run(host="127.0.0.1", port=3000, debug=True)
