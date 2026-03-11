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
				<h2 class="text-2xl font-bold text-rose-700 sm:text-3xl">Meow Meow Food Detection</h2>
				<p class="mt-2 text-sm text-slate-600 sm:text-base">เปิดกล้องเพื่อวิเคราะห์ภาพแบบเรียลไทม์ ระบบจะประเมินจากลักษณะเม็ดอาหารและพื้นผิวของภาพแต่ละเฟรม</p>
			</div>

			<div class="rounded-2xl border border-rose-100 bg-rose-50/60 p-2 sm:p-3">
				<div class="relative aspect-video overflow-hidden rounded-lg border border-rose-200 bg-slate-900 sm:rounded-xl">
					<video id="camera" autoplay playsinline muted class="absolute inset-0 h-full w-full object-cover"></video>
					<canvas id="overlay" class="pointer-events-none absolute inset-0 h-full w-full"></canvas>
				</div>
				<div id="uploadPreview" class="relative mt-2 hidden aspect-video overflow-hidden rounded-lg border border-rose-200 bg-slate-900 sm:rounded-xl">
					<img id="previewImg" class="absolute inset-0 h-full w-full object-cover" alt="preview" />
					<canvas id="previewOverlay" class="pointer-events-none absolute inset-0 h-full w-full"></canvas>
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
				<div>
					<div class="mb-1 flex items-center justify-between text-xs text-slate-500">
						<span>ความมั่นใจ</span>
						<span id="confidencePctLabel" class="font-semibold">-</span>
					</div>
					<div class="h-4 w-full overflow-hidden rounded-full bg-slate-200">
						<div id="confidenceBar" class="h-4 rounded-full bg-violet-400 transition-all duration-500" style="width: 0%"></div>
					</div>
				</div>
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
			const confidencePctLabel = document.getElementById("confidencePctLabel");
			const confidenceBar = document.getElementById("confidenceBar");
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

			function drawFoodMarker(ctx, canvasEl, box, detected, color) {
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
				const cx = x + w / 2;
				const cy = y + h / 2;
				const radius = Math.max(8, Math.min(w, h) / 2);
				ctx.strokeStyle = color;
				ctx.lineWidth = 3;
				ctx.lineJoin = "round";
				ctx.beginPath();
				ctx.arc(cx, cy, radius, 0, Math.PI * 2);
				ctx.stroke();

				const fontSize = Math.max(12, Math.round(dh * 0.045));
				const label = detected ? "พบอาหารแมว" : "ไม่พบอาหารแมว";
				ctx.font = `bold ${fontSize}px Mali, sans-serif`;
				const textW = ctx.measureText(label).width;
				const padX = 6, padY = 4;
				const labelH = fontSize + padY * 2;
				const labelX = 8;
				const labelY = 8;
				ctx.fillStyle = color;
				ctx.beginPath();
				ctx.roundRect(labelX, labelY, textW + padX * 2, labelH, 4);
				ctx.fill();
				ctx.fillStyle = "#fff";
				ctx.fillText(label, labelX + padX, labelY + fontSize + padY - 2);
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
				const bowlBox = result.bowl_box ?? null;
				const boxColor = status === "empty" ? "#f43f5e" : status === "low" ? "#f59e0b" : "#10b981";
				if (isUpload) {
					drawFoodMarker(previewCtx, previewOverlay, bowlBox, Boolean(result.detected), boxColor);
				} else {
					drawFoodMarker(overlayCtx, overlay, bowlBox, Boolean(result.detected), boxColor);
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

				const confidence = result.confidence ?? 0;
				confidencePctLabel.textContent = `${confidence}%`;
				confidenceBar.style.width = `${confidence}%`;
				if (confidence < 40) {
					confidenceBar.className = "h-4 rounded-full bg-rose-400 transition-all duration-500";
				} else if (confidence < 70) {
					confidenceBar.className = "h-4 rounded-full bg-amber-400 transition-all duration-500";
				} else {
					confidenceBar.className = "h-4 rounded-full bg-emerald-400 transition-all duration-500";
				}
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
					confidencePctLabel.textContent = "-";
					confidenceBar.style.width = "0%";
					confidenceBar.className = "h-4 rounded-full bg-violet-400 transition-all duration-500";
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
				confidencePctLabel.textContent = "-";
				confidenceBar.style.width = "0%";
				confidenceBar.className = "h-4 rounded-full bg-violet-400 transition-all duration-500";
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
				result = model.predict(arr, imgsz=320, conf=0.20, verbose=False)[0]
		except Exception as exc:
				return np.zeros((height, width), dtype=bool), f"yolo inference failed: {exc}", None

		if result.boxes is None or len(result.boxes) == 0:
				return np.zeros((height, width), dtype=bool), "yolo bowl not found", None

		classes = result.boxes.cls.detach().cpu().numpy().astype(int)
		confidences = result.boxes.conf.detach().cpu().numpy()
		xyxy = result.boxes.xyxy.detach().cpu().numpy()
		img_area = float(height * width)
		box_areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
		candidates: list[tuple[float, int, str]] = []
		for idx in range(len(classes)):
				class_id = int(classes[idx])
				conf = float(confidences[idx])
				area_ratio = float(box_areas[idx] / max(img_area, 1.0))
				if area_ratio < 0.01 or area_ratio > 0.75:
						continue
				if class_id not in {45, 41, 39}:
						continue
				if class_id == 39 and conf < 0.45:
						continue
				if class_id in {45, 41} and conf < 0.24:
						continue

				x1, y1, x2, y2 = xyxy[idx]
				cx = ((x1 + x2) * 0.5) / max(width, 1.0)
				cy = ((y1 + y2) * 0.5) / max(height, 1.0)
				center_dist = float(np.hypot(cx - 0.5, cy - 0.5))
				center_score = max(0.0, 1.0 - (center_dist / 0.72))
				area_score = max(0.0, 1.0 - abs(area_ratio - 0.18) / 0.22)
				class_bias = 1.00 if class_id == 45 else (0.92 if class_id == 41 else 0.70)
				score = (0.56 * conf) + (0.24 * center_score) + (0.20 * area_score)
				score *= class_bias
				candidates.append((score, idx, f"class={class_id}"))

		if not candidates:
				return np.zeros((height, width), dtype=bool), "yolo no container candidate", None

		candidates.sort(key=lambda x: x[0], reverse=True)
		_, best_idx, best_note = candidates[0]
		x1, y1, x2, y2 = xyxy[best_idx]
		pad_x = int((x2 - x1) * 0.08)
		pad_y = int((y2 - y1) * 0.08)
		left = max(0, int(x1) - pad_x)
		top = max(0, int(y1) - pad_y)
		right = min(width, int(x2) + pad_x)
		bottom = min(height, int(y2) + pad_y)

		mask = np.zeros((height, width), dtype=bool)
		if right <= left or bottom <= top:
				return mask, "yolo invalid container box", None
		yy, xx = np.indices((height, width))
		cx = (left + right) * 0.5
		cy = (top + bottom) * 0.5
		rx = max((right - left) * 0.5, 1.0)
		ry = max((bottom - top) * 0.5, 1.0)
		ellipse = (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) <= 1.0
		mask = ellipse
		box_norm = {
				"x1": left / width,
				"y1": top / height,
				"x2": right / width,
				"y2": bottom / height,
				"confidence": float(confidences[best_idx]),
				"class_id": int(classes[best_idx]),
		}
		return mask, f"yolo container selected ({best_note})", box_norm


def _texture_metrics(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
		gx, gy = np.gradient(gray)
		grad_mag = np.hypot(gx, gy)
		lap = np.abs(
				(4.0 * gray)
				- np.roll(gray, 1, axis=0)
				- np.roll(gray, -1, axis=0)
				- np.roll(gray, 1, axis=1)
				- np.roll(gray, -1, axis=1)
		)

		gray_sq = gray * gray
		local_mean = (
				gray
				+ np.roll(gray, 1, axis=0)
				+ np.roll(gray, -1, axis=0)
				+ np.roll(gray, 1, axis=1)
				+ np.roll(gray, -1, axis=1)
				+ np.roll(np.roll(gray, 1, axis=0), 1, axis=1)
				+ np.roll(np.roll(gray, 1, axis=0), -1, axis=1)
				+ np.roll(np.roll(gray, -1, axis=0), 1, axis=1)
				+ np.roll(np.roll(gray, -1, axis=0), -1, axis=1)
		) / 9.0
		local_sq_mean = (
				gray_sq
				+ np.roll(gray_sq, 1, axis=0)
				+ np.roll(gray_sq, -1, axis=0)
				+ np.roll(gray_sq, 1, axis=1)
				+ np.roll(gray_sq, -1, axis=1)
				+ np.roll(np.roll(gray_sq, 1, axis=0), 1, axis=1)
				+ np.roll(np.roll(gray_sq, 1, axis=0), -1, axis=1)
				+ np.roll(np.roll(gray_sq, -1, axis=0), 1, axis=1)
				+ np.roll(np.roll(gray_sq, -1, axis=0), -1, axis=1)
		) / 9.0
		local_std = np.sqrt(np.maximum(local_sq_mean - (local_mean * local_mean), 0.0))
		return grad_mag, lap, local_std


def _extract_kibble_components(mask: np.ndarray) -> tuple[list[dict], np.ndarray]:
		height, width = mask.shape
		visited = np.zeros((height, width), dtype=bool)
		kept_mask = np.zeros((height, width), dtype=bool)
		boxes: list[dict] = []

		for y in range(height):
				for x in range(width):
						if not mask[y, x] or visited[y, x]:
								continue

						stack = [(y, x)]
						visited[y, x] = True
						pixels: list[tuple[int, int]] = []
						min_x = x
						max_x = x
						min_y = y
						max_y = y

						while stack:
								cy, cx = stack.pop()
								pixels.append((cy, cx))
								if cx < min_x:
										min_x = cx
								if cx > max_x:
										max_x = cx
								if cy < min_y:
										min_y = cy
								if cy > max_y:
										max_y = cy

								for ny in range(max(0, cy - 1), min(height, cy + 2)):
										for nx in range(max(0, cx - 1), min(width, cx + 2)):
												if mask[ny, nx] and not visited[ny, nx]:
														visited[ny, nx] = True
														stack.append((ny, nx))

						area = len(pixels)
						box_w = (max_x - min_x + 1)
						box_h = (max_y - min_y + 1)
						aspect = max(box_w / max(box_h, 1), box_h / max(box_w, 1))
						fill = area / max(box_w * box_h, 1)

						is_kibble = area >= 3 and area <= 260 and aspect <= 3.6 and fill >= 0.14
						if not is_kibble:
								continue

						for py, px in pixels:
								kept_mask[py, px] = True

						boxes.append(
								{
										"x1": min_x / width,
										"y1": min_y / height,
										"x2": (max_x + 1) / width,
										"y2": (max_y + 1) / height,
										"pixel_area": area,
								}
						)

		return boxes, kept_mask


def _estimate_kibble_stats(
		*,
		gray: np.ndarray,
		saturation: np.ndarray,
		grad_mag: np.ndarray,
		lap: np.ndarray,
		local_std: np.ndarray,
		container_mask: np.ndarray,
) -> dict:
		container_pixels = int(container_mask.sum())
		in_container_grad = grad_mag[container_mask]
		in_container_lap = lap[container_mask]
		in_container_std = local_std[container_mask]
		in_container_sat = saturation[container_mask]

		grad_threshold = float(np.percentile(in_container_grad, 60))
		lap_threshold = float(np.percentile(in_container_lap, 58))
		std_threshold = float(np.percentile(in_container_std, 55))
		val = gray
		brown_mask = (
				(saturation >= 0.18)
				& (val >= 0.10)
				& (val <= 0.92)
		)
		dark_kibble_mask = (
				(saturation >= 0.10)
				& (val >= 0.06)
				& (val <= 0.58)
		)
		texture_mask = (
				(grad_mag >= grad_threshold)
				& (lap >= lap_threshold)
				& (local_std >= std_threshold)
		)
		kibble_like = (brown_mask & texture_mask) | (dark_kibble_mask & texture_mask)

		neighbor_count = (
				kibble_like.astype(np.uint8)
				+ np.roll(kibble_like.astype(np.uint8), 1, axis=0)
				+ np.roll(kibble_like.astype(np.uint8), -1, axis=0)
				+ np.roll(kibble_like.astype(np.uint8), 1, axis=1)
				+ np.roll(kibble_like.astype(np.uint8), -1, axis=1)
				+ np.roll(np.roll(kibble_like.astype(np.uint8), 1, axis=0), 1, axis=1)
				+ np.roll(np.roll(kibble_like.astype(np.uint8), 1, axis=0), -1, axis=1)
				+ np.roll(np.roll(kibble_like.astype(np.uint8), -1, axis=0), 1, axis=1)
				+ np.roll(np.roll(kibble_like.astype(np.uint8), -1, axis=0), -1, axis=1)
		)
		kibble_like = kibble_like & (neighbor_count >= 2)
		raw_kibble_mask = kibble_like & container_mask
		kibble_boxes, food_mask = _extract_kibble_components(raw_kibble_mask)
		kibble_ratio = float(np.sum(food_mask) / max(container_pixels, 1))
		kibble_count = len(kibble_boxes)
		total_food_pixels = int(np.sum(food_mask))
		largest_kibble_pixels = max((int(b["pixel_area"]) for b in kibble_boxes), default=0)
		largest_kibble_share = float(largest_kibble_pixels / max(total_food_pixels, 1))
		component_density = float(kibble_count / max(container_pixels / 1000.0, 1e-6))

		return {
				"kibble_ratio": kibble_ratio,
				"kibble_count": kibble_count,
				"largest_kibble_share": largest_kibble_share,
				"component_density": component_density,
		}


def detect_cat_food(image: Image.Image) -> dict:
		image = image.convert("RGB").resize((320, 320))
		arr = np.array(image)

		height, width = arr.shape[:2]

		gray = np.mean(arr.astype(np.float32), axis=2) / 255.0
		hsv = rgb_to_hsv(arr)
		saturation = hsv[..., 1]
		dx = np.abs(np.diff(gray, axis=1))
		dy = np.abs(np.diff(gray, axis=0))
		texture = float((dx.mean() + dy.mean()) / 2.0)
		grad_mag, lap, local_std = _texture_metrics(gray)

		brightness = float(gray.mean())

		yolo_mask, yolo_reason, bowl_box = detect_bowl_mask(arr)
		container_mask = yolo_mask
		container_pixels = int(container_mask.sum())
		if container_pixels == 0:
				return {
						"detected": False,
						"status": "empty",
						"confidence": 0,
						"fill_percent": 0,
						"bowl_box": None,
						"reason": f"strict reject: no container, yolo={yolo_reason}, brightness={brightness:.2f}",
				}
		container_source = "yolo"

		in_container_grad = grad_mag[container_mask]
		in_container_lap = lap[container_mask]
		in_container_std = local_std[container_mask]
		in_container_sat = saturation[container_mask]

		edge_density = float(np.mean(in_container_grad))
		lap_density = float(np.mean(in_container_lap))
		texture_density = float(np.mean(in_container_std))
		bowl_conf = float(bowl_box.get("confidence", 0.0)) if bowl_box else 0.0

		kibble_stats = _estimate_kibble_stats(
				gray=gray,
				saturation=saturation,
				grad_mag=grad_mag,
				lap=lap,
				local_std=local_std,
				container_mask=container_mask,
		)
		kibble_ratio = float(kibble_stats["kibble_ratio"])
		kibble_count = int(kibble_stats["kibble_count"])
		largest_kibble_share = float(kibble_stats["largest_kibble_share"])
		component_density = float(kibble_stats["component_density"])

		strict_reject = False
		if kibble_ratio < 0.0035:
				strict_reject = True
		if kibble_count < 2 and kibble_ratio < 0.018:
				strict_reject = True
		if largest_kibble_share > 0.72 and kibble_ratio < 0.055:
				strict_reject = True
		if component_density < 0.030 and kibble_ratio < 0.030:
				strict_reject = True

		raw_fill_ratio = kibble_ratio
		calibrated_capacity = 0.12
		fill_ratio = min(raw_fill_ratio / calibrated_capacity, 1.0)
		food_score = min(raw_fill_ratio / max(calibrated_capacity * 0.80, 0.01), 1.0)
		component_score = min(kibble_count / 18.0, 1.0)
		texture_score = min(texture_density / 0.08, 1.0)
		score = (
				(0.34 * food_score)
				+ (0.16 * component_score)
				+ (0.18 * texture_score)
				+ (0.32 * min(bowl_conf / 0.65, 1.0))
		)
		if strict_reject:
			score = 0.0
			fill_ratio = 0.0

		confidence = int(max(0.0, min(score, 1.0)) * 100)
		fill_percent = int(max(0.0, min(fill_ratio, 1.0)) * 100)

		if fill_percent <= 3:
				status = "empty"
		elif fill_percent <= 20:
				status = "low"
		else:
				status = "ok"
		detected = (fill_percent > 3) and (confidence >= 30)

		return {
				"detected": detected,
				"status": status,
				"confidence": confidence,
				"fill_percent": fill_percent,
				"bowl_box": bowl_box,
				"reason": (
						f"kibble_ratio={kibble_ratio:.2f}, "
						f"kibble_count={kibble_count}, largest_share={largest_kibble_share:.2f}, comp_density={component_density:.3f}, "
						f"fill_raw={raw_fill_ratio:.2f}, texture={texture:.3f}, "
						f"edge={edge_density:.3f}, lap={lap_density:.3f}, local_std={texture_density:.3f}, "
						f"capacity={calibrated_capacity:.3f}, bowl_conf={bowl_conf:.2f}, "
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
