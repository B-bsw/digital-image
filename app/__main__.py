import base64
import io

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
	<body class="min-h-screen bg-gradient-to-br from-rose-50 via-pink-50 to-violet-100 p-5 sm:p-8 text-slate-700" style="font-family: 'Mali', cursive;">
		<div class="mx-auto w-full max-w-3xl rounded-3xl border border-rose-100 bg-white/90 p-5 shadow-xl shadow-rose-100/60 backdrop-blur sm:p-8">
			<div class="mb-5 border-b border-rose-100 pb-4">
				<h2 class="text-2xl font-bold text-rose-700 sm:text-3xl">เว็บตรวจจับอาหารแมว</h2>
				<p class="mt-2 text-sm text-slate-600 sm:text-base">เปิดกล้องเพื่อวิเคราะห์ภาพแบบเรียลไทม์ ระบบจะประเมินจากสีและพื้นผิวของภาพแต่ละเฟรม</p>
			</div>

			<div class="rounded-2xl border border-rose-100 bg-rose-50/60 p-3">
				<video id="camera" autoplay playsinline muted class="aspect-video w-full rounded-xl border border-rose-200 bg-slate-900"></video>
			</div>

			<div class="mt-4 flex flex-wrap gap-2">
				<button id="startBtn" type="button" class="rounded-xl bg-rose-500 px-4 py-2 text-sm font-semibold text-white transition hover:bg-rose-600 focus:outline-none focus:ring-2 focus:ring-rose-300">เริ่มกล้อง</button>
				<button id="stopBtn" type="button" disabled class="rounded-xl bg-slate-300 px-4 py-2 text-sm font-semibold text-slate-600 transition disabled:cursor-not-allowed">หยุด</button>
			</div>

			<div class="mt-5 space-y-2 rounded-2xl border border-violet-100 bg-violet-50/60 p-4">
				<p id="resultText" class="text-sm font-semibold text-violet-700 sm:text-base">ผลลัพธ์: ยังไม่ได้เริ่ม</p>
				<p id="confidenceText" class="text-sm text-slate-700">ความมั่นใจ: -</p>
				<p id="reasonText" class="text-xs text-slate-500 sm:text-sm">-</p>
			</div>
		</div>

		<script>
			const video = document.getElementById("camera");
			const startBtn = document.getElementById("startBtn");
			const stopBtn = document.getElementById("stopBtn");
			const resultText = document.getElementById("resultText");
			const confidenceText = document.getElementById("confidenceText");
			const reasonText = document.getElementById("reasonText");

			const canvas = document.createElement("canvas");
			let stream = null;
			let timerId = null;

			const statusBaseClass = "text-sm font-semibold sm:text-base";
			const statusOkClass = "text-emerald-600";
			const statusNoClass = "text-rose-600";
			const statusIdleClass = "text-violet-700";

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
				startBtn.className = isRunning
					? "rounded-xl bg-rose-300 px-4 py-2 text-sm font-semibold text-white transition disabled:cursor-not-allowed"
					: "rounded-xl bg-rose-500 px-4 py-2 text-sm font-semibold text-white transition hover:bg-rose-600 focus:outline-none focus:ring-2 focus:ring-rose-300";
				stopBtn.className = isRunning
					? "rounded-xl bg-violet-500 px-4 py-2 text-sm font-semibold text-white transition hover:bg-violet-600"
					: "rounded-xl bg-slate-300 px-4 py-2 text-sm font-semibold text-slate-600 transition disabled:cursor-not-allowed";
			}

			async function analyzeFrame() {
				if (!stream || video.videoWidth === 0 || video.videoHeight === 0) {
					return;
				}

				canvas.width = video.videoWidth;
				canvas.height = video.videoHeight;
				const ctx = canvas.getContext("2d");
				ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

				const imageData = canvas.toDataURL("image/jpeg", 0.7);

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
					setStatus(result.detected ? "ok" : "no", `ผลลัพธ์: ${result.detected ? "พบอาหารแมว" : "ไม่พบอาหารแมว"}`);
					confidenceText.textContent = `ความมั่นใจ: ${result.confidence}%`;
					reasonText.textContent = result.reason;
				} catch (_) {
					setStatus("no", "ผลลัพธ์: วิเคราะห์ไม่สำเร็จ");
					confidenceText.textContent = "ความมั่นใจ: -";
					reasonText.textContent = "โปรดลองใหม่อีกครั้ง";
				}
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
				setButtonsRunning(false);
				setStatus("idle", "ผลลัพธ์: หยุดการตรวจจับ");
				confidenceText.textContent = "ความมั่นใจ: -";
				reasonText.textContent = "-";
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


def detect_cat_food(image: Image.Image) -> dict:
		image = image.convert("RGB").resize((320, 320))
		arr = np.array(image)

		hsv = rgb_to_hsv(arr)
		h = hsv[..., 0] * 360.0
		s = hsv[..., 1]
		v = hsv[..., 2]

		warm_brown = (h >= 12) & (h <= 45) & (s >= 0.25) & (v >= 0.12) & (v <= 0.82)
		warm_ratio = float(np.mean(warm_brown))

		gray = np.mean(arr.astype(np.float32), axis=2)
		dx = np.abs(np.diff(gray, axis=1))
		dy = np.abs(np.diff(gray, axis=0))
		texture = float((dx.mean() + dy.mean()) / 2.0)
		texture_norm = min(texture / 35.0, 1.0)

		score = 0.65 * min(warm_ratio / 0.16, 1.0) + 0.35 * texture_norm
		confidence = int(max(0.0, min(score, 1.0)) * 100)
		detected = confidence >= 52

		return {
				"detected": detected,
				"confidence": confidence,
				"reason": f"สัดส่วนโทนสีน้ำตาล={warm_ratio:.2f}, ความหยาบพื้นผิว={texture:.2f}",
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
		app.run(host="127.0.0.1", port=5000, debug=True)
