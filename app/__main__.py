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
		<style>
			body { font-family: Arial, sans-serif; margin: 32px; max-width: 760px; }
			.card { border: 1px solid #ddd; border-radius: 10px; padding: 20px; }
			.ok { color: #0a7d1f; font-weight: 700; }
			.no { color: #b00020; font-weight: 700; }
			video { width: 100%; max-width: 640px; border-radius: 8px; border: 1px solid #ddd; background: #111; }
			button { margin-right: 8px; }
			small { color: #666; }
		</style>
	</head>
	<body>
		<div class="card">
			<h2>เว็บตรวจจับอาหารแมว</h2>
			<p>เปิดกล้องเพื่อวิเคราะห์ภาพแบบ realtime ระบบจะประเมินจากสีและพื้นผิวของภาพแต่ละเฟรม</p>
			<video id="camera" autoplay playsinline muted></video>
			<div style="margin-top: 10px;">
				<button id="startBtn" type="button">เริ่มกล้อง</button>
				<button id="stopBtn" type="button" disabled>หยุด</button>
			</div>
			<hr />
			<p id="resultText">ผลลัพธ์: ยังไม่ได้เริ่ม</p>
			<p id="confidenceText">ความมั่นใจ: -</p>
			<p><small id="reasonText">-</small></p>
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
					resultText.className = result.detected ? "ok" : "no";
					resultText.textContent = `ผลลัพธ์: ${result.detected ? "พบอาหารแมว" : "ไม่พบอาหารแมว"}`;
					confidenceText.textContent = `ความมั่นใจ: ${result.confidence}%`;
					reasonText.textContent = result.reason;
				} catch (_) {
					resultText.className = "no";
					resultText.textContent = "ผลลัพธ์: วิเคราะห์ไม่สำเร็จ";
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
					startBtn.disabled = true;
					stopBtn.disabled = false;

					timerId = window.setInterval(analyzeFrame, 500);
					analyzeFrame();
				} catch (_) {
					resultText.className = "no";
					resultText.textContent = "ผลลัพธ์: ไม่สามารถเปิดกล้องได้";
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
				startBtn.disabled = false;
				stopBtn.disabled = true;
				resultText.className = "";
				resultText.textContent = "ผลลัพธ์: หยุดการตรวจจับ";
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
