import base64
import io

import numpy as np
from flask import Flask, render_template_string, request
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
			img { max-width: 100%; border-radius: 8px; border: 1px solid #ddd; }
			small { color: #666; }
		</style>
	</head>
	<body>
		<div class="card">
			<h2>เว็บตรวจจับอาหารแมว</h2>
			<p>อัปโหลดรูปถ้วย/จานอาหารแมว ระบบจะประเมินจากลักษณะสีและพื้นผิวของเม็ดอาหาร</p>
			<form method="post" enctype="multipart/form-data">
				<input type="file" name="image" accept="image/*" required />
				<button type="submit">ตรวจจับ</button>
			</form>

			{% if result %}
				<hr />
				<p class="{{ 'ok' if result.detected else 'no' }}">
					ผลลัพธ์: {{ 'พบอาหารแมว' if result.detected else 'ไม่พบอาหารแมว' }}
				</p>
				<p>ความมั่นใจ: {{ result.confidence }}%</p>
				<p><small>{{ result.reason }}</small></p>
				<img src="data:image/jpeg;base64,{{ preview }}" alt="preview" />
			{% endif %}
		</div>
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


def to_preview_base64(image: Image.Image) -> str:
		out = io.BytesIO()
		image.convert("RGB").save(out, format="JPEG", quality=85)
		return base64.b64encode(out.getvalue()).decode("utf-8")


@app.route("/", methods=["GET", "POST"])
def index():
		result = None
		preview = None

		if request.method == "POST":
				file = request.files.get("image")
				if file and file.filename:
						image = Image.open(file.stream)
						result = detect_cat_food(image)
						preview = to_preview_base64(image.resize((480, 360)))

		return render_template_string(HTML, result=result, preview=preview)


if __name__ == "__main__":
		app.run(host="127.0.0.1", port=5000, debug=True)
