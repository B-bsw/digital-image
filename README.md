# Cat Food Detector
ตรวจจับอาหารแมวจากกล้องแบบเรียลไทม์ โดยใช้ YOLO ตรวจจับตำแหน่งชาม และประเมิน `% ปริมาณในชาม` จากพื้นที่อาหารที่พบใน ROI ชาม

## ติดตั้ง

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> หมายเหตุ: ครั้งแรกที่รัน YOLO อาจดาวน์โหลดโมเดล `yolov8n.pt` อัตโนมัติ


```bash
python3 -m app
```


```text
http://127.0.0.1:3000
```
