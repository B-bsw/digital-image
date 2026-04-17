# Cat Food Detector

> สำหรับวิชา DIP ( Digital image processing )

# About

> Cat Food Detector เป็นโปรเจคที่พัฒนาเพื่อใช้ในวิชา Digital Image Processing (DIP) โดยมีเป้าหมายในการตรวจจับและจำแนกอาหารแมวจากภาพถ่า>ยโดยใช้เทคนิคด้าน Computer Vision และโมเดล Deep Learning
>
> โปรเจคนี้ใช้โมเดล YOLOv8 ในการตรวจจับวัตถุ (object detection) เพื่อระบุประเภทของอาหารแมวจากภาพที่ผู้ใช้อัปโหลด ผ่าน Web Application ที่ส>ามารถใช้งานได้บนเบราว์เซอร์

# Getting Started

```bash
# check python3 version ( recommend: python v3.11  )
python3 --version

# download dependencies
pip3 install -r requirements.txt

# run app
python3 -m app
```

> หมายเหตุ: ครั้งแรกที่รัน YOLO อาจดาวน์โหลดโมเดล yolov8n.pt อัตโนมัติ

```bash
# follow port 300 on website
http://localhost:3000
#or
http://0.0.0.0:3000
```

# Team

- [@BB](https://github.com/b-bsw)
- [@Kittichai Raksawong](https://github.com/jrKitt)
