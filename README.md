# vehicle-plate-recognition

> Đồ án này tập trung xây dựng một hệ thống nhận diện biển số xe dựa trên Computer Vision và Deep Learning, bao gồm:

+ Trần Biểu Hương : Backend / Server
Xây dựng server Python, API xử lý ảnh, tích hợp mô hình
+ Phạm Quốc Huy :Computer Vision
Train mô hình nhận diện biển số bằng YOLOv5
+ Nguyễn Lê Huy : Frontend
Xây dựng web tĩnh HTML hiển thị kết quả

## 1/Phân tích Computer Vision

Bài toán nhận diện biển số xe gồm 2 bước chính:

+ Phát hiện biển số :
→ Xác định vùng chứa biển số trong ảnh

+ Nhận diện ký tự :
→ Trích xuất và đọc các ký tự trên biển số

Mô hình sử dụng : YOLOv5

Dataset : roboflow , github

## 2/Phân tích Server Python (Backend)

Công nghệ sử dụng

+ Ngôn ngữ: Python

+ Mô hình: YOLOv5

+ Server: HTTP Server

+ Môi trường: Virtual Environment (venv)

## 3/Giao diện Web (Frontend)

Web tĩnh viết bằng HTML

Chức năng:

+ Gửi ảnh lên server

+ Hiển thị kết quả nhận diện

+ Đóng vai trò minh họa và kiểm thử hệ thống