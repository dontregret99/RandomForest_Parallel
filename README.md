# Lập trình song song ứng dụng 
## Song song hóa thuật toán Random Forest

## Thành viên:
| No.  | Họ và tên | MSSV | Email |
| ------------- | ------------- | ------------- | ------------- |
| 1  | Đặng Thái Gia Thuận | 1712173 | giathuandang16031999@gmail.com|
| 2  | Lê Đức Hòa | 1712449 | leduchoabatr12@gmail.com| 
| 3  | Ngô Hoàng Sinh  | 1412457|  |

## 1. Mô tả ứng dụng: (Song song hóa) Ứng dụng thuật toán Random Forest để phân lớp tập dữ liệu Fashion MNIST

- Input: tập dữ liệu Fashion MNIST

- Output: Mô hình Random Forest đã được huấn luyện để phân loại trang phục từ bộ dữ liệu Fashion MNIST

- Ứng dụng có cần tăng tốc không ? Đối với các trường hợp số lượng cây (tree) nhiều thì Random Forest rất chậm, không phù hợp cho những dự đoán yêu cầu real-time.

- Ứng dụng có thể song song hóa không ? Có thể song song hóa ở 2 bước: 
  + Quá trình training: có thể song song huấn luyện cho từng cây (tree) riêng biệt
  + Quá trình predict: có thể song song dự đoán với từng cây (rồi sau đó tổng hợp kết quả từ tất cả cây để đưa ra predict cuối cùng)
  
## 2. Cài đặt tuần tự
## 3. Cài đặt song song trên GPU
## 4. Cài đặt song song trên GPU + tối ưu 
## 5. Nhìn lại quá trình làm đồ án
## 6. Tài liệu tham khảo
