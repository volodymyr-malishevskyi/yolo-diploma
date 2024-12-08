from ultralytics import YOLO
import cv2

model = YOLO('model/weights/best.pt')

# Ініціалізуємо відеопотік (0 - камера)
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('test1.mp4')
# cap = cv2.VideoCapture('http://192.168.3.6:8080/video')

# Перевіряємо, чи камера успішно відкрилась
if not cap.isOpened():
    print("Не вдалося відкрити камеру")
    exit()

# Ширина кадр для конвертації вхідного зображення моделі
desired_width = 640

# Цільова впевненість для відображення
confidence_threshold = 0.6

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Зміна розмірів кадру перед подачею на вхід моделі
    height, width = frame.shape[:2]
    aspect_ratio = desired_width / width
    new_height = int(height * aspect_ratio)

    frame = cv2.resize(frame, (desired_width, new_height))

    # Виконання детекції
    results = model(frame, conf=confidence_threshold)
    
    for box in results[0].boxes:
        # Отримання координат коробки (x1, y1, x2, y2)
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Отримання класу та його назви
        cls = int(box.cls[0])
        label = model.names[cls]
        
        # Отримання впевненості (можна використовувати для відображення)
        confidence = box.conf[0]
        label_with_confidence = f"{label} {confidence:.2f}"
        
        # Малювання прямокутника (рамки)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Відображення назви об'єкта
        cv2.putText(frame, label_with_confidence, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Отримання кількості детектованих об'єктів
    pen_caps = results[0].boxes.cls  # Отримання класів об'єктів
    count = len(pen_caps)

    # Відображення кількості на кадрі
    cv2.putText(frame, f'Count: {count}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Відображення кадру
    cv2.imshow('Pen Cap Detection', frame)

    # Вихід по натисканню 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()