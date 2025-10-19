import cv2
import mediapipe as mp

# Инициализируем модель Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2)

# Загружаем изображение
image_path = "2chel.jpg"  # <- замени на имя твоего файла
img = cv2.imread(image_path)
if img is None:
    print(f"Ошибка: не удалось загрузить изображение {image_path}")
    exit()

# Обрабатываем изображение
results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# Проверяем, найдена ли поза
if results.pose_landmarks:
    print("Поза обнаружена!")
else:
    print("Поза не найдена.")

# Рисуем ключевые точки, если они есть
mp.solutions.drawing_utils.draw_landmarks(
    img,
    results.pose_landmarks,
    mp_pose.POSE_CONNECTIONS
)

# Показываем результат
cv2.imshow("Pose Detection Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
