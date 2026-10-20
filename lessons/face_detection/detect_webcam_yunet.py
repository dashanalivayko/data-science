"""
Детекция лиц с веб-камеры в реальном времени.
Два детектора с переключением на лету.

Запуск:
    python detect_webcam_yunet.py

Клавиши:
    h - переключиться на каскад Хаара
    y - переключиться на YuNet (по умолчанию)
    q или Esc - выход

ВАЖНО (macOS): при первом запуске система спросит доступ к камере.
Разрешение выдаётся не Python, а приложению-терминалу, из которого запускаешь:
Системные настройки -> Конфиденциальность и безопасность -> Камера.
"""
from pathlib import Path
import time

import cv2

MODEL_PATH = Path(__file__).parent / "models" / "yunet.onnx"
CAMERA_INDEX = 0        # 0 - встроенная камера; если камер несколько, пробуй 1, 2
CONF_THRESHOLD = 0.8    # порог уверенности для YuNet


def main():
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if cascade.empty():
        raise RuntimeError("Не удалось загрузить каскад Хаара")

    if not MODEL_PATH.exists():
        raise RuntimeError(f"Нет файла модели: {MODEL_PATH}")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Камера не открылась.")
        print("Проверь доступ: Системные настройки -> Конфиденциальность -> Камера")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Камера: {w}x{h}")

    # Детектор создаём ОДИН раз до цикла: создавать его на каждом кадре
    # значит заново читать модель с диска - это убьёт скорость.
    yunet = cv2.FaceDetectorYN.create(
        str(MODEL_PATH), "", (w, h), CONF_THRESHOLD, 0.3, 5000
    )

    mode = "yunet"      # какой детектор активен
    prev = time.time()
    fps = 0.0

    print("Клавиши: h - Хаар, y - YuNet, q - выход")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Кадр не получен, выходим.")
                break

            # Зеркалим: так картинка ведёт себя привычно, как в зеркале.
            frame = cv2.flip(frame, 1)

            if mode == "haar":
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)
                faces = cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.2,    # на видео шаг крупнее: важнее скорость
                    minNeighbors=5,
                    minSize=(60, 60),   # лицо у камеры крупное, мелочь не ищем
                )
                for (x, y, fw, fh) in faces:
                    cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
                count = len(faces)

            else:
                yunet.setInputSize((frame.shape[1], frame.shape[0]))
                _, faces = yunet.detect(frame)
                faces = faces if faces is not None else []
                for f in faces:
                    x, y, fw, fh = [int(v) for v in f[:4]]
                    conf = float(f[14])
                    cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
                    cv2.putText(frame, f"{conf:.2f}", (x, max(20, y - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    for i in range(5):
                        cv2.circle(frame, (int(f[4 + i * 2]), int(f[5 + i * 2])),
                                   2, (255, 0, 255), -1)
                count = len(faces)

            # FPS считаем сглаженно, иначе число прыгает и его не прочитать.
            now = time.time()
            inst = 1.0 / max(now - prev, 1e-6)
            prev = now
            fps = inst if fps == 0 else 0.9 * fps + 0.1 * inst

            label = "YuNet" if mode == "yunet" else "Haar"
            cv2.putText(frame, f"{label} | faces: {count} | fps: {fps:.1f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, "h=Haar  y=YuNet  q=exit",
                        (10, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("webcam", frame)

            # waitKey(1) - ждём 1 мс и идём дальше. Без этого кадр не отрисуется.
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("h"):
                mode = "haar"
            elif key == ord("y"):
                mode = "yunet"
    finally:
        # Камеру освобождать обязательно, иначе останется занятой.
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
