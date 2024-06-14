import face_recognition
from PIL import Image, ImageDraw
image = face_recognition.load_image_file("/mnt/c/Users/darin/ai_robots_v2/lab1/main/python/car.jpg")
face_landmarks_list = face_recognition.face_locations(image)
print(face_landmarks_list)
pil_image = Image.fromarray(image)
draw = ImageDraw.Draw(pil_image)

if len(face_landmarks_list) > 0:
    left = face_landmarks_list[0][3]
    right = face_landmarks_list[0][1]
    top = face_landmarks_list[0][0]
    bottom = face_landmarks_list[0][2]
    draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
    pil_image.save("/mnt/c/Users/darin/ai_robots_v2/lab1/main/python/car.jpg")

    # Calculating area
    area = (right - left) * (bottom - top)
    print(area)