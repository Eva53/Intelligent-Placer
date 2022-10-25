import imghdr
import cv2
from os import listdir, path
import matplotlib.pyplot as plt

COMPRESS_PERCENT = 10

def get_images_path(folder_path):
    images = []
    for image_path in listdir(folder_path):
        image_full_path = path.join(folder_path, image_path)

        if imghdr.what(image_full_path) == 'jpeg':
            images.append(image_full_path)
    return images

def compress_image(image):
    height, width = image.shape[:2]
    if height > 500 or width > 500:
        width = int(image.shape[1] * COMPRESS_PERCENT / 100)
        height = int(image.shape[0] * COMPRESS_PERCENT / 100)
        new_size = (width, height)
        return cv2.resize(image, new_size)
    return image

def get_object_mask(path_to_image):
    image = cv2.imread(path_to_image)
    image = compress_image(image)

    image_BGR = image.copy()
    image_BGR = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB)

    # Преобразование изображения в изображение в градациях серого, выполнение размытия
    # по Гауссу и преобразование в двоичное изображение
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (1, 1), 0)
    edged = cv2.Canny(blur, 100, 400)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    image_binary = cv2.threshold(closed, 150, 250, cv2.THRESH_BINARY)[1]

    # Извлечение контуров из двоичного изображения
    # contours содержит все обнаруженные контуры, а также точки координат каждого контура
    contours = cv2.findContours(image_binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Обвести все обнаруженные контуры и нарисовать обнаруженные точки координат на изображении
    for c in contours:
        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
        cv2.fillPoly(image_binary, pts=[c], color=(255, 0, 0))

    image_contours = image

    # выведем изображения
    imgs = []
    imgs.append(image_BGR)
    imgs.append(image_binary)
    imgs.append(image_contours)

    axs = plt.subplots(1, 3, figsize=(12, 12))[1]
    axs = axs.flatten()

    for im, ax in zip(imgs, axs):
        ax.imshow(im)
    axs[0].set_title("image_BGR")
    axs[1].set_title("image_binary")
    axs[2].set_title('{} contours'.format(len(contours)))

    plt.show()



