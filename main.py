import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

noise = np.random.normal(0, 25, gray.shape).astype(np.uint8)
noisy_img = cv2.add(gray, noise)

gauss_blur = cv2.GaussianBlur(noisy_img, (5, 5), 0)

mean_blur = cv2.blur(noisy_img, (5, 5))

kernel_binom = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
binom_blur = cv2.filter2D(noisy_img, -1, kernel_binom)

s_p_noise = np.copy(gray)
s_p_noise[np.random.randint(0, gray.shape[0], 500), np.random.randint(0, gray.shape[1], 500)] = 255
s_p_noise[np.random.randint(0, gray.shape[0], 500), np.random.randint(0, gray.shape[1], 500)] = 0

median_blur = cv2.medianBlur(s_p_noise, 5)

kernel = np.ones((5,5), np.uint8)
max_filter = cv2.dilate(s_p_noise, kernel)

titles = ['Original', 'Bruité', 'Gaussien', 'Moyenne', 'Médian', 'Maximum']
images = [gray, noisy_img, gauss_blur, mean_blur, median_blur, max_filter]

plt.figure(figsize=(15, 10))
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()


# Plus la valeur est haute, meilleure est la qualité
psnr_median = cv2.PSNR(gray, median_blur)
print(f"PSNR Médian: {psnr_median} dB")