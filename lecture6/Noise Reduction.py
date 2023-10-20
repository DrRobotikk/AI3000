import cv2


# Load the noisy image
noisy_image = cv2.imread('noisy_image.jpg')

# Apply Gaussian blur for noise reduction
kernel_size = (5, 5)
blurred_image = cv2.GaussianBlur(noisy_image, kernel_size, 0)

# Save the denoised image
cv2.imwrite('denoised_image.jpg', blurred_image)

# Display the original and denoised images
cv2.imshow('Original Image', noisy_image)
cv2.imshow('Denoised Image', blurred_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
