import cv2

# Load the reference image
reference_image = cv2.imread('burn.jpg', cv2.IMREAD_GRAYSCALE)

# Create the SIFT detector
sift = cv2.SIFT_create()

# Find keypoints and descriptors in the reference image
kp1, des1 = sift.detectAndCompute(reference_image, None)

# Create a FLANN-based matcher
flann = cv2.FlannBasedMatcher()

# Initialize a video capture object (you can also use cv2.VideoCapture(0) for live webcam feed)
cap = cv2.VideoCapture('video.mp4')

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find keypoints and descriptors in the frame
    kp2, des2 = sift.detectAndCompute(gray_frame, None)

    # Match keypoints between the reference image and the frame
    matches = flann.knnMatch(des1, des2, k=2)

    # Apply ratio test to get good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Draw matches on the frame
    result_frame = cv2.drawMatches(reference_image, kp1, gray_frame, kp2,
                                   good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Display the resulting frame
    cv2.imshow('Object Detection', result_frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()
