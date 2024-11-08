
import tensorflow as tf
import numpy as np
import cv2

# Load the saved model
def load_model():
    try:
        model = tf.keras.models.load_model('final_mnist_model.keras')
        print('Loaded saved model: final_mnist_model.keras')
        print(model.summary())
    except Exception as e:
        print("Error loading the model:", e)
        model = None
    return model

# Preprocess the input image for prediction
def preprocess_image(img):
    img = np.expand_dims(img, axis=-1)  # Add a single channel dimension
    img = tf.image.resize(img, (28, 28))  # Resize to 28x28
    img = img / 255.0  # Normalize
    return img


# Predict digit using the model and processed image
def predict(model, img):
    img = preprocess_image(img)
    imgs = np.array([img])  # Add batch dimension
    res = model.predict(imgs)
    index = np.argmax(res)
    return str(index)

# OpenCV functions for webcam display
startInference = False
def ifClicked(event, x, y, flags, params):
    global startInference
    if event == cv2.EVENT_LBUTTONDOWN:
        startInference = not startInference

threshold = 100
def on_threshold(x):
    global threshold
    threshold = x

def start_cv(model):
    global threshold
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('background')
    cv2.setMouseCallback('background', ifClicked)
    cv2.createTrackbar('threshold', 'background', 150, 255, on_threshold)
    background = np.zeros((480, 640), np.uint8)
    frameCount = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if startInference:
            frameCount += 1
            frame[0:480, 0:80] = 0
            frame[0:480, 560:640] = 0
            grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thr = cv2.threshold(grayFrame, threshold, 255, cv2.THRESH_BINARY_INV)
            resizedFrame = thr[240-75:240+75, 320-75:320+75]
            background[240-75:240+75, 320-75:320+75] = resizedFrame
            iconImg = cv2.resize(resizedFrame, (28, 28))
            res = predict(model, iconImg)

            if frameCount == 5:
                background[0:480, 0:80] = 0
                frameCount = 0

            cv2.putText(background, res, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            cv2.rectangle(background, (320-80, 240-80), (320+80, 240+80), (255, 255, 255), thickness=3)
            cv2.imshow('background', background)
        else:
            cv2.imshow('background', frame)

        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main function to load the model and start the OpenCV loop
def main():
    model = load_model()
    if model is not None:
        print("Starting OpenCV interface...")
        start_cv(model)

if __name__ == '__main__':
    main()






