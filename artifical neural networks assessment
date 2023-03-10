In [ ]:
import numpy as np
from tensorflow.keras.preprocessing import image
from sklearn.metrics import classification_report
In [ ]:
# Load test data
test_data = []
test_labels = []
for label in ['burger', 'pizza', 'coke']:
    for i in range(1, 101):
        img_path = f'test/{label}/{label}_{i}.jpg'
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        test_data.append(x)
        test_labels.append(label)

test_data = np.vstack(test_data)
test_labels = np.array(test_labels)
In [ ]:
# Make predictions
predictions = model.predict(test_data)
predicted_labels = np.argmax(predictions, axis=1)
predicted_labels = ['burger' if label==0 else 'pizza' if label==1 else 'coke' for label in predicted_labels]
In [ ]:
# Compute performance metrics
report = classification_report(test_labels, predicted_labels)
print(report)
