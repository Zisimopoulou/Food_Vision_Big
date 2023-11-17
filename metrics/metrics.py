from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score
import tensorflow_datasets as tfds
from sklearn.preprocessing import LabelEncoder
import itertools

def create_confusion_matrix(true_labels, predictions, classes=None, figsize=(20,20), text_size=10, norm=False, savefig=False): 

  cm = confusion_matrix(true_labels, predictions)
  cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] 
  n_classes = cm.shape[0] 

  fig, ax = plt.subplots(figsize=figsize)
  cax = ax.matshow(cm, cmap=plt.cm.Blues)
  fig.colorbar(cax)

  if classes:
    labels = classes
  else:
    labels = np.arange(cm.shape[0])
  
  ax.set(title="Confusion Matrix",
         xlabel="Predicted label",
         ylabel="True label",
         xticks=np.arange(n_classes), 
         yticks=np.arange(n_classes), 
         xticklabels=labels, 
         yticklabels=labels)
  
  ax.xaxis.set_label_position("bottom")
  ax.xaxis.tick_bottom()

  plt.xticks(rotation=70, fontsize=text_size)
  plt.yticks(fontsize=text_size)

  threshold = (cm.max() + cm.min()) / 2.

  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    if norm:
      plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
              horizontalalignment="center",
              color="white" if cm[i, j] > threshold else "black",
              size=text_size)
    else:
      plt.text(j, i, f"{cm[i, j]}",
              horizontalalignment="center",
              color="white" if cm[i, j] > threshold else "black",
              size=text_size)

  if savefig:
    fig.savefig("/content/drive/MyDrive/Projects/food_vision/images/confusion_matrix.png")
    
def calculate_metrics(model, test_data):
    predictions = model.predict(test_data, verbose=1)
    pred_classes = predictions.argmax(axis=1)

    # label_encoder = LabelEncoder()
    true_labels = []
    for images, labels in test_data.unbatch():  
        true_labels.append(labels.numpy())  
    # true_labels = label_encoder.fit_transform(true_labels)  

    return pred_classes, true_labels
    
def find_most_wrong_predictions(true_labels, predictions):
    most_wrong_preds = []

    predictions = np.array(predictions)   

    for i in range(len(predictions)):
        true_label = true_labels[i]
        predicted_label = predictions[i]

        if true_label != predicted_label:
            most_wrong_preds.append((i, true_label, predicted_label))

    for idx, true_label, predicted_label in most_wrong_preds[:10]:
        print(f"Index: {idx}, True Label: {true_label}, Predicted Label: {predicted_label}")


def plot_f1_scores(true_labels, predicted_labels, class_names, sklearn_acc, figure_size=(15,25)):
    class_f1_scores = {}
    for k, v in classification_report_dict.items():
        if k == "accuracy": 
            break
    else:
        class_f1_scores[class_names[int(k)]] = v["f1-score"]
        
    fig, ax = plt.subplots(figsize=figure_size)
    scores = ax.barh(range(len(f1_scores)), f1_scores["f1-score"].values)
    ax.set_yticks(range(len(f1_scores)))
    ax.set_yticklabels(list(f1_scores["class_name"]))
    ax.set_xlabel("f1-score")
    ax.set_title("F1-Scores")
    ax.invert_yaxis();  
    plt.axvline(x=sklearn_acc, linestyle='--', color='r')

    autolabel(scores)

def autolabel(rects): 
    for rect in rects:
        width = rect.get_width()
        ax.text(1.03*width, rect.get_y() + rect.get_height()/1.5,
            f"{width:.2f}",
            ha='center', va='bottom')

def visualize_predictions(test_images, true_labels, predicted_labels, class_names, prediction_probabilities):
    num_samples = len(test_images)
    sample_indices = np.random.choice(num_samples, size=5, replace=False)  

    plt.figure(figsize=(15, 6))
    for i, idx in enumerate(sample_indices):
        plt.subplot(1, 5, i + 1)
        plt.imshow(test_images[idx])
        plt.title(f"Prediction: {class_names[predicted_labels[idx]]}\n"
                  f"Probability: {prediction_probabilities[idx]:.2f}\n"
                  f"Ground Truth: {class_names[true_labels[idx]]}")
    plt.show()

