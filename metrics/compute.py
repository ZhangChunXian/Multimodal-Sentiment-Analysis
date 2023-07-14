from sklearn.metrics import precision_recall_fscore_support
import numpy as np

def calculate_f1(true_labels, predicted_labels):
    predicted_classes = np.argmax(predicted_labels, axis=-1) 
    real_classes = true_labels
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(real_classes, predicted_classes, average='weighted')
    return f_macro
