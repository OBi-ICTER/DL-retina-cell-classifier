import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from itertools import cycle
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay
from sklearn.metrics import PrecisionRecallDisplay
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import interp

def predict(img_path, model, image_size, directory):
        test =tf.keras.preprocessing.image_dataset_from_directory(directory=directory, labels='inferred', label_mode='int', image_size=image_size)
        class_names=test.class_names
        model = tf.keras.models.load_model(model)
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=image_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        plt.figure()
        plt.imshow(img)
        plt.title( "This is {} cell with a {:.2f} percent confidence." .format(class_names[np.argmax(score)], 100 * np.max(score)))
        plt.show()

def parameters(model_path, image_size, directory, loop):
        parameters=[0,0,0,0]  
        y_label=[]
        y_pred=[]
        y_pred1=[]
        #loop=10

        for a in range(loop):

            test =tf.keras.preprocessing.image_dataset_from_directory(
                directory=directory, labels='inferred', label_mode='int',
                batch_size=32, image_size=image_size) 

            class_names = test.class_names
            
            AUTOTUNE = tf.data.experimental.AUTOTUNE

            test =test.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

            model = tf.keras.models.load_model(model_path)
            
            for i in range(3):
                image, label = test.as_numpy_iterator().next()
                predictions = model.predict_on_batch(image)
            
            for x in range(len(label)):
                y = label.take(x)
                y_label.append(y)
            predictions = tf.nn.softmax(predictions)
            
            predictions1= np.argmax(predictions, axis=-1)
            
            for x in range(len(predictions)):
                y=predictions.numpy()[x]
                y_pred.append(y)
            for x in range(len(predictions1)):
                y=predictions1.take(x)
                y_pred1.append(y)
            
            matrix=[]
            for y in range(len(class_names)):
                TP = 0
                TN = 0
                FP = 0
                FN = 0
                for x in range(len(label)): 
                    if (y == label.take(x)) and (label.take(x)==predictions1.take(x)):
                        TP=TP+1
                    elif (y != label.take(x)) and (y!=predictions1.take(x)):
                        TN= TN+1
                    elif (y != label.take(x)) and (predictions1.take(x)==y):
                        FP=FP+1
                    elif (y == label.take(x)) and (predictions1.take(x)!=y):
                        FN=FN+1
                y = (TP, TN, FP, FN)
                matrix.append(y)
            parameters=np.add(matrix, parameters)
            
            tab=[0,0,0,0]
            param=[]
            for i in range(len(class_names)):  
                precision=round(parameters[i][0] / (parameters[i][0] + parameters[i][3]),2)
                if ((parameters[i][0] + parameters[i][2])<=0):
                    recall = 0.00
                else:     
                    recall= round(parameters[i][0] / (parameters[i][0] + parameters[i][2]),2)
                if ((precision and recall) <= 0):
                    F1=0.00
                else:
                    F1 = round((2*(precision * recall) / (precision + recall)),2)
                Acc=round((parameters[i][0]+parameters[i][1])/(parameters[i][0]+parameters[i][1]+parameters[i][2]+parameters[i][3]),2)
                a=(precision, recall, F1, Acc)
                param.append(a) 
            tab=np.add(param, tab)
        '''
        fig, ax = plt.subplots()
        # hide axes
        fig.patch.set_visible(False)
        ax.axis('off')
        ax.axis('tight')
        table = pd.DataFrame(parameters, index=class_names, columns=['TP', 'TN', 'FP', 'FN'])
        ax.table(cellText=table.values, cellLoc='center', colLabels=table.columns, rowLabels=table.index, loc='center')
        plt.title('Parameters')
        plt.show()

        fig, ax = plt.subplots()
        # hide axes
        fig.patch.set_visible(False)
        ax.axis('off')
        ax.axis('tight')
        table = pd.DataFrame(tab, index=class_names, columns=['Precision', 'Recall', 'F1', 'accuracy'])
        ax.table(cellText=table.values, cellLoc='center', colLabels=table.columns, rowLabels=table.index, loc='center')
        plt.title('Metrics')
        plt.show()
        '''
        return y_label, y_pred, y_pred1, class_names 


def report(label, pred1):
    #y_pred=np.asarray(y_pred)
    x = classification_report(label, pred1, output_dict=True)
    fig, ax = plt.subplots()
    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    table = pd.DataFrame(x)
    table = np.round(table, decimals=2)
    ax.table(cellText=table.values, cellLoc='center', colLabels=table.columns, rowLabels=table.index, loc='center')
    plt.title('Report')
    plt.show()

def conf_matrix(label, pred1):
    ConfusionMatrixDisplay.from_predictions(label, pred1, cmap='Blues')
    plt.show()

def roc(y_label, y_pred, class_names):
    y_pred=np.asarray(y_pred)
    classes = [x for x in range(len(class_names))]
    y_label = label_binarize(y_label, classes=classes)
    n_classes = y_label.shape[1]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_label[:,i], y_pred[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_label.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    colors = cycle(["purple", "orange", "blue","red","green","yellow","cyan","navy"])
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=2,
            label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Some extension of Receiver operating characteristic to multiclass")
    plt.legend(loc="lower right")
    plt.show()

def precision_recall(y_label, y_pred, class_names):
    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    y_pred=np.asarray(y_pred)
    classes = [x for x in range(len(class_names))]
    y_label = label_binarize(y_label, classes=classes)
    n_classes = y_label.shape[1]
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_label[:, i], y_pred[:, i])
        average_precision[i] = average_precision_score(y_label[:, i], y_pred[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        y_label.ravel(), y_pred.ravel()
    )
    average_precision["micro"] = average_precision_score(y_label, y_pred, average="micro")

    display = PrecisionRecallDisplay(
        recall=recall["micro"],
        precision=precision["micro"],
        average_precision=average_precision["micro"],
    )
    display.plot()
    _ = display.ax_.set_title("Micro-averaged over all classes")

    colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal"])

    _, ax = plt.subplots(figsize=(7, 8))

    f_scores = np.linspace(0.2, 0.8, num=4)
    lines, labels = [], []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
        plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

    display = PrecisionRecallDisplay(
        recall=recall["micro"],
        precision=precision["micro"],
        average_precision=average_precision["micro"],
    )
    display.plot(ax=ax, name="Micro-average precision-recall", color="gold")

    for i, color in zip(range(n_classes), colors):
        display = PrecisionRecallDisplay(
            recall=recall[i],
            precision=precision[i],
            average_precision=average_precision[i],
        )
        display.plot(ax=ax, name=f"Precision-recall for class {i}", color=color)

    # add the legend for the iso-f1 curves
    handles, labels = display.ax_.get_legend_handles_labels()
    handles.extend([l])
    labels.extend(["iso-f1 curves"])
    # set the legend and the axes
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(handles=handles, labels=labels, loc="best")
    ax.set_title("Extension of Precision-Recall curve to multi-class")
    plt.show()


def param_metrics(class_names):
    fig, ax = plt.subplots()
    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    table = pd.DataFrame(parameters, index=class_names, columns=['TP', 'TN', 'FP', 'FN'])
    ax.table(cellText=table.values, cellLoc='center', colLabels=table.columns, rowLabels=table.index, loc='center')
    plt.title('Parameters')
    plt.show()

    tab=[0,0,0,0]
    param=[]
    for i in range(len(class_names)):  
        precision=round(parameters[i][0] / (parameters[i][0] + parameters[i][3]),2)
        if ((parameters[i][0] + parameters[i][2])<=0):
            recall = 0.00
        else:     
            recall= round(parameters[i][0] / (parameters[i][0] + parameters[i][2]),2)
        if ((precision and recall) <= 0):
            F1=0.00
        else:
            F1 = round((2*(precision * recall) / (precision + recall)),2)
        Acc=round((parameters[i][0]+parameters[i][1])/(parameters[i][0]+parameters[i][1]+parameters[i][2]+parameters[i][3]),2)
        a=(precision, recall, F1, Acc)
        param.append(a)
    tab=np.add(param, tab)

    fig, ax = plt.subplots()
    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    table = pd.DataFrame(tab, index=class_names, columns=['Precision', 'Recall', 'F1', 'accuracy'])
    ax.table(cellText=table.values, cellLoc='center', colLabels=table.columns, rowLabels=table.index, loc='center')
    plt.title('Metrics')
    plt.show()

'''
class Evaluate:
    def __init__(self, path, model, image_size, loop):
        self.path=path
        self.model=model
        self.image_size=image_size
        self.loop=loop

    x, y, z = parameters()
    report(x, z)
    conf_matrix(x, z)
    roc(x,y)
    precision_recall(x,y)
'''


'''
def param_metrics():
fig, ax = plt.subplots()
# hide axes
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')
table = pd.DataFrame(parameters, index=class_names, columns=['TP', 'TN', 'FP', 'FN'])
ax.table(cellText=table.values, cellLoc='center', colLabels=table.columns, rowLabels=table.index, loc='center')
plt.title('Parameters')
plt.show()

tab=[0,0,0,0]
param=[]
for i in range(len(class_names)):  
 precision=round(parameters[i][0] / (parameters[i][0] + parameters[i][3]),2)
 if ((parameters[i][0] + parameters[i][2])<=0):
  recall = 0.00
 else:     
  recall= round(parameters[i][0] / (parameters[i][0] + parameters[i][2]),2)
 if ((precision and recall) <= 0):
  F1=0.00
 else:
  F1 = round((2*(precision * recall) / (precision + recall)),2)
 Acc=round((parameters[i][0]+parameters[i][1])/(parameters[i][0]+parameters[i][1]+parameters[i][2]+parameters[i][3]),2)
 a=(precision, recall, F1, Acc)
 param.append(a)
tab=np.add(param, tab)

fig, ax = plt.subplots()
# hide axes
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')
table = pd.DataFrame(tab, index=class_names, columns=['Precision', 'Recall', 'F1', 'accuracy'])
ax.table(cellText=table.values, cellLoc='center', colLabels=table.columns, rowLabels=table.index, loc='center')
plt.title('Metrics')
plt.show()
'''