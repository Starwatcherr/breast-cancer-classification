import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

class LinearClassification:
    def __init__(self, learning_rate=0.001, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None # weights, empty on first initialization, gets value in fit()
        self.bias = None    # bias, empty on first initialization, gets value in fit()
        self.mean = None    # mean, empty on first initialization, gets value in fit()
        self.std = None     # standard, empty on first initialization, gets value in fit()

    def fit(self, X, y):
        # pandas -> numpy
        X = X.to_numpy()
        y = y.to_numpy()

        self.loss_history_fit = []

        self.mean = X.mean(axis=0)  # mean of each feature, learned from training data, __init__ actualized
        self.std = X.std(axis=0)  # standard deviation of each feature, learned from training data, __init__ actualized

        X_scaled = (X - self.mean) / self.std   # standardize features

        n_features = X_scaled.shape[1]  # number of features
        n_samples = X_scaled.shape[0]   # number of patients

        self.weights = np.zeros(n_features) # initialize weights
        self.bias = 0.0

        for epoch in range(self.epochs):
            predictions = self.linear_output(X_scaled)
            error = y - predictions
            self.loss_history_fit.append(0.5 * np.sum(error ** 2))

            weight_gradient = - np.dot(X_scaled.T, error) / n_samples
            bias_gradient = - np.sum(error) / n_samples
            self.weights = self.weights - self.learning_rate * weight_gradient
            self.bias = self.bias - self.learning_rate * bias_gradient

            # debug
            #if epoch % 10 == 0:
            #    print(f"Epoch {epoch}: loss = {0.5 * np.sum(error ** 2):.2f}")

    def predict(self, X):
        # pandas -> numpy
        X = X.to_numpy()
        X_scaled = (X - self.mean) / self.std

        scores = self.linear_output(X_scaled)
        predictions = np.where(scores >= 0, 1, -1)    # above or 0=1 below 0=-1, no np.sign() because np.sign(0) = 0

        return predictions

    def linear_output(self, X_scaled):  # own function because of reusing in prediction
        return np.dot(X_scaled, self.weights) + self.bias   # np.dot does matrix multiplication,
                                                            # like f(x) = w * x; + b is bias
def accuracy(predictions, y):
    length = len(y)
    correct_prediction = np.sum(predictions == y)
    accuracy_val = correct_prediction / length
    return accuracy_val

def load_data(filepath):
    data = pd.read_csv(filepath)
    labels = data["diagnosis"]
    features = data.drop(columns=["diagnosis"])
    return features, labels

def plot_data(X_train, y_train):
    benign = X_train[y_train == 1]
    malignant = X_train[y_train == -1]
    X_scaled = (X_train - X_train.mean()) / X_train.std()
    benign_scaled = X_scaled[y_train == 1]
    malignant_scaled = X_scaled[y_train == -1]

    # scatter plots of selected feature pairs
    feature_pairs = [
        ("radius_worst", "texture_worst"),
        ("area_worst", "concavity_worst"),
        ("perimeter_worst", "smoothness_worst"),
    ]
    for x_feature, y_feature in feature_pairs:
        plt.figure()
        plt.scatter(benign[x_feature], benign[y_feature], label="benign")
        plt.scatter(malignant[x_feature], malignant[y_feature], label="malignant")
        plt.legend()
        plt.xlabel(x_feature)
        plt.ylabel(y_feature)
        plt.title(f"Visualization of {x_feature} and {y_feature} in benign and malignant", fontsize=11)
        plt.show()

    # class distribution
    class_counts = y_train.value_counts()
    plt.figure()
    plt.bar(["benign (+1)", "malignant (-1)"],
            [class_counts[1], class_counts[-1]],
            color=["steelblue", "indianred"])
    plt.ylabel("Number of patients")
    plt.title("Class distribution in training set")
    plt.text(0, class_counts[1], class_counts[1], ha="center", va="bottom", fontsize=12)
    plt.text(1, class_counts[-1], class_counts[-1], ha="center", va="bottom", fontsize=12)
    plt.savefig("class_distribution.png", dpi=300, bbox_inches="tight")
    plt.show()

    # discriminative power per feature: standardized mean difference between classes
    mean_diff = (benign.mean() - malignant.mean()).abs()
    pooled_std = X_train.std()
    separation = (mean_diff / pooled_std).sort_values(ascending=True)
    plt.figure(figsize=(8, 8))
    plt.barh(separation.index, separation.values, color="steelblue")
    plt.xlabel("Standardized mean difference", fontsize=14)
    plt.title("Discriminative power per feature", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig("feature_separation.png", dpi=300, bbox_inches="tight")
    plt.show()

    # boxplots per class, all features side by side
    fig, ax = plt.subplots(figsize=(14, 6))
    positions_benign = np.arange(len(X_train.columns)) * 3
    positions_malignant = positions_benign + 1
    bp1 = ax.boxplot(benign_scaled.values, positions=positions_benign, widths=0.8,
                     patch_artist=True, boxprops=dict(facecolor="steelblue"))
    bp2 = ax.boxplot(malignant_scaled.values, positions=positions_malignant, widths=0.8,
                     patch_artist=True, boxprops=dict(facecolor="indianred"))
    ax.set_xticks(positions_benign + 0.5)
    ax.set_xticklabels(X_train.columns, rotation=90, fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylabel("Standardized value", fontsize=14)
    ax.set_title("Feature distributions per class (standardized)", fontsize=16)
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ["benign", "malignant"], fontsize=12)
    plt.savefig("boxplots_per_class.png", dpi=300, bbox_inches="tight")
    plt.show()

    # boxplot of all standardized features, shows spread and outliers
    plt.figure(figsize=(14, 6))
    plt.boxplot(X_scaled.values, tick_labels=X_train.columns)
    plt.xticks(rotation=90)
    plt.ylabel("Standardized value")
    plt.title("Distribution of standardized features")
    plt.savefig("boxplots_all_features.png", dpi=300, bbox_inches="tight")
    plt.show()

    # correlation heatmap of all 30 features
    correlation_matrix = X_train.corr()
    plt.figure(figsize=(10, 10))
    plt.imshow(correlation_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    cbar = plt.colorbar(label="Correlation coefficient")
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("Correlation coefficient", fontsize=14)
    plt.xticks(range(len(X_train.columns)), X_train.columns, rotation=90, fontsize=12)
    plt.yticks(range(len(X_train.columns)), X_train.columns, fontsize=12)
    plt.title("Feature correlation matrix", fontsize=16)
    plt.savefig("correlation_heatmap.png", dpi=300, bbox_inches="tight")
    plt.show()

def plot_loss(model):
    plt.plot(model.loss_history_fit)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training loss over epochs")
    plt.show()

def plot_confusion_matrix(test_predictions, y_test):
    y_test = y_test.to_numpy()

    correct_malignant = np.sum((test_predictions == -1) & (y_test == -1))
    correct_benign = np.sum((test_predictions == 1) & (y_test == 1))
    false_benign = np.sum((test_predictions == 1) & (y_test == -1))
    false_malignant = np.sum((test_predictions == -1) & (y_test == 1))

    matrix = np.array([[correct_benign, false_malignant],
                       [false_benign, correct_malignant]])

    plt.imshow(matrix, cmap="Blues")
    plt.colorbar()

    plt.xticks([0, 1], ["benign", "malignant"])
    plt.yticks([0, 1], ["benign", "malignant"])
    plt.xlabel("Prediction")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.text(0, 0, correct_benign, ha="center", va="center", fontsize=20)
    plt.text(1, 0, false_malignant, ha="center", va="center", fontsize=20)
    plt.text(0, 1, false_benign, ha="center", va="center", fontsize=20)
    plt.text(1, 1, correct_malignant, ha="center", va="center", fontsize=20)

    plt.show()

def analyze_data(features, labels):
    print("Class distribution: ", labels.value_counts()) # class imbalance noticeable
    print(f"\nFeature Statistics: \n{features.describe()}")  # great difference between values, standardization necessary

def tune_model(X_train, y_train):
    # sklearn's split only to split the data (here) into 80% train and 20% test. 42
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state = 42)

    learning_rates = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    epoch_options = [100, 250, 500, 1000, 1500, 2000]

    best_accuracy = 0
    best_settings = None

    for lr in learning_rates:
        for ep in epoch_options:
            model = LinearClassification(learning_rate=lr, epochs=ep)
            model.fit(X_tr, y_tr)
            val_predictions = model.predict(X_val)
            val_acc = accuracy(val_predictions, y_val)
            print(f"lr={lr}, epochs={ep}: val accuracy = {val_acc:.4f}")
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                best_settings = (lr, ep)

    print(f"\nBest accuracy: {best_accuracy}\nBest settings: {best_settings}")
    return best_settings

if __name__ == "__main__":
    pd.set_option("display.max_columns", None)  # global value for pandas to use everywhere
    plt.rcParams["figure.autolayout"] = True    # global value for matplotlib to use tight_layout() everywhere

    X_train, y_train = load_data("breast-cancer_train.csv")
    X_test, y_test = load_data("breast-cancer_test.csv")

    #analyze_data(X_train, y_train)

    plot_data(X_train, y_train)

    best_settings = tune_model(X_train, y_train)    # best_setting not further used bc usage is already in tune_model

    # tune_model gives a single "best" combination, but the result depends on the random
    # train/val split (small validation set, no clear optimum). A wide range of small learning
    # rates reaches ~97% stably, so we deliberately choose a robust setting from that region
    # instead of the noisy winner.
    final_model = LinearClassification(learning_rate=0.01, epochs=1000)
    final_model.fit(X_train, y_train)

    test_predictions = final_model.predict(X_test)
    print("Final test accuracy:", accuracy(test_predictions, y_test))

    plot_loss(final_model)

    plot_confusion_matrix(test_predictions, y_test)
