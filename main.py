import sklearn.datasets
from utils.preprocessing import data_loader, separate, preprocess
from utils.visualize import plot_loss, plot_accuracy
from model.model import create_model
data = sklearn.datasets.load_breast_cancer()
df = data_loader(data)
X_train, X_test, y_train, y_test = separate(df, 'label', 0.3, 42)
X_train_std, X_test_std = preprocess(X_train, X_test)
model = create_model()
history = model.fit(X_train_std, y_train, epochs=10, batch_size=10, validation_split=0.1)
plot_accuracy(history)
plot_loss(history)
loss, accuracy=model.evaluate(X_test_std, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')
