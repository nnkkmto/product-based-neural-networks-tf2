import tensorflow as tf

from dao import MovieLens100kDao as learning_dao
from models import PNN

dao = learning_dao()
dao.build()
X_train, X_test, y_train, y_test = dao.fetch_dataset()
features_info = dao.features_info

auc = tf.keras.metrics.AUC(num_thresholds=1000)
optimizer = tf.keras.optimizers.Adam(lr=0.01, decay=0.1)
model = PNN(features_info)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[auc])
model.fit(x=X_train, y=y_train, epochs=100, batch_size=100000, validation_split=0.2)
model.evaluate(x=X_test, y=y_test)