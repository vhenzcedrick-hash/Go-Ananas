import tensorflow as tf

# Load your existing full model
model = tf.keras.models.load_model("go_ananas_model.h5", compile=False)

# Save just the weights into a clean new file
model.save_weights("go_ananas_weights.h5")
print("Weights extracted successfully!")
