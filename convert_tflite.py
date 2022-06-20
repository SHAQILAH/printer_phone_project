import tensorflow as tf 

# this points to the directory where you export the model 
saved_model_dir = "/home/SHAQILAH/printer_phone_project/exported_models/centernet_resnet101_v1_fpn_512x512_coco17_tpu-8/run1/saved_model"
tflite_model_dir = "/home/SHAQILAH/printer_phone_project"

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir) # path to the SavedModel directory

tflite_model = converter.convert()

# Save the TF-lite model.
with open('/home/shaqilah_celeste/model.tflite', 'wb') as f:
  f.write(tflite_model)
