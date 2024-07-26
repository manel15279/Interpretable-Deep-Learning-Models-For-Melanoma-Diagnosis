import numpy as np
import tensorflow as tf
import cv2

def generate_grad_cam_heatmap(m, model, img_array):
    with tf.GradientTape() as tape:
        pred = m.predict(img_array)
        if pred[0][0] > 0.5:
            c = 'mel'
        else:
            c = 'not_mel'
        cls = np.argmax(pred)
        inputs = tf.cast(img_array, tf.float32)
        tape.watch(inputs)
        conv_outputs, predictions = model(inputs)
        # Since it's a binary classification, we use the output directly
        prob_melanoma = predictions[0, 0]
    grads = tape.gradient(prob_melanoma, conv_outputs)
    guided_grads = tf.cast(conv_outputs > 0, 'float32') * tf.cast(grads > 0, 'float32') * grads
    conv_outputs = conv_outputs[0]
    guided_grads = guided_grads[0]
    weights = tf.reduce_mean(guided_grads, axis=(0, 1))
    cam = np.zeros(conv_outputs.shape[0:2], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * conv_outputs[:, :, i]
    cam = cv2.resize(cam.numpy(), (256, 256))
    
    cam = np.maximum(cam, 0)
    heatmap = cam / cam.max()
    
    hmconf = heatmap if c == 'not_mel' else 1-heatmap
    heatmap = np.uint8(255 * heatmap)
    if c == 'mel':
        heatmap = cv2.applyColorMap(255 - heatmap, cv2.COLORMAP_JET)  # Invert the heatmap colors
    else:
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # Invert the heatmap colors

    return hmconf, heatmap