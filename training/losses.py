import tensorflow as tf

def compute_loss(real, generated, edge_layer, edge_weight=5.0):
    mae = tf.reduce_mean(tf.abs(real - generated))
    real_edges = edge_layer(real)
    gen_edges = edge_layer(generated)
    edge_loss = tf.reduce_mean(tf.abs(real_edges - gen_edges))
    ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(real, generated, max_val=1.0))
    return 1.0 * mae + edge_weight * edge_loss + 2.0 * ssim_loss

def gradient_penalty(discriminator, real, fake):
    alpha = tf.random.uniform(shape=[tf.shape(real)[0], 1, 1, 1])
    interpolated = alpha * real + (1 - alpha) * fake
    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        pred = discriminator(interpolated)
    grads = gp_tape.gradient(pred, [interpolated])[0]
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
    return tf.reduce_mean((norm - 1.0) ** 2)
