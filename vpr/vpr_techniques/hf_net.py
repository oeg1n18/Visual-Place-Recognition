import cv2
import numpy as np
from pathlib import Path

from hfnet.settings import EXPER_PATH
from notebooks.utils import plot_images, plot_matches, add_frame

import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
tf.contrib.resampler  # import C++ op

query_idx = 1  # also try with 2 and 3
read_image = lambda n: cv2.imread('doc/demo/' + n)[:, :, ::-1]
image_query = read_image(f'query{query_idx}.jpg')
images_db = [read_image(f'db{i}.jpg') for i in range(1, 5)]

plot_images([image_query] + images_db, dpi=50)

class HFNet:
    def __init__(self, model_path, outputs):
        self.session = tf.Session()
        self.image_ph = tf.placeholder(tf.float32, shape=(None, None, 3))

        net_input = tf.image.rgb_to_grayscale(self.image_ph[None])
        tf.saved_model.loader.load(
            self.session, [tag_constants.SERVING], str(model_path),
            clear_devices=True,
            input_map={'image:0': net_input})

        graph = tf.get_default_graph()
        self.outputs = {n: graph.get_tensor_by_name(n + ':0')[0] for n in outputs}
        self.nms_radius_op = graph.get_tensor_by_name('pred/simple_nms/radius:0')
        self.num_keypoints_op = graph.get_tensor_by_name('pred/top_k_keypoints/k:0')

    def inference(self, image, nms_radius=4, num_keypoints=1000):
        inputs = {
            self.image_ph: image[..., ::-1].astype(np.float),
            self.nms_radius_op: nms_radius,
            self.num_keypoints_op: num_keypoints,
        }
        return self.session.run(self.outputs, feed_dict=inputs)


model_path = Path(EXPER_PATH, 'saved_models/hfnet')
outputs = ['global_descriptor', 'keypoints', 'local_descriptors']
hfnet = HFNet(model_path, outputs)


db = [hfnet.inference(i) for i in images_db]
global_index = np.stack([d['global_descriptor'] for d in db])
query = hfnet.inference(image_query)


def compute_distance(desc1, desc2):
    # For normalized descriptors, computing the distance is a simple matrix multiplication.
    return 2 * (1 - desc1 @ desc2.T)



nearest = np.argmin(compute_distance(query['global_descriptor'], global_index))

disp_db = [add_frame(im, (0, 255, 0)) if i == nearest else im
           for i, im in enumerate(images_db)]
plot_images([image_query] + disp_db, dpi=50)


def match_with_ratio_test(desc1, desc2, thresh):
    dist = compute_distance(desc1, desc2)
    nearest = np.argpartition(dist, 2, axis=-1)[:, :2]
    dist_nearest = np.take_along_axis(dist, nearest, axis=-1)
    valid_mask = dist_nearest[:, 0] <= (thresh**2)*dist_nearest[:, 1]
    matches = np.stack([np.where(valid_mask)[0], nearest[valid_mask][:, 0]], 1)
    return matches


matches = match_with_ratio_test(query['local_descriptors'],
                                db[nearest]['local_descriptors'], 0.8)

plot_matches(image_query, query['keypoints'],
             images_db[nearest], db[nearest]['keypoints'],
             matches, color=(0, 1, 0), dpi=50)