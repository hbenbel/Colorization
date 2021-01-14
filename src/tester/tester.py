import os

import matplotlib.pyplot as plt
import numpy as np
from skimage import color
from tqdm import tqdm


class DCGANTester:
    def __init__(self, g_model, config, test_data_loader, device):
        self.device = device
        self.g_model = g_model
        self.test_data_loader = test_data_loader
        self.save_path = config['save_path']
        self.cpt = 0

    def _postprocess(self, in_image, prediction, min_val, max_val):
        in_image = in_image[0, :, :, :]
        prediction = prediction[0, :, :, :]

        predicted_image = np.vstack([in_image, prediction])
        predicted_image = np.moveaxis(predicted_image, 0, -1)
        predicted_image += 1
        predicted_image /= 2
        predicted_image *= (max_val - min_val)
        predicted_image += min_val
        predicted_image = color.lab2rgb(predicted_image)

        return predicted_image

    def _postprocess_all(self, in_images, predictions, min_vals, max_vals):
        data = list(zip(in_images, predictions, min_vals, max_vals))
        return list(
                    map(
                        lambda x: self._postprocess(x[0], x[1], x[2], x[3]),
                        data
                    )
                )

    def _save_result(self, original_image, predicted_image):
        test_save_path = os.path.join(self.save_path, "test")
        if not os.path.exists(test_save_path):
            os.makedirs(test_save_path)

        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(original_image)

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(predicted_image)
        fig.savefig(os.path.join(test_save_path, str(self.cpt) + '.png'))

        self.cpt += 1

    def _save_results(self, original_images, predicted_images):
        data = list(zip(original_images, predicted_images))
        return list(map(lambda x: self._save_result(x[0], x[1]), data))

    def test(self):
        for images, min_vals, max_vals in tqdm(
                                                self.testing_data_loader,
                                                desc="Validation"
                                            ):
            l_images = images[:, 0, :, :].unsqueeze(1).double()
            ab_images = images[:, 1:, :, :].double()

            original_images = self._postprocess_all(
                                in_images=l_images,
                                predictions=ab_images,
                                min_vals=min_vals,
                                max_vals=max_vals
                            )

            predicted_ab_images = self.g_model(
                                    l_images.to(self.device)
                                ).cpu().data.numpy()

            predicted_images = self._postprocess_all(
                                    in_images=l_images,
                                    predictions=predicted_ab_images,
                                    min_vals=min_vals,
                                    max_vals=max_vals
                                )

            self._save_results(original_images, predicted_images)
