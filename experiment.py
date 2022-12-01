'''
A module that manages and runs all experiments
'''

import argparse
import os
import pandas as pd

import style_transfer
import style_classifier
import content_classifier

def params_str(params: style_transfer.Params):
    return f'image_size_{params.image_size}_num_steps_{params.num_steps}_style_weight_{params.style_weight}'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--style_images_dir', type=str, default='data/by-artist-4artists-256/test')
    parser.add_argument('--images_per_artist', type=int, default=5)
    parser.add_argument('--content_images_dir', type=str, default='data/by-content/val')
    parser.add_argument('--images_per_class', type=int, default=5)
    parser.add_argument('--output_dir', type=str, default='data/output/style_transfered')
    # parser.add_argument('--cnn', type=str, default='vgg19', choices=['vgg19', 'vgg16']) TODO
    parser.add_argument('--image_size', type=int, default=style_transfer.DEFAULT_IMAGE_SIZE)
    parser.add_argument('--num_steps', type=int, default=300)
    parser.add_argument('--style_weight', type=int, nargs='+', default=1000000)
    parser.add_argument('--content_weight', type=int, default=1)
    parser.add_argument('--start_image', type=style_transfer.StartImage,
                        default=style_transfer.StartImage.content,
                        choices=list(style_transfer.StartImage))
    parser.add_argument('--style_classifier_model_path', type=str, default='saved-models/model_4artists_256.pth')
    parser.add_argument('--content_classifier_model_path', type=str, default='saved-models/content_classifier.pth')
    config = parser.parse_args()

    output_index = []

    estimated_total_images = ((len(os.listdir(config.style_images_dir)) - 1) *
                              config.images_per_artist *
                              (len(os.listdir(config.content_images_dir)) - 1) *
                              config.images_per_class *
                              len(config.style_weight))
    total_images = 0

    for style_weight in config.style_weight:
        params = style_transfer.Params(config.image_size, config.num_steps, style_weight,
                                       config.content_weight, config.start_image)
        print(f'Params: {params}')
        num_artists = 0
        output_dir_base = os.path.join(config.output_dir, params_str(params))
        for artist in sorted(os.listdir(config.style_images_dir)):
            if artist.startswith('.'):
                continue
            per_artist = 0
            num_artists += 1
            output_dir_style = f"{output_dir_base}/style/{artist}"
            if not os.path.exists(output_dir_style):
                os.makedirs(output_dir_style)
            for style_image_id in sorted(os.listdir(f'{config.style_images_dir}/{artist}')):
                if style_image_id.startswith('.'):
                    continue

                style_image_path = f"{config.style_images_dir}/{artist}/{style_image_id}"

                num_classes = 0
                per_class = 0
                for content_image_class in sorted(os.listdir(config.content_images_dir)):
                    if content_image_class.startswith('.'):
                        continue

                    num_classes += 1
                    output_dir_content = f"{output_dir_base}/content/{content_image_class}"
                    if not os.path.exists(output_dir_content):
                        os.makedirs(output_dir_content)

                    for content_image_id in sorted(os.listdir(f'{config.content_images_dir}/{content_image_class}')):
                        if content_image_id.startswith('.'):
                            continue
                        content_image_path = f"{config.content_images_dir}/{content_image_class}/{content_image_id}"
                        output_name = f'{artist}_{style_image_id}_{content_image_class}_{content_image_id}'

                        style_transfer.run(content_image_path, style_image_path, output_name, output_dir_style, output_dir_content, params)

                        total_images += 1
                        print(f'Processed {total_images} images, overall progress: {total_images / estimated_total_images * 100:.2f}%')
                        per_class += 1
                        if per_class >= config.images_per_class:
                            break
                per_artist += 1
                if per_artist >= config.images_per_artist:
                    break
        style_accuracy, content_accuracy = classify(output_dir_base, config.style_classifier_model_path, config.content_classifier_model_path)
        output_index.append({
            'path': output_dir_base,
            'image_size': params.image_size,
            'num_steps': params.num_steps,
            'style_weight': style_weight,
            'content_weight': params.content_weight,
            'start_image': params.start_image.name,
            'images_per_artist': per_artist,
            'images_per_class': per_class,
            'num_artists': num_artists,
            'num_classes': num_classes,
            'total_images': total_images,
            'style_accuracy': style_accuracy,
            'content_accuracy': content_accuracy
        })
        pd.DataFrame(output_index).to_csv(f'{config.output_dir}/results.csv', index=False)

def classify(output_dir_base, style_classifier_model_path, content_classifier_model_path):
        batch_size = 16

        style_classification_path = os.path.join(output_dir_base, 'style')
        style_accuracy = style_classifier.classify_and_report(style_classifier_model_path,
            style_classification_path, batch_size)

        content_classification_path = os.path.join(output_dir_base, 'content')
        content_accuracy = content_classifier.classify_and_report(content_classifier_model_path,
            content_classification_path, batch_size)

        return style_accuracy, content_accuracy


if __name__ == '__main__':
    main()
