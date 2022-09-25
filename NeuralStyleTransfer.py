from torch.autograd import Variable
import os
import torch
from image_operations import gram_matrix, prepare_img, save_and_maybe_display
from model import prepare_model
from parametersTune import build_loss
from torch.optim import LBFGS

def neural_style_transfer(config):
    '''
    This method uses functions from model.py, parametersTune.py and image_operations.py to implement a fine-tuned VGG-19 architecture that adapts the Style image to a Content Image. 
    The steps are:
    a) preprocess the image to the input size.
    b) define an optimzation function for fine-tuning the vgg19 architecture.
    c) save images based on saving frequency (note this parameter needs to be changed in the function's code: save_and_maybe_display.)

    params:
    config: contains a dictionary with the following keys- content_img_name, style_img_name, content_weight, style_weight, height, tv_weight, content_images_dir, style_images_dir, ouptut_images_dir and img_format.
    '''
    content_img_path = os.path.join(config['content_images_dir'], config['content_img_name'])
    style_img_path = os.path.join(config['style_images_dir'], config['style_img_name'])
    out_dir_name = 'combined_' + os.path.split(content_img_path)[1].split('.')[0] + '_' + os.path.split(style_img_path)[1].split('.')[0]
    dump_path = os.path.join(config['output_img_dir'], out_dir_name)
    os.makedirs(dump_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    content_img = prepare_img(content_img_path, config['height'], device)
    style_img = prepare_img(style_img_path, config['height'], device)
    
    init_img = content_img
    
    optimizing_img = Variable(init_img, requires_grad=True)
    neural_net, content_feature_maps_index_name, style_feature_maps_indices_names = prepare_model(device)
    print(f'Using VGG19 in the optimization procedure.')
    content_img_set_of_feature_maps = neural_net(content_img)
    style_img_set_of_feature_maps = neural_net(style_img)
    target_content_representation = content_img_set_of_feature_maps[content_feature_maps_index_name[0]].squeeze(axis=0)
    target_style_representation = [gram_matrix(x) for cnt, x in enumerate(style_img_set_of_feature_maps) if cnt in style_feature_maps_indices_names[0]]
    target_representations = [target_content_representation, target_style_representation]
    num_of_iterations = 1000
    
    optimizer = LBFGS((optimizing_img,), max_iter=num_of_iterations, line_search_fn='strong_wolfe')
    cnt = 0

    def closure():
        nonlocal cnt
        if torch.is_grad_enabled():
            optimizer.zero_grad()
        total_loss, content_loss, style_loss, _ = build_loss(neural_net, optimizing_img, target_representations, content_feature_maps_index_name[0], style_feature_maps_indices_names[0], config)
        if total_loss.requires_grad:
            total_loss.backward()
        with torch.no_grad():
            print(f'iteration: {cnt:03}, total loss={total_loss.item():12.4f}, content_loss={config["content_weight"] * content_loss.item():12.4f}, style loss={config["style_weight"] * style_loss.item():12.4f}')
            save_and_maybe_display(optimizing_img, dump_path, config, cnt, num_of_iterations)

        cnt += 1
        return total_loss
        
    optimizer.step(closure)
    return dump_path

#Code to initialize the parameters and call the Neural Style Transfer function.

PATH = ''
CONTENT_IMAGE = 'c1.jpg'
STYLE_IMAGE = 's1.jpg'


default_resource_dir = os.path.join(PATH, 'data')
content_images_dir = os.path.join(default_resource_dir, 'content-images')
style_images_dir = os.path.join(default_resource_dir, 'style-images')
output_img_dir = os.path.join(default_resource_dir, 'output-images')
img_format = (4, '.jpg')



# import os
# assign directory
# directory = 'files'
 
# iterate over files in
# that directory

for contentfilename in list(['c6.jpg']):
    for stylefilename in list(['c7.jpg']):
        optimization_config = {'content_img_name': contentfilename, 'style_img_name': stylefilename, 'height': 400, 'content_weight': 100000.0, 'style_weight': 30000.0, 'tv_weight': 1.0}
        optimization_config['content_images_dir'] = content_images_dir
        optimization_config['style_images_dir'] = content_images_dir
        optimization_config['output_img_dir'] = output_img_dir
        optimization_config['img_format'] = img_format

        results_path = neural_style_transfer(optimization_config)

